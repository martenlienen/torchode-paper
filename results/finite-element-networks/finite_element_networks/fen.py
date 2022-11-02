from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import skfem
import torch
import torch.nn as nn
from cachetools import cachedmethod
from skfem.assembly.form.form import FormExtraParams
from skfem.helpers import grad
from torch_scatter import scatter
from torchode_utils.timing import cuda_timing

from .data import STBatch, TimeEncoder
from .domain import Domain
from .mlp import MLP
from .ode_solver import ODESolver


@skfem.BilinearForm
def mass_form(u, v, w):
    return u * v


def lumped_mass_matrix(domain: Domain) -> torch.Tensor:
    """Compute the diagonal of the lumped mass matrix."""

    A = skfem.asm(mass_form, domain.basis)
    return torch.from_numpy(np.array(A.sum(axis=1))[:, 0])


def assemble_linear_cell_contributions(
    basis: skfem.assembly.Basis, form: skfem.assembly.LinearForm
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate a linear form over all basis functions over all mesh cells.

    This is adapated from `skfem.LinearForm._assemble`.

    The returned rows tell you which coefficient from each cell contributes to which node.
    """

    def kernel(v, w, dx):
        return np.sum(form.form(*v, w) * dx, axis=1)

    vbasis = basis

    nt = vbasis.nelems
    dx = vbasis.dx
    w = FormExtraParams(vbasis.default_parameters())

    # initialize COO data structures
    data = np.zeros((vbasis.Nbfun, nt), dtype=np.float64)
    rows = np.zeros((vbasis.Nbfun, nt), dtype=np.int64)

    for i in range(vbasis.Nbfun):
        rows[i] = vbasis.element_dofs[i]
        data[i] = kernel(vbasis.basis[i], w, dx)

    assert np.all(rows == vbasis.mesh.t)

    rows = torch.from_numpy(rows).long()
    data = torch.from_numpy(data).float()

    return data, rows


def assemble_free_form_parts(domain: Domain):
    """Compute the free-form inner products of test functions."""

    @skfem.LinearForm
    def test_function(v, w):
        return v

    return assemble_linear_cell_contributions(domain.basis, test_function)


def assemble_bilinear_cell_contributions(
    basis: skfem.assembly.Basis, form: skfem.assembly.BilinearForm
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,]:
    """Integrate a bilinear form over all pairs of basis functions over all mesh cells.

    This is adapated from `skfem.BilinearForm._assemble`.

    In the output shapes, `j` is the basis function associated with the trial function u
    and `i` is associated with the test function v.
    """

    def kernel(u, v, w, dx):
        return np.sum(form.form(*u, *v, w) * dx, axis=1)

    ubasis = vbasis = basis

    nt = ubasis.nelems
    dx = ubasis.dx
    wdict = FormExtraParams(ubasis.default_parameters())

    # initialize COO data structures
    data = np.zeros((ubasis.Nbfun, vbasis.Nbfun, nt), dtype=np.float64)
    rows = np.zeros((ubasis.Nbfun, vbasis.Nbfun, nt), dtype=np.int64)
    cols = np.zeros((ubasis.Nbfun, vbasis.Nbfun, nt), dtype=np.int64)

    # loop over the indices of local stiffness matrix
    for j in range(ubasis.Nbfun):
        for i in range(vbasis.Nbfun):
            rows[j, i] = vbasis.element_dofs[i]
            cols[j, i] = ubasis.element_dofs[j]
            data[j, i] = kernel(ubasis.basis[j], vbasis.basis[i], wdict, dx)

    rows = torch.from_numpy(rows).long()
    cols = torch.from_numpy(cols).long()
    data = torch.from_numpy(data).float()

    return data, rows, cols


def assemble_convection_parts(domain: Domain):
    """Compute the convection inner products of trial and test function for each pair of
    cell vertices.
    """

    parts = []
    for i in range(domain.dim):

        @skfem.BilinearForm
        def convection_component(u, v, w, i=i):
            return grad(u)[i] * v

        data, rows, cols = assemble_bilinear_cell_contributions(
            domain.basis, convection_component
        )
        parts.append(data)

    return -torch.stack(parts, dim=-1), rows, cols


class FENDomainInfo(NamedTuple):
    """Precomputed domain attributes relevant to FENs."""

    @staticmethod
    def from_domain(domain: Domain):
        """Construct a FEN specific domain from a general domain object."""

        n_nodes = len(domain)

        node_pos = torch.from_numpy(domain.x).float()
        triangulation = torch.from_numpy(domain.mesh.t.T)
        vertex_pos: torch.Tensor = node_pos[triangulation]
        cell_centers = vertex_pos.mean(dim=1)
        cell_local_vertex_pos = vertex_pos - cell_centers.unsqueeze(dim=1)

        inverse_lumped_mass_matrix = (1 / lumped_mass_matrix(domain)).float()
        fixed_values_mask = (
            torch.from_numpy(domain.fixed_values_mask)
            if domain.fixed_values_mask is not None
            else None
        )
        free_form_parts = assemble_free_form_parts(domain)
        convection_parts = assemble_convection_parts(domain)

        # Assert that the parts are in the same order as the vertices in the
        # triangulation, so that we can rely on that order implicitly when combining the
        # parts with learned coefficients.
        _, free_form_rows = free_form_parts
        assert torch.all(free_form_rows.T == triangulation)

        return FENDomainInfo(
            n_nodes=n_nodes,
            triangulation=triangulation,
            cell_centers=cell_centers,
            cell_local_vertex_pos=cell_local_vertex_pos,
            inverse_lumped_mass_matrix=inverse_lumped_mass_matrix,
            fixed_values_mask=fixed_values_mask,
            free_form_parts=free_form_parts,
            convection_parts=convection_parts,
            n_vertices=triangulation.shape[-1],
            space_dim=cell_centers.shape[-1],
        )

    n_nodes: int
    triangulation: torch.Tensor
    cell_centers: torch.Tensor
    cell_local_vertex_pos: torch.Tensor
    inverse_lumped_mass_matrix: torch.Tensor
    fixed_values_mask: Optional[torch.Tensor]
    free_form_parts: Tuple[torch.Tensor, torch.Tensor]
    convection_parts: Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
    n_vertices: int
    space_dim: int


class FENQuery(NamedTuple):
    """A prediction query for an FEN model.

    `t` contains the time of `u0` followed by the prediction time steps.
    """

    domain: FENDomainInfo
    t: torch.Tensor
    u0: torch.Tensor
    time_encoder: TimeEncoder

    @property
    def batch_size(self):
        return self.t.shape[0]

    @staticmethod
    def from_batch(batch: STBatch[FENDomainInfo], *, standardize: bool = True):
        u = batch.context_u
        if standardize:
            u = batch.context_standardizer.do(u)
        u0 = u[:, -1]
        t = batch.t[:, batch.context_steps - 1 :]
        return FENQuery(batch.domain_info, t, u0, batch.time_encoder)


class SystemState:
    """The current state of a physical system, possibly batched.

    Attributes
    ----------
    domain
        The domain mesh that the system is discretized on
    t
        The current timestamp
    u
        The node features
    """

    def __init__(
        self,
        domain: FENDomainInfo,
        t: torch.Tensor,
        u: torch.Tensor,
    ):
        self.domain = domain
        self.t = t
        self.u = u
        self._cache = {}

    # Cache the cell features to minimize the backwards graph
    # @cachedmethod(lambda self: self._cache)
    def cell_features(self, stationary: bool, autonomous: bool) -> torch.Tensor:
        """Assemble the feature matrix for each cell."""

        T = self.domain.triangulation
        vertex_pos = self.domain.cell_local_vertex_pos
        vertex_features = self.u[:, T]

        ncells = T.shape[0]
        batch_size = self.u.shape[0]

        # Collect all the information about a cell into a per-cell feature matrix
        cell_features = [
            # eo.repeat(vertex_pos, "c v s -> b c (v s)", b=batch_size),
            vertex_pos.flatten(start_dim=-2).expand(batch_size, -1, -1),
            # eo.rearrange(vertex_features, "b c v f -> b c (v f)"),
            vertex_features.flatten(start_dim=-2),
        ]
        if not stationary:
            cell_pos = self.domain.cell_centers
            # cell_features.insert(0, eo.repeat(cell_pos, "c s -> b c s", b=batch_size))
            cell_features.insert(0, cell_pos.expand(batch_size, -1, -1))
        if not autonomous:
            time = self.t
            # cell_features.insert(0, eo.repeat(time, "b f -> b c f", c=ncells))
            cell_features.insert(0, time.unsqueeze(dim=1).expand(-1, ncells, -1))

        return torch.cat(cell_features, dim=-1)


class PDETerm(nn.Module):
    def forward(self, state: SystemState) -> torch.Tensor:
        pass


class FreeFormTerm(PDETerm):
    """A PDE term that does not make any assumptions on the form of the dynamics."""

    @staticmethod
    def build_coefficient_mlp(
        *,
        n_features: int,
        space_dim: int,
        time_dim: int,
        hidden_dim: int,
        n_layers: int,
        non_linearity,
        stationary: bool,
        autonomous: bool,
    ):
        """Build an MLP to estimate the free-form coefficients with the correct in/out
        dimensions.
        """
        n_vertices = space_dim + 1
        extra_in = 0
        if not stationary:
            extra_in += space_dim
        if not autonomous:
            extra_in += time_dim
        return MLP(
            in_dim=n_vertices * (space_dim + n_features) + extra_in,
            out_dim=n_vertices * n_features,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            non_linearity=non_linearity,
        )

    def __init__(
        self,
        coefficient_mlp: MLP,
        *,
        stationary: bool,
        autonomous: bool,
        zero_init: bool,
    ):
        super().__init__()

        self.coefficient_mlp = coefficient_mlp
        self.stationary = stationary
        self.autonomous = autonomous
        self.zero_init = zero_init

        # The zero (constant) dynamic is often a good first approximation, so we start
        # with that.
        if self.zero_init:
            self.coefficient_mlp[-1].weight.data.zero_()
            self.coefficient_mlp[-1].bias.data.zero_()

    def forward(self, state: SystemState):
        return self.build_messages(state.domain, self.estimate_coefficients(state))

    def estimate_coefficients(self, state: SystemState) -> torch.Tensor:
        """Estimate the free-form coefficients as in Equation (16)."""
        # return eo.rearrange(
        #     self.coefficient_mlp(state.cell_features(self.stationary, self.autonomous)),
        #     "b c (v f) -> b c v f",
        #     v=state.domain.n_vertices,
        # )
        coeffs = self.coefficient_mlp(
            state.cell_features(self.stationary, self.autonomous)
        )
        return coeffs.unflatten(-1, (state.domain.n_vertices, -1))

    def build_messages(
        self,
        domain: FENDomainInfo,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        """Build the free-form messages as n Equation (17)."""
        free_form_data, _ = domain.free_form_parts
        return torch.einsum("vc,bcvf->bcvf", free_form_data, coefficients)


class TransportTerm(PDETerm):
    """A PDE term that models transport through convection."""

    @staticmethod
    def build_flow_field_mlp(
        *,
        n_features: int,
        space_dim: int,
        time_dim: int,
        hidden_dim: int,
        n_layers: int,
        non_linearity,
        stationary: bool,
        autonomous: bool,
    ):
        """Build an MLP to estimate the flow field with the correct in/out dimensions."""
        n_vertices = space_dim + 1
        extra_in = 0
        if not stationary:
            extra_in += space_dim
        if not autonomous:
            extra_in += time_dim
        return MLP(
            in_dim=n_vertices * (space_dim + n_features) + extra_in,
            out_dim=n_features * space_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            non_linearity=non_linearity,
        )

    def __init__(
        self,
        flow_field_mlp: MLP,
        *,
        stationary: bool,
        autonomous: bool,
        zero_init: bool,
    ):
        super().__init__()

        self.flow_field_mlp = flow_field_mlp
        self.stationary = stationary
        self.autonomous = autonomous
        self.zero_init = zero_init

        # The zero (constant) dynamic is often a good first approximation, so we start
        # with that.
        if self.zero_init:
            self.flow_field_mlp[-1].weight.data.zero_()
            self.flow_field_mlp[-1].bias.data.zero_()

    def forward(self, state: SystemState):
        flow_field = self.estimate_flow_field(state)
        return self.build_messages(state.domain, flow_field, state.u)

    def estimate_flow_field(self, state: SystemState) -> torch.Tensor:
        """Estimate the cell-wise velocity field as in Equation (20)."""
        # return eo.rearrange(
        #     self.flow_field_mlp(state.cell_features(self.stationary, self.autonomous)),
        #     "b c (f s) -> b c f s",
        #     s=state.domain.space_dim,
        # )
        flow_field = self.flow_field_mlp(
            state.cell_features(self.stationary, self.autonomous)
        )
        return flow_field.unflatten(-1, (-1, state.domain.space_dim))

    def build_messages(
        self,
        domain: FENDomainInfo,
        flow_field: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """Build the transport messages as in Equation (21)."""

        convection_data, _, convection_cols = domain.convection_parts

        # The contration order was optimized with opt-einsum, but the torch implementation
        # runs faster (according to %timeit), so we use torch.einsum anyway.
        return torch.einsum(
            "jics,bjicf,bcfs -> bcif",
            convection_data,
            u[:, convection_cols, :],
            flow_field,
        )


class FENDynamics(nn.Module):
    def __init__(self, terms: list[PDETerm]):
        super().__init__()

        assert len(terms) > 0
        self.terms = nn.ModuleList(terms)

    def forward(self, state: SystemState):
        # msgs = sum([term(state) for term in self.terms])
        msgs = 0
        for term in self.terms:
            msgs = msgs + term(state)
        du = self._send_msgs(msgs, state.domain)

        # Don't change anything for fixed values by setting the derivative to 0
        fixed_values_mask = state.domain.fixed_values_mask
        if fixed_values_mask is not None:
            du = torch.where(fixed_values_mask, du.new_zeros(1), du)

        return du

    def _send_msgs(self, cell_msgs, domain: FENDomainInfo):
        # Send the messages from each cell to its vertices
        # msgs = eo.rearrange(cell_msgs, "b c v f -> b (c v) f")
        msgs = cell_msgs.flatten(start_dim=1, end_dim=2)
        target = domain.triangulation.ravel()
        received: torch.Tensor = scatter(
            msgs, target, dim=1, reduce="sum", dim_size=domain.n_nodes
        )

        return torch.einsum("bnf,n -> bnf", received, domain.inverse_lumped_mass_matrix)

    @property
    def free_form_terms(self):
        terms: list[FreeFormTerm] = []
        for term in self.terms:
            if isinstance(term, FreeFormTerm):
                terms.append(term)
        return terms

    @property
    def transport_terms(self):
        terms: list[TransportTerm] = []
        for term in self.terms:
            if isinstance(term, TransportTerm):
                terms.append(term)
        return terms


class DynWrapper(nn.Module):
    def __init__(self, dynamics: FENDynamics, time_encoder: TimeEncoder):
        super().__init__()

        self.dynamics = dynamics
        self.time_encoder = time_encoder

        self.times = []

    times: List[Any]

    def forward(self, t, y, args: Any):
        if torch.jit.is_scripting():
            assert isinstance(args, tuple[list[int], FENDomainInfo])

        shape, domain_info = args
        y_ = y.unflatten(dim=1, sizes=shape)

        if not torch.jit.is_scripting():
            with cuda_timing() as time:
                self.times.append(time)

                time = self.time_encoder.encode(t)
                state = SystemState(domain_info, time, y_)
                out = self.dynamics(state)
        else:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            self.times.append(start)
            self.times.append(end)
            stream = torch.cuda.current_stream(t.device)
            start.record(stream)
            time = self.time_encoder.encode(t)
            state = SystemState(domain_info, time, y_)
            out = self.dynamics(state)
            end.record(stream)

        return out.flatten(start_dim=1)


class FEN(nn.Module):
    def __init__(self, dynamics: DynWrapper, ode_solver: ODESolver):
        super().__init__()

        self.dynamics = dynamics
        self.ode_solver = ode_solver

        self.stats = {}

        self.times = []
        self.solver_time = (None, None)

    stats: Dict[str, Any]
    times: List[Any]
    solver_time: Tuple[Any, Any]

    def forward(self, query: FENQuery) -> torch.Tensor:
        shape = query.u0.shape[1:]
        args = (shape, query.domain)
        if not torch.jit.is_scripting():
            self.dynamics.times = []
            with cuda_timing() as solver_time:
                u_hat, self.stats = self.ode_solver(
                    query.u0.flatten(start_dim=1), query.t, args
                )
            self.solver_time = solver_time
            self.times = self.dynamics.times
        else:
            adjoint = self.ode_solver.adjoint
            events: List[Any] = []
            adjoint.step_method.term.f.times = events
            adjoint.step_size_controller.term.f.times = events
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            self.solver_time = (start, end)
            stream = torch.cuda.current_stream(query.t.device)
            start.record(stream)
            u_hat, self.stats = self.ode_solver(
                query.u0.flatten(start_dim=1), query.t, args
            )
            end.record(stream)
            self.times = adjoint.step_method.term.f.times

        return u_hat.unflatten(dim=2, sizes=shape)
