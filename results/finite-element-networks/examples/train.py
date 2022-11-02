#!/usr/bin/env python

import os

# In pytorch 1.12.1, nvfuser has a problem compiling torchode and while the built-in
# fallback compiler achieves a speed-up, it is larger if we go with the "old" JIT
# compiler NNC right away.
os.environ["PYTORCH_JIT_USE_NNC_NOT_NVFUSER"] = "1"

import argparse
import logging
from pathlib import Path

import pytorch_lightning as pl
import torch

import finite_element_networks as fen
from finite_element_networks import (
    FEN,
    MLP,
    FENDomainInfo,
    FENDynamics,
    FreeFormTerm,
    ODESolver,
    TransportTerm,
)
from finite_element_networks.fen import DynWrapper
from finite_element_networks.lightning import (
    BlackSeaDataModule,
    CylinderFlowDataModule,
    MultipleShootingCallback,
    ScalarFlowDataModule,
    SequenceRegressionTask,
)
from finite_element_networks.ode_solver import TorchDynSolver, TorchODESolver

try:
    import wandb

    from finite_element_networks.lightning.wandb import PlotsCallback

    wandb_available = True
except:
    wandb_available = False

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints", default="checkpoints", help="Checkpoint directory"
    )
    parser.add_argument(
        "--library",
        required=True,
        help="ODE solver",
        choices=["torchode", "torchode-jit", "torchdiffeq", "torchdyn"],
    )
    parser.add_argument("--batch-size", required=True, type=int, help="Batch size")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--offline", default=False, action="store_true")
    parser.add_argument("--group", default="fen", help="W&B group name")
    parser.add_argument("--name", help="W&B run name")
    parser.add_argument(
        "--max-grad-norm", type=float, help="Max norm for gradient clipping"
    )
    parser.add_argument(
        "--jit", default=False, action="store_true", help="JIT compile the whole thing"
    )
    parser.add_argument(
        "dataset",
        choices=["black-sea", "scalar-flow", "cylinder-flow"],
        help="Dataset name",
    )
    args = parser.parse_args()

    checkpoint_dir = args.checkpoints
    library_name = args.library
    batch_size = args.batch_size
    max_grad_norm = args.max_grad_norm
    run_name = args.name or library_name
    jit_compile = args.jit or "-jit" in run_name
    dataset_name = args.dataset
    group = args.group

    pl.utilities.seed.seed_everything(args.seed)

    project_root = Path(fen.__file__).resolve().parent.parent
    data_root = project_root / "data" / dataset_name
    if dataset_name == "black-sea":
        dm_class = BlackSeaDataModule
        stationary, autonomous = False, False
        time_dim = 2
        n_features = 3
    elif dataset_name == "scalar-flow":
        dm_class = ScalarFlowDataModule
        stationary, autonomous = False, True
        time_dim = 1
        n_features = 4
    elif dataset_name == "cylinder-flow":
        dm_class = CylinderFlowDataModule
        stationary, autonomous = True, True
        time_dim = 1
        n_features = 3
    else:
        raise RuntimeError(f"Unknown dataset {dataset_name}")
    dm = dm_class(
        data_root,
        FENDomainInfo.from_domain,
        num_workers=2,
        pin_memory=True,
        train_target_steps=10,
        eval_target_steps=10,
        batch_size=batch_size,
    )

    dynamics = FENDynamics(
        [
            FreeFormTerm(
                FreeFormTerm.build_coefficient_mlp(
                    n_features=n_features,
                    time_dim=time_dim,
                    space_dim=2,
                    hidden_dim=96,
                    n_layers=4,
                    non_linearity=torch.nn.Tanh,
                    stationary=stationary,
                    autonomous=autonomous,
                ),
                stationary=stationary,
                autonomous=autonomous,
                zero_init=True,
            ),
            TransportTerm(
                TransportTerm.build_flow_field_mlp(
                    n_features=n_features,
                    time_dim=time_dim,
                    space_dim=2,
                    hidden_dim=96,
                    n_layers=4,
                    non_linearity=torch.nn.Tanh,
                    stationary=stationary,
                    autonomous=autonomous,
                ),
                stationary=stationary,
                autonomous=autonomous,
                zero_init=True,
            ),
        ]
    )
    dynamics = torch.jit.script(dynamics)
    dm.prepare_data()
    dm.setup("train")
    wrapped_dynamics = DynWrapper(dynamics, dm.train_data.time_encoder)
    if library_name == "torchdiffeq":
        ode_solver = ODESolver(
            wrapped_dynamics, "dopri5", atol=1e-6, rtol=1e-6, adjoint=False
        )
    elif library_name == "torchdyn":
        ode_solver = TorchDynSolver(wrapped_dynamics, "dopri5", atol=1e-6, rtol=1e-6)
    elif library_name.startswith("torchode"):
        ode_solver = TorchODESolver(
            wrapped_dynamics, "dopri5", atol=1e-6, rtol=1e-6, adjoint=False
        )

    model = FEN(wrapped_dynamics, ode_solver)
    if jit_compile:
        model = torch.jit.script(model)
    task = SequenceRegressionTask(model, standardize=True, max_grad_norm=max_grad_norm)

    wandb.init(
        project="torchode",
        group=group,
        name=run_name,
        mode="offline" if args.offline else "online",
        config=args,
    )

    logger = pl.loggers.WandbLogger() if wandb_available else None
    callbacks = [MultipleShootingCallback(initial_steps=3, increase=1)]
    if wandb_available:
        callbacks.append(pl.callbacks.ModelCheckpoint(monitor="val/mae", mode="min"))
        # We don't need plots
        # callbacks.append(PlotsCallback())
    else:
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=checkpoint_dir, monitor="val/mae", mode="min"
            )
        )
    gpus = 1 if torch.cuda.is_available() else 0
    trainer = pl.Trainer(
        max_epochs=50,
        callbacks=callbacks,
        gpus=gpus,
        logger=logger,
        log_every_n_steps=10,
    )
    trainer.fit(task, dm)
    trainer.test(task, dm)


if __name__ == "__main__":
    main()
