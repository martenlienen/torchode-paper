import argparse
import os
import random

import numpy as np
import torch
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as tforms
from torchode_utils.timing import cuda_timing
from torchvision.utils import save_image

import lib.layers as layers
import lib.multiscale_parallel as multiscale_parallel
import lib.odenvp as odenvp
import lib.utils as utils
import wandb
from train_misc import (
    add_spectral_norm,
    append_regularization_to_log,
    collect_solver_and_model_times,
    collect_stats,
    count_nfe,
    count_parameters,
    count_total_time,
    create_regularization_fns,
    get_regularization,
    set_cnf_options,
    spectral_norm_power_iteration,
    standard_normal_logprob,
)

# go fast boi!!
torch.backends.cudnn.benchmark = True
SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", "adams", "explicit_adams"]
parser = argparse.ArgumentParser("Continuous Normalizing Flow")
parser.add_argument(
    "--data",
    choices=["mnist", "svhn", "cifar10", "lsun_church"],
    type=str,
    default="mnist",
)
parser.add_argument("--dims", type=str, default="8,32,32,8")
parser.add_argument("--strides", type=str, default="2,2,1,-2,-2")
parser.add_argument("--num_blocks", type=int, default=1, help="Number of stacked CNFs.")

parser.add_argument("--conv", type=eval, default=True, choices=[True, False])
parser.add_argument(
    "--layer_type",
    type=str,
    default="ignore",
    choices=[
        "ignore",
        "concat",
        "concat_v2",
        "squash",
        "concatsquash",
        "concatcoord",
        "hyper",
        "blend",
    ],
)
parser.add_argument(
    "--divergence_fn",
    type=str,
    default="approximate",
    choices=["brute_force", "approximate"],
)
parser.add_argument(
    "--nonlinearity",
    type=str,
    default="softplus",
    choices=["tanh", "relu", "softplus", "elu", "swish"],
)
parser.add_argument("--solver", type=str, default="dopri5", choices=SOLVERS)
parser.add_argument("--atol", type=float, default=1e-5)
parser.add_argument("--rtol", type=float, default=1e-5)
parser.add_argument(
    "--step_size", type=float, default=None, help="Optional fixed step size."
)

parser.add_argument("--test_solver", type=str, default=None, choices=SOLVERS + [None])
parser.add_argument("--test_atol", type=float, default=None)
parser.add_argument("--test_rtol", type=float, default=None)

parser.add_argument("--imagesize", type=int, default=None)
parser.add_argument("--alpha", type=float, default=1e-6)
parser.add_argument("--time_length", type=float, default=1.0)
parser.add_argument("--train_T", type=eval, default=True)

parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument(
    "--batch_size_schedule",
    type=str,
    default="",
    help="Increases the batchsize at every given epoch, dash separated.",
)
parser.add_argument("--test_batch_size", type=int, default=200)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--warmup_iters", type=float, default=1000)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--spectral_norm_niter", type=int, default=10)

parser.add_argument("--add_noise", type=eval, default=True, choices=[True, False])
parser.add_argument("--batch_norm", type=eval, default=False, choices=[True, False])
parser.add_argument("--residual", type=eval, default=False, choices=[True, False])
parser.add_argument("--autoencode", type=eval, default=False, choices=[True, False])
parser.add_argument("--rademacher", type=eval, default=True, choices=[True, False])
parser.add_argument("--spectral_norm", type=eval, default=False, choices=[True, False])
parser.add_argument("--multiscale", type=eval, default=False, choices=[True, False])
parser.add_argument("--parallel", type=eval, default=False, choices=[True, False])

# Regularizations
parser.add_argument("--l1int", type=float, default=None, help="int_t ||f||_1")
parser.add_argument("--l2int", type=float, default=None, help="int_t ||f||_2")
parser.add_argument("--dl2int", type=float, default=None, help="int_t ||f^T df/dt||_2")
parser.add_argument("--JFrobint", type=float, default=None, help="int_t ||df/dx||_F")
parser.add_argument(
    "--JdiagFrobint", type=float, default=None, help="int_t ||df_i/dx_i||_F"
)
parser.add_argument(
    "--JoffdiagFrobint", type=float, default=None, help="int_t ||df/dx - df_i/dx_i||_F"
)

parser.add_argument(
    "--time_penalty", type=float, default=0, help="Regularization on the end_time."
)
parser.add_argument(
    "--max_grad_norm",
    type=float,
    default=1e10,
    help="Max norm of graidents (default is just stupidly high to avoid any clipping)",
)

parser.add_argument("--begin_epoch", type=int, default=1)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--save", type=str, default="experiments/cnf")
parser.add_argument("--val_freq", type=int, default=1)
parser.add_argument("--log_freq", type=int, default=10)

parser.add_argument(
    "--manual_seed", type=int, help="manual seed, if not given resorts to random seed."
)
parser.add_argument(
    "--library", required=True, help="ODE solver", choices=["torchode", "torchode-joint", "torchdiffeq", "torchdyn"]
)
parser.add_argument(
    "--offline", default=False, action="store_true", help="Disable W&B logging"
)

args = parser.parse_args()

if args.manual_seed is None:
    args.manual_seed = random.randint(1, 100000)
random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
np.random.seed(args.manual_seed)

# logger
utils.makedirs(args.save)
logger = utils.get_logger(
    logpath=os.path.join(args.save, "logs"), filepath=os.path.abspath(__file__)
)

if args.layer_type == "blend":
    logger.info("!! Setting time_length from None to 1.0 due to use of Blend layers.")
    args.time_length = 1.0

logger.info(args)


def add_noise(x):
    """
    [0, 1] -> [0, 255] -> add noise -> [0, 1]
    """
    if args.add_noise:
        noise = x.new().resize_as_(x).uniform_()
        x = x * 255 + noise
        x = x / 256
    return x


def update_lr(optimizer, itr):
    iter_frac = min(float(itr + 1) / max(args.warmup_iters, 1), 1.0)
    lr = args.lr * iter_frac
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_train_loader(train_set, epoch):
    if args.batch_size_schedule != "":
        epochs = [0] + list(map(int, args.batch_size_schedule.split("-")))
        n_passed = sum(np.array(epochs) <= epoch)
        current_batch_size = int(args.batch_size * n_passed)
    else:
        current_batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=current_batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    logger.info(
        "===> Using batch size {}. Total {} iterations/epoch.".format(
            current_batch_size, len(train_loader)
        )
    )
    return train_loader


def get_dataset(args):
    trans = lambda im_size: tforms.Compose(
        [tforms.Resize(im_size), tforms.ToTensor(), add_noise]
    )

    if args.data == "mnist":
        im_dim = 1
        im_size = 28 if args.imagesize is None else args.imagesize
        train_set = dset.MNIST(
            root="./data", train=True, transform=trans(im_size), download=True
        )
        test_set = dset.MNIST(
            root="./data", train=False, transform=trans(im_size), download=True
        )
    elif args.data == "svhn":
        im_dim = 3
        im_size = 32 if args.imagesize is None else args.imagesize
        train_set = dset.SVHN(
            root="./data", split="train", transform=trans(im_size), download=True
        )
        test_set = dset.SVHN(
            root="./data", split="test", transform=trans(im_size), download=True
        )
    elif args.data == "cifar10":
        im_dim = 3
        im_size = 32 if args.imagesize is None else args.imagesize
        train_set = dset.CIFAR10(
            root="./data",
            train=True,
            transform=tforms.Compose(
                [
                    tforms.Resize(im_size),
                    tforms.RandomHorizontalFlip(),
                    tforms.ToTensor(),
                    add_noise,
                ]
            ),
            download=True,
        )
        test_set = dset.CIFAR10(
            root="./data", train=False, transform=trans(im_size), download=True
        )
    elif args.data == "celeba":
        im_dim = 3
        im_size = 64 if args.imagesize is None else args.imagesize
        train_set = dset.CelebA(
            train=True,
            transform=tforms.Compose(
                [
                    tforms.ToPILImage(),
                    tforms.Resize(im_size),
                    tforms.RandomHorizontalFlip(),
                    tforms.ToTensor(),
                    add_noise,
                ]
            ),
        )
        test_set = dset.CelebA(
            train=False,
            transform=tforms.Compose(
                [
                    tforms.ToPILImage(),
                    tforms.Resize(im_size),
                    tforms.ToTensor(),
                    add_noise,
                ]
            ),
        )
    elif args.data == "lsun_church":
        im_dim = 3
        im_size = 64 if args.imagesize is None else args.imagesize
        train_set = dset.LSUN(
            "data",
            ["church_outdoor_train"],
            transform=tforms.Compose(
                [
                    tforms.Resize(96),
                    tforms.RandomCrop(64),
                    tforms.Resize(im_size),
                    tforms.ToTensor(),
                    add_noise,
                ]
            ),
        )
        test_set = dset.LSUN(
            "data",
            ["church_outdoor_val"],
            transform=tforms.Compose(
                [
                    tforms.Resize(96),
                    tforms.RandomCrop(64),
                    tforms.Resize(im_size),
                    tforms.ToTensor(),
                    add_noise,
                ]
            ),
        )
    data_shape = (im_dim, im_size, im_size)
    if not args.conv:
        data_shape = (im_dim * im_size * im_size,)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=args.test_batch_size, shuffle=False, drop_last=True
    )
    return train_set, test_loader, data_shape


def compute_bits_per_dim(x, model):
    zero = x.new_zeros(x.shape[0], 1)

    # Don't use data parallelize if batch size is small.
    # if x.shape[0] < 200:
    #     model = model.module

    z, delta_logp = model(x, zero)  # run model forward

    logpz = (
        standard_normal_logprob(z).view(z.shape[0], -1).sum(1, keepdim=True)
    )  # logp(z)
    logpx = logpz - delta_logp

    logpx_per_dim = torch.sum(logpx) / x.nelement()  # averaged over batches
    bits_per_dim = -(logpx_per_dim - np.log(256)) / np.log(2)

    return bits_per_dim


def create_model(args, data_shape, regularization_fns):
    hidden_dims = tuple(map(int, args.dims.split(",")))
    strides = tuple(map(int, args.strides.split(",")))

    if args.multiscale:
        model = odenvp.ODENVP(
            (args.batch_size, *data_shape),
            n_blocks=args.num_blocks,
            intermediate_dims=hidden_dims,
            nonlinearity=args.nonlinearity,
            alpha=args.alpha,
            cnf_kwargs={
                "T": args.time_length,
                "train_T": args.train_T,
                "regularization_fns": regularization_fns,
                "library": args.library,
            },
        )
    elif args.parallel:
        model = multiscale_parallel.MultiscaleParallelCNF(
            (args.batch_size, *data_shape),
            n_blocks=args.num_blocks,
            intermediate_dims=hidden_dims,
            alpha=args.alpha,
            time_length=args.time_length,
        )
    else:
        if args.autoencode:

            def build_cnf():
                autoencoder_diffeq = layers.AutoencoderDiffEqNet(
                    hidden_dims=hidden_dims,
                    input_shape=data_shape,
                    strides=strides,
                    conv=args.conv,
                    layer_type=args.layer_type,
                    nonlinearity=args.nonlinearity,
                )
                odefunc = layers.AutoencoderODEfunc(
                    autoencoder_diffeq=autoencoder_diffeq,
                    divergence_fn=args.divergence_fn,
                    residual=args.residual,
                    rademacher=args.rademacher,
                )
                cnf = layers.CNF(
                    odefunc=odefunc,
                    T=args.time_length,
                    regularization_fns=regularization_fns,
                    solver=args.solver,
                    library=args.library,
                )
                return cnf

        else:

            def build_cnf():
                diffeq = layers.ODEnet(
                    hidden_dims=hidden_dims,
                    input_shape=data_shape,
                    strides=strides,
                    conv=args.conv,
                    layer_type=args.layer_type,
                    nonlinearity=args.nonlinearity,
                )
                odefunc = layers.ODEfunc(
                    diffeq=diffeq,
                    divergence_fn=args.divergence_fn,
                    residual=args.residual,
                    rademacher=args.rademacher,
                )
                cnf = layers.CNF(
                    odefunc=odefunc,
                    T=args.time_length,
                    train_T=args.train_T,
                    regularization_fns=regularization_fns,
                    solver=args.solver,
                    library=args.library,
                )
                return cnf

        chain = (
            [layers.LogitTransform(alpha=args.alpha)]
            if args.alpha > 0
            else [layers.ZeroMeanTransform()]
        )
        chain = chain + [build_cnf() for _ in range(args.num_blocks)]
        if args.batch_norm:
            chain.append(layers.MovingBatchNorm2d(data_shape[0]))
        model = layers.SequentialFlow(chain)
    return model


if __name__ == "__main__":

    # get deivce
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cvt = lambda x: x.type(torch.float32).to(device, non_blocking=True)

    # load dataset
    train_set, test_loader, data_shape = get_dataset(args)

    # build model
    regularization_fns, regularization_coeffs = create_regularization_fns(args)
    model = create_model(args, data_shape, regularization_fns)

    if args.spectral_norm:
        add_spectral_norm(model, logger)
    set_cnf_options(args, model)

    # logger.info(model)
    logger.info("Number of trainable parameters: {}".format(count_parameters(model)))

    # optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # restore parameters
    if args.resume is not None:
        checkpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpt["state_dict"])
        if "optim_state_dict" in checkpt.keys():
            optimizer.load_state_dict(checkpt["optim_state_dict"])
            # Manually move optimizer state to device.
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = cvt(v)

    if torch.cuda.is_available():
        model = model.cuda()

    # For visualization.
    fixed_z = cvt(torch.randn(100, *data_shape))

    loss_meter = utils.RunningAverageMeter(0.97)

    if args.spectral_norm and not args.resume:
        spectral_norm_power_iteration(model, 500)

    wandb.init(
        project="torchode",
        group="cnf",
        name=args.library,
        mode="offline" if args.offline else "online",
        config=args,
    )

    best_loss = float("inf")
    best_epoch = 0
    itr = 0
    for epoch in range(args.begin_epoch, args.num_epochs + 1):
        model.train()
        train_loader = get_train_loader(train_set, epoch)
        epoch_metrics = []
        for _, (x, y) in enumerate(train_loader):
            update_lr(optimizer, itr)
            optimizer.zero_grad()

            if not args.conv:
                x = x.view(x.shape[0], -1)

            # cast data and move to device
            x = cvt(x)
            with cuda_timing() as fw_time:
                # compute loss
                loss = compute_bits_per_dim(x, model)
                if regularization_coeffs:
                    reg_states = get_regularization(model, regularization_coeffs)
                    reg_loss = sum(
                        reg_state * coeff
                        for reg_state, coeff in zip(reg_states, regularization_coeffs)
                        if coeff != 0
                    )
                    loss = loss + reg_loss

            if args.library in ("torchdiffeq", "torchdyn"):
                n_ode_layers, forward_nfe = count_nfe(model)

            with cuda_timing() as bw_time:
                loss.backward()

            if args.library in ("torchdiffeq", "torchdyn"):
                _, backward_nfe = count_nfe(model)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm
            )
            optimizer.step()

            if args.spectral_norm:
                spectral_norm_power_iteration(model, args.spectral_norm_niter)

            (
                fw_solver_time,
                bw_solver_time,
                fw_model_time,
                bw_model_time,
            ) = collect_solver_and_model_times(model)

            if args.library in ("torchdiffeq", "torchdyn"):
                initial_dt_nfes = 2 * n_ode_layers
                nfe_per_step = 6
                fw_steps = np.array((forward_nfe - initial_dt_nfes) / nfe_per_step)
                bw_steps = np.array(
                    (backward_nfe - forward_nfe - initial_dt_nfes) / nfe_per_step
                )
            else:
                fw_stats, bw_stats = collect_stats(model)
                fw_steps = fw_stats["n_steps"]
                bw_steps = bw_stats["n_steps"]

            loss_meter.update(loss.item())

            if itr % args.log_freq == 0:
                log_message = (
                    "Iter {:04d} | Time {:.1f}ms / {:.1f}ms | Solve {:.1f}ms / {:.1f}ms | "
                    "Model {:.1f}ms / {:.1f}ms | Bit/dim {:.4f}({:.4f}) | "
                    "Steps {:.1f} / {:.1f} | Grad Norm {:.2f}".format(
                        itr,
                        fw_time(),
                        bw_time(),
                        fw_solver_time,
                        bw_solver_time,
                        fw_model_time,
                        bw_model_time,
                        loss_meter.val,
                        loss_meter.avg,
                        fw_steps.mean(),
                        bw_steps.mean(),
                        grad_norm,
                    )
                )
                if regularization_coeffs:
                    log_message = append_regularization_to_log(
                        log_message, regularization_fns, reg_states
                    )
                logger.info(log_message)

                metrics = {
                    "loss": loss_meter.val,
                    "grad_norm": grad_norm.item(),
                    "fw_time": fw_time(),
                    "fw_steps": fw_steps.mean(),
                    "fw_time_per_step": fw_time() / fw_steps.mean(),
                    "fw_model_time": fw_model_time,
                    "fw_model_time_per_step": fw_model_time / fw_steps.mean(),
                    "fw_solver_time": fw_solver_time,
                    "fw_loop_time": fw_solver_time / fw_steps.max(),
                    "bw_time": bw_time(),
                    "bw_steps": bw_steps.mean(),
                    "bw_time_per_step": bw_time() / bw_steps.mean(),
                    "bw_model_time": bw_model_time,
                    "bw_model_time_per_step": bw_model_time / bw_steps.mean(),
                    "bw_solver_time": bw_solver_time,
                    "bw_loop_time": bw_solver_time / bw_steps.max(),
                }
                wandb.log(
                    {
                        "epoch": epoch,
                        **{"train/" + k + "_step": v for k, v in metrics.items()},
                    }
                )
                epoch_metrics.append(metrics)

            itr += 1

        summary_metrics = {
            "train/" + key + "_epoch": np.mean([m[key] for m in epoch_metrics])
            for key in epoch_metrics[0].keys()
        }
        wandb.log({"epoch": epoch, **summary_metrics})

        # compute test loss
        model.eval()
        if epoch % args.val_freq == 0:
            with torch.no_grad():
                logger.info("validating...")
                val_metrics = []
                for (x, y) in test_loader:
                    if not args.conv:
                        x = x.view(x.shape[0], -1)
                    x = cvt(x)
                    with cuda_timing() as fw_time:
                        loss = compute_bits_per_dim(x, model)

                    if args.library in ("torchdiffeq", "torchdyn"):
                        n_ode_layers, forward_nfe = count_nfe(model)
                        initial_dt_nfes = 2 * n_ode_layers
                        nfe_per_step = 6
                        fw_steps = np.array(
                            (forward_nfe - initial_dt_nfes) / nfe_per_step
                        )
                    elif args.library.startswith("torchode"):
                        fw_stats, _ = collect_stats(model)
                        fw_steps = fw_stats["n_steps"]

                    (
                        fw_solver_time,
                        _,
                        fw_model_time,
                        _,
                    ) = collect_solver_and_model_times(model)

                    metrics = {
                        "loss": loss.item(),
                        "fw_time": fw_time(),
                        "fw_steps": fw_steps.mean(),
                        "fw_time_per_step": fw_time() / fw_steps.mean(),
                        "fw_model_time": fw_model_time,
                        "fw_model_time_per_step": fw_model_time / fw_steps.mean(),
                        "fw_solver_time": fw_solver_time,
                        "fw_loop_time": fw_solver_time / fw_steps.max(),
                    }
                    val_metrics.append(metrics)

                summary_metrics = {
                    "val/" + key: np.mean([m[key] for m in val_metrics])
                    for key in val_metrics[0].keys()
                }
                wandb.log({"epoch": epoch, **summary_metrics})

                log_message = (
                    "Epoch {:04d} | Time {:.1f}ms | Solve {:.1f}ms | "
                    "Model {:.1f}ms | Bit/dim {:.4f} | Steps {:.1f}".format(
                        epoch,
                        summary_metrics["val/fw_time"],
                        summary_metrics["val/fw_solver_time"],
                        summary_metrics["val/fw_model_time"],
                        summary_metrics["val/loss"],
                        summary_metrics["val/fw_steps"],
                    )
                )
                logger.info(log_message)

                val_loss = summary_metrics["val/loss"]
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = epoch
                    utils.makedirs(args.save)
                    torch.save(
                        {
                            "args": args,
                            "state_dict": model.state_dict(),
                            "optim_state_dict": optimizer.state_dict(),
                        },
                        os.path.join(wandb.run.dir, "best.pth"),
                    )
                    torch.save(
                        val_metrics, os.path.join(wandb.run.dir, "best-stats.pth")
                    )

                    wandb.run.summary["best_epoch"] = best_epoch
                    wandb.run.summary.update(summary_metrics)

        # visualize samples and density
        with torch.no_grad():
            fig_filename = os.path.join(args.save, "figs", "{:04d}.jpg".format(epoch))
            utils.makedirs(os.path.dirname(fig_filename))
            generated_samples = model(fixed_z, reverse=True).view(-1, *data_shape)
            save_image(generated_samples, fig_filename, nrow=10)

            wandb.log({"epoch": epoch, "val/sample": wandb.Image(fig_filename)})
