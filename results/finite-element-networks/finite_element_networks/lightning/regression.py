from typing import Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchode_utils.timing import cuda_timing

from ..data import STBatch
from ..fen import FEN, FENQuery
from .metrics import main_metrics


class SequenceRegressionTask(pl.LightningModule):
    def __init__(
        self,
        model: FEN,
        *,
        loss: Optional[nn.Module] = None,
        standardize: bool = True,
        max_grad_norm: Optional[float] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=("model",))

        self.model = model
        self.loss = loss or nn.L1Loss()
        self.standardize = standardize
        self.max_grad_norm = max_grad_norm

        metrics = main_metrics()
        self.metrics = nn.ModuleDict(
            {
                # The underscores are here because ModuleDict is stupid and does not allow
                # you to have a 'train' key because it also has a `train` method
                "train_": metrics.clone(prefix="train/"),
                "val_": metrics.clone(prefix="val/"),
                "test_": metrics.clone(prefix="test/"),
            }
        )

    def forward(self, batch: STBatch):
        query = FENQuery.from_batch(batch, standardize=self.standardize)
        u_hat = self.model(query)
        if self.standardize:
            u_hat = batch.target_standardizer.undo(u_hat)

        return u_hat

    def training_step(self, batch: STBatch, batch_idx: int):
        with cuda_timing() as fw_time:
            u_hat = self(batch)
        loss = self.loss(u_hat, batch.target_u)

        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, batch_size=batch.batch_size
        )
        self.log_times("train", fw_time, batch.batch_size)
        self.log_model_metrics("train", u_hat, batch)
        return {"loss": loss}

    def log_times(self, stage, fw_time, batch_size):
        if isinstance(self.model.stats, dict):
            fw_steps = self.model.stats["n_steps"].cpu().numpy()
        else:
            fw_steps = self.model.stats.forward_steps()
        metrics = {
            f"{stage}/fw_time": fw_time(),
            f"{stage}/fw_steps": fw_steps.mean(),
            f"{stage}/fw_time_per_step": fw_time() / fw_steps.mean(),
        }
        if isinstance(self.model.solver_time, tuple):
            model_times = []
            for i in range(0, len(self.model.times), 2):
                s, e = self.model.times[i:i+2]
                e.synchronize()
                model_times.append(s.elapsed_time(e))
            model_time = sum(model_times)
            s, e = self.model.solver_time
            e.synchronize()
            solver_time = s.elapsed_time(e) - model_time
        else:
            model_time = sum(t() for t in self.model.times)
            solver_time = self.model.solver_time() - model_time
        loop_time = solver_time / fw_steps.max()
        metrics = {
            **metrics,
            f"{stage}/fw_model_time": model_time,
            f"{stage}/fw_model_time_per_step": model_time / fw_steps.mean(),
            f"{stage}/fw_solver_time": solver_time,
            f"{stage}/fw_loop_time": loop_time,
        }
        self.log_dict(
            metrics,
            on_step=stage == "train",
            on_epoch=True,
            batch_size=batch_size,
        )

    def configure_gradient_clipping(
        self,
        optimizer,
        optimizer_idx,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ):
        if self.max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )

    def backward(self, loss, optimizer, optimizer_idx, *args, **kwargs):
        with cuda_timing() as bw_time:
            loss.backward(*args, **kwargs)

        if isinstance(self.model.stats, dict):
            fw_steps = self.model.stats["n_steps"].cpu().numpy()
        else:
            fw_steps = self.model.stats.forward_steps()

        self.log_dict(
            {
                "train/bw_time": bw_time(),
                "train/bw_time_per_step": bw_time() / fw_steps.mean(),
            },
            # Batch size not available here and I don't think it is too important
            # for this measurement
            # batch_size=batch.batch_size,
        )

    def validation_step(self, batch: STBatch, batch_idx: int):
        with cuda_timing() as fw_time:
            u_hat = self(batch)
        self.log_times("val", fw_time, batch.batch_size)
        self.log_model_metrics("val", u_hat, batch)
        return {}

    def test_step(self, batch: STBatch, batch_idx: int):
        with cuda_timing() as fw_time:
            u_hat = self(batch)
        self.log_times("test", fw_time, batch.batch_size)
        self.log_model_metrics("test", u_hat, batch)
        return {}

    def log_model_metrics(self, stage: str, u_hat, batch):
        with torch.no_grad():
            values = self.metrics[stage + "_"](u_hat, batch.target_u)
            self.log_dict(
                values,
                on_step=stage == "train",
                on_epoch=True,
                batch_size=batch.batch_size,
            )

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())
