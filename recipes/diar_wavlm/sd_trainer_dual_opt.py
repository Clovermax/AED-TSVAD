# Licensed under the MIT license.
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)
# Copyright 2025 Nanjing University (author: Zeyan Song, clovermaxszy@gmail.com)

import torch
import numpy as np

from accelerate import Accelerator
from accelerate.logging import get_logger

from pyannote.audio.utils.permutation import permutate
from pyannote.audio.utils.loss import binary_cross_entropy

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from SDtrainZen.trainer_dual_opt import Trainer as BaseTrainer

logger = get_logger(__name__)

class Trainer(BaseTrainer):
    def __init__(
        self,
        accelerator: Accelerator,
        config,
        resume,
        model,
        optimizer_small,
        optimizer_big,
        loss_function=None,
    ):
        super().__init__(
            accelerator=accelerator,
            config=config,
            resume=resume,
            model=model,
            optimizer_small=optimizer_small,
            optimizer_big=optimizer_big,
            loss_function=loss_function,
        )
        self.accelerator.print(self.model)

        # auto GN
        self.grad_history = []

        # custom parameters

    def compute_grad_norm(self, model):
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm 

    def auto_clip_grad_norm_(self, model):
        grad_norm = self.compute_grad_norm(model)
        self.grad_history.append(grad_norm)
        if len(self.grad_history) > self.gradient_history_size:
            self.grad_history.pop(0)
        clip_value = np.percentile(self.grad_history, self.gradient_percentile)
        value = self.accelerator.clip_grad_norm_(model.parameters(), clip_value)  
        return value

    def training_step(self, batch, batch_idx):
        self.optimizer_small.zero_grad()
        self.optimizer_big.zero_grad()

        xs, xs_emb, target = batch['xs'], batch['xs_emb'], batch['ts']

        y_pred = self.model(xs, xs_emb)

        loss = binary_cross_entropy(y_pred, target)
 
        if torch.isnan(loss):
            print('Error, skipping batch')
            return None

        self.accelerator.backward(loss)

        if self.accelerator.sync_gradients:
            norm_value = self.auto_clip_grad_norm_(self.model)

        self.optimizer_small.step()
        self.optimizer_big.step()

        return {"Loss": loss.detach().float().cpu(), "Norm": norm_value}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        xs, xs_emb, target = batch['xs'], batch['xs_emb'], batch['ts']
        sil_all_target = torch.zeros_like(target)

        y_pred = self.model(xs, xs_emb)

        loss = binary_cross_entropy(y_pred, target)

        val_metrics = self.unwrap_model.validation_metric(
            torch.transpose(y_pred, 1, 2),
            torch.transpose(target, 1, 2),
        )

        if not torch.equal(target, sil_all_target):
            val_DER = val_metrics['DiarizationErrorRate']
            val_FA = val_metrics['DiarizationErrorRate/FalseAlarm']
            val_Miss = val_metrics['DiarizationErrorRate/Miss']
            val_Confusion = val_metrics['DiarizationErrorRate/Confusion']
            val_OptThreshold = val_metrics['DiarizationErrorRate/Threshold']
        else:  # if all targets are silence
            val_DER = torch.zeros_like(val_metrics['DiarizationErrorRate'])
            val_FA = torch.zeros_like(val_metrics['DiarizationErrorRate/FalseAlarm'])
            val_Miss = torch.zeros_like(val_metrics['DiarizationErrorRate/Miss'])
            val_Confusion = torch.zeros_like(val_metrics['DiarizationErrorRate/Confusion'])
            val_OptThreshold = val_metrics['DiarizationErrorRate/Threshold']

        return {"Loss": loss, "DER": val_DER, "FA": val_FA, "Miss": val_Miss, "Confusion": val_Confusion, "OptThreshold": val_OptThreshold}

    def validation_epoch_end(self, validation_epoch_output):
        metric_keys = validation_epoch_output[0].keys()
        # Compute mean loss on all loss items on a epoch
        for key in metric_keys:
            metric_items = [torch.mean(step_out[key]) for step_out in validation_epoch_output]
            metric_mean = torch.mean(torch.tensor(metric_items))
            if key == "Loss":
                Loss_val = metric_mean
            if key == "DER":
                DER_val = metric_mean
            if key == "FA":
                FA_val = metric_mean
            if key == "Miss":
                Miss_val = metric_mean
            if key == "Confusion":
                Confusion_val = metric_mean
            if key == "OptThreshold":
                OptThreshold_val = metric_mean
            self.writer.add_scalar(f"Validation_Epoch/{key}", metric_mean, self.state.epochs_trained)

        if 'Miss_val' in locals() and 'FA_val' in locals():
            Miss_FA_sum = Miss_val + FA_val
            self.writer.add_scalar("Validation_Epoch/Miss_FA_sum", Miss_FA_sum, self.state.epochs_trained)
        logger.info(
            f"Validation Loss/DER/FA/Miss/Confusion/Miss_FA_sum on epoch {self.state.epochs_trained}: "
            f"{round(Loss_val.item(), 3)} / {round(DER_val.item(), 3)} / {round(FA_val.item(), 3)} / "
            f"{round(Miss_val.item(), 3)} / {round(Confusion_val.item(), 3)} / {round(Miss_FA_sum.item(), 3)}"
            )
        # metric reset
        self.unwrap_model.validation_metric.reset()
        return Loss_val