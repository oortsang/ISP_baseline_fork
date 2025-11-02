# (OOT, 2025-11-01)
# An additional callback object for use


from collections.abc import Mapping, Sequence
import dataclasses
import os
import time
from typing import Any

# from absl import logging
import logging
from clu import metric_writers
from clu import parameter_overview
from clu import periodic_actions
import gin
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
import orbax.checkpoint as ocp
from swirl_dynamics.templates import train_states
from swirl_dynamics.templates import trainers
from swirl_dynamics.templates import utils
from swirl_dynamics.templates.callbacks import Callback, TrainStateCheckpoint

import tqdm.auto as tqdm

Array = jax.Array
ComputedMetrics = Mapping[str, Array | Mapping[str, Array]]
Trainer = trainers.BaseTrainer


class LoggingOutput(Callback):
    """logging.info callback to monitor training progress in real time."""

    def __init__(
        self,
        total_train_steps: int | None,
        train_monitors: Sequence[str],
        eval_monitors: Sequence[str] = (),
    ):
      """Sets up logging to stderr at the INFO level
      Args:
      total_train_steps: the total number of training steps, which is displayed
        as the maximum progress on the bar.
      train_monitors: keys in the training metrics whose values are updated on
        the progress bar after every training metric aggregation.
      eval_monitors: same as `train_monitors` except applying to evaluation.
      """
      self.total_train_steps = total_train_steps
      self.train_monitors = train_monitors
      self.eval_monitors = eval_monitors
      self.current_step = 0
      self.eval_postfix = {}  # keeps record of the most recent eval monitor
      # self.bar = None
      self.start_time = None
      self.last_time = None
      self.last_step = 0

    def on_train_begin(self, trainer: Trainer) -> None:
        del trainer
        # self.bar = tqdm.tqdm(total=self.total_train_steps, unit="step")
        self.start_time = time.perf_counter()
        self.last_time  = time.perf_counter()

    def on_train_batches_end(
        self, trainer: Trainer, train_metrics: ComputedMetrics
    ) -> None:
        # self.bar.update(trainer.train_state.int_step - self.current_step)
        self.current_step = trainer.train_state.int_step
        self.postfix = {
            monitor: train_metrics[monitor] for monitor in self.train_monitors
        }
        # self.bar.set_postfix(**postfix, **self.eval_postfix)
        monitored_vals_dict = {**self.postfix, **self.eval_postfix}
        monitored_vals_str = ", ".join([
            f"{k}={v:.5e}"
            for k,v in monitored_vals_dict.items()
        ])
        curr_time = time.perf_counter()
        rel_time     = curr_time - self.start_time
        step_time    = (
            (curr_time - self.last_time)
            / (self.current_step-self.last_step)
        )
        steps_left   = self.total_train_steps-self.current_step
        est_tot_time = rel_time + step_time * steps_left
        logging.info(
            f"({rel_time:.2f}s/~{est_tot_time:.0f}s) "
            f"step {self.current_step}/{self.total_train_steps}: "
            f"{monitored_vals_str}"
        )
        self.last_step = self.current_step
        self.last_time = curr_time

    def on_eval_batches_end(
        self, trainer: Trainer, eval_metrics: ComputedMetrics
    ) -> None:
        del trainer
        self.eval_postfix = {
            monitor: eval_metrics[monitor] for monitor in self.eval_monitors
        }

    def on_train_end(self, trainer: Trainer) -> None:
        del trainer
        logging.info(f"Training is complete")


# class TrainStateCheckpointEpochChoice(Callback):
class TrainStateCheckpointEpochChoice(TrainStateCheckpoint):
    """Callback that periodically saves train state checkpoints."""

    # def __init__(
    #     self,
    #     base_dir: str,
    #     folder_prefix: str = "checkpoints",
    #     train_state_field: str = "default",
    #     options: ocp.CheckpointManagerOptions | None = None,
    # ):
    #   self.save_dir = os.path.join(base_dir, folder_prefix)
    #   self.train_state_field = train_state_field
    #   self.ckpt_manager = ocp.CheckpointManager(
    #       self.save_dir,
    #       item_handlers={self.train_state_field: ocp.StandardCheckpointHandler()},
    #       options=options,
    #   )

    # def on_train_begin(self, trainer: Trainer) -> None:
    #   """Sets up directory, saves initial or restore the most recent state."""
    #   self.last_eval_metric = {}
    #   # retrieve from existing checkpoints if possible
    #   if self.ckpt_manager.latest_step() is not None:

    #     def to_shard_shape_dtype(x):
    #       aval = jax.api_util.shaped_abstractify(x)
    #       if trainer.is_distributed:
    #         return jax.ShapeDtypeStruct(aval.shape[1:], dtype=aval.dtype)
    #       else:
    #         return jax.ShapeDtypeStruct(aval.shape, dtype=aval.dtype)

    #     # Load a single shard and then replicate explicitly.
    #     restored = self.ckpt_manager.restore(
    #         self.ckpt_manager.latest_step(),
    #         args=ocp.args.Composite(**{
    #             self.train_state_field: ocp.args.StandardRestore(
    #                 item=jax.tree.map(to_shard_shape_dtype, trainer.train_state)
    #             )
    #         }),
    #     )

    #     trainer.train_state = trainer._maybe_replicate(  # pylint: disable=protected-access
    #         getattr(restored, self.train_state_field)
    #     )

    def on_train_batches_end(
        self, trainer: Trainer, train_metrics: ComputedMetrics
    ) -> None:
        assert self.last_eval_metric is not None
        cur_step = trainer.train_state.int_step
        self.last_train_metric = {
            k: np.array(v).item()
            for k,v in train_metrics.items()
        }
        last_eval_metric_exists = (
            hasattr(self, "last_eval_metric")
            and self.last_eval_metric is not None
            and len(self.last_eval_metric) != 0
        )
        should_save = (
            last_eval_metric_exists
            and self.ckpt_manager.should_save(cur_step)
        )
        if should_save:
            # (OOT, 2025-11-01) ocp checkpoint manager complained because
            # the metrics dict had entries as jax arrays rather than
            # floats or python lists, which would be needed for serialization
            metrics_for_ser = {
                k: np.array(v).item()
                for k,v in
                dict(**train_metrics, **self.last_eval_metric).items()
            }
            # print(f"metrics...{metrics_for_ser}")
            self.ckpt_manager.save(
                step=cur_step,
                # This always saves the unreplicated train state.
                # Converting to np array seems necessary for multi-host environments.
                args=ocp.args.Composite(**{
                    self.train_state_field: ocp.args.StandardSave(
                        jax.tree.map(np.array, trainer.unreplicated_train_state)
                    )
                }),
                # metrics=dict(**train_metrics, **self.last_eval_metric),
                metrics=metrics_for_ser,
            )
    def on_eval_batches_end(
        self, trainer: Trainer, eval_metrics: ComputedMetrics
    ) -> None:
        del trainer
        self.last_eval_metric = {
            k: np.array(v).item()
            for k,v in eval_metrics.items()
        }
        # print(f"eval metrics: {self.last_eval_metric}")
  
    def on_train_end(self, trainer: Trainer) -> None:
        # Always save a checkpoint at the end of training.
        if self.ckpt_manager.latest_step() != trainer.train_state.int_step:

            metrics_for_ser = {
                k: np.array(v).item()
                for k,v in
                {**self.last_train_metric, **self.last_eval_metric}.items()
            }
            # print(f"metrics...{metrics_for_ser}")
            self.ckpt_manager.save(
                trainer.train_state.int_step,
                args=ocp.args.Composite(**{
                    self.train_state_field: ocp.args.StandardSave(
                        jax.tree.map(np.array, trainer.unreplicated_train_state)
                        # jax.tree.map(
                        #     # lambda x: np.array(x).tolist(),
                        #     # lambda x: np.array(x).item(),
                        #     trainer.unreplicated_train_state,
                        # )
                    )
                }),
                metrics=metrics_for_ser,
                force=True,
            )
            self.ckpt_manager.wait_until_finished()
            
