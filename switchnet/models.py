
from collections.abc import Callable, Mapping
import dataclasses
import functools
from typing import Any, ClassVar, Protocol

from clu import metrics as clu_metrics
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from swirl_dynamics.templates import models
from swirl_dynamics.templates import trainers
from swirl_dynamics.templates import train_states
from swirl_dynamics.lib import metrics

Array = jax.Array
CondDict = Mapping[str, Array]
Metrics = clu_metrics.Collection
ShapeDict = Mapping[str, Any]  # may be nested
PyTree = Any
VariableDict = trainers.VariableDict


class SwitchNet(Protocol):
  def __call__(
      self, x: Array, is_training: bool
  ) -> Array:
    ...


@dataclasses.dataclass(frozen=True, kw_only=True)
class SwitchNetModel(models.BaseModel):

  input_shape: tuple[int, ...]
  core_module: nn.Module
  
  def initialize(self, rng: Array):
    # TODO: Add a dtype object to ensure consistency of types.
    x = jnp.ones((1,) + self.input_shape)

    return self.core_module.init(rng, x)

  def loss_fn(
      self,
      params: models.PyTree,
      batch: models.BatchType,
      rng: Array,
      mutables: models.PyTree,
  ) -> models.LossAndAux:
    
    y = self.core_module.apply({'params': params}, batch["scatter"])
    
    loss = jnp.mean(jnp.square(y - batch["eta"]))
    metric = dict(loss=loss)
    return loss, (metric, mutables)

  def eval_fn(
      self,
      variables: models.PyTree,
      batch: models.BatchType,
      rng: Array,
  ) -> models.ArrayDict:
    
    x = batch['scatter']
    core = self.inference_fn(variables, self.core_module)
    y = core(x)
    rrmse = functools.partial(
        metrics.mean_squared_error,
        sum_axes=(-1, -2),
        relative=True,
        squared=False,
    )

    return dict(rrmse=rrmse(pred=y, true=batch['eta']))

  @staticmethod
  def inference_fn(variables: models.PyTree, core_module: nn.Module):

    def _core(
        x: Array 
    ) -> Array:
      return core_module.apply(
          variables, x
      )

    return _core


class DMLayer(nn.Module):
    output_dim: int  # Output dimension as a class attribute

    @nn.compact
    def __call__(self, x):
        # x shape expected: [batch_size, a, b]
        batch_size = x.shape[0]

        # Define kernel: [a, b, c]
        kernel_shape = (x.shape[-2], x.shape[-1], self.output_dim)
        kernel = self.param('kernel', nn.initializers.uniform(), kernel_shape)
        
        bias_shape = (1, x.shape[-2], self.output_dim)  # Broadcastable shape for bias
        bias = self.param('bias', nn.initializers.uniform(), bias_shape)

        b = jnp.einsum('ijk,jkl->ijl', x, kernel, optimize=True)

        # Add bias (broadcasting will take care of batch dimension)
        b += bias

        return b
    
class SwitchNet(nn.Module):
    L1: int
    L2x: int
    L2y: int
    Nw1: int
    Nb1: int
    Nw2x: int
    Nw2y: int
    Nb2x: int
    Nb2y: int
    r: int
    w: int
    rc: int

    @nn.compact
    def __call__(self, x):
        batch_size = x.shape[0]
        n = x.shape[-1]
        # First set of operations (Reshape, Permute, DMLayer)
        x = x.reshape((batch_size, self.Nb1, self.Nw1, self.Nb1, self.Nw1, n))
        x = x.transpose((0, 1, 3, 2, 4, 5))
        x = x.reshape((batch_size, self.Nb1**2, n*self.Nw1**2))
        x = DMLayer(self.Nb2x*self.Nb2y*self.r)(x)
        x = x.reshape((batch_size, self.Nb1*self.Nb1, self.Nb2x*self.Nb2y, self.r))
        x = x.transpose((0, 3, 1, 2))

        # Second set of operations (Reshape, DMLayer)
        x = x.reshape((batch_size, self.Nb2x*self.Nb2y*self.Nb1**2, self.r))
        x = DMLayer(self.r)(x)
        x = x.reshape((batch_size, self.Nb2x*self.Nb2y, self.Nb1**2*self.r))

        # Third set of operations (DMLayer, Reshape, Permute)
        x = DMLayer(2*self.Nw2x*self.Nw2y)(x)
        x = x.reshape((batch_size, self.Nb2x, self.Nb2y, self.Nw2x, self.Nw2y, 2))
        x = x.transpose((0, 1, 3, 2, 4, 5))
        x = x.reshape((batch_size, self.Nb2x*self.Nw2x, self.Nb2y*self.Nw2y, 2))

        # Convolutional layers
        x = nn.Conv(features=6*self.r, kernel_size=(self.w, self.w), padding='SAME', strides=(1, 1))(x)
        x = nn.relu(x)
        x = nn.Conv(features=6*self.r, kernel_size=(self.w, self.w), padding='SAME', strides=(1, 1))(x)
        x = nn.relu(x)
        x = nn.Conv(features=6*self.r, kernel_size=(self.w, self.w), padding='SAME', strides=(1, 1))(x)
        x = nn.relu(x)
        x = nn.Conv(features=2, kernel_size=(self.w, self.w), padding='SAME', strides=(1, 1))(x)
        x = nn.relu(x)

        # Final operations (Reshape, DMLayer)
        x = x.reshape((batch_size, self.L2x*self.L2y, 2))
        x = DMLayer(1)(x)
        x = x.reshape((batch_size, self.L2x, self.L2y))

        return x
