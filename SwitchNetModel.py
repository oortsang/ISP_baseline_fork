
import numpy as np
import jax
from jax import lax
import jax.numpy as jnp
import flax.linen as nn

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
