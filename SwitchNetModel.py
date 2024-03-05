
import numpy as np
import jax
from jax import lax
import jax.numpy as jnp
import flax.linen as nn

class DMLayer(nn.Module):
    output_dim: int  

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
        b += bias

        return b

class switchnet(nn.Module):
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

    @nn.compact
    def __call__(self, x):
        batch_size = x.shape[0]

        # First set of operations (Reshape, Permute, DMLayer)
        x = x.reshape((batch_size, self.Nb1, self.Nw1, self.Nb1, self.Nw1, 2))
        x = x.transpose((0, 1, 3, 2, 4, 5))
        x = x.reshape((batch_size, self.Nb1**2, 2*self.Nw1**2))
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
        
        return x

# Define the main model using Flax
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

    def setup(self):
        self.switchnet0 = switchnet(L1=self.L1, L2x=self.L2x, L2y=self.L2y, Nw1=self.Nw1, Nb1=self.Nb1, 
                      Nw2x=self.Nw2x, Nw2y=self.Nw2y, Nb2x=self.Nb2x, Nb2y=self.Nb2y, r=self.r)
        self.switchnet1 = switchnet(L1=self.L1, L2x=self.L2x, L2y=self.L2y, Nw1=self.Nw1, Nb1=self.Nb1, 
                      Nw2x=self.Nw2x, Nw2y=self.Nw2y, Nb2x=self.Nb2x, Nb2y=self.Nb2y, r=self.r)
        self.switchnet2 = switchnet(L1=self.L1, L2x=self.L2x, L2y=self.L2y, Nw1=self.Nw1, Nb1=self.Nb1, 
                      Nw2x=self.Nw2x, Nw2y=self.Nw2y, Nb2x=self.Nb2x, Nb2y=self.Nb2y, r=self.r)
        self.convs = [nn.Conv(features=6, kernel_size=(3, 3), padding='SAME') for _ in range(9)]
        self.final_conv = nn.Conv(features=2, kernel_size=(3, 3), padding='SAME')
        self.DMLayer = DMLayer(1)
        
    def __call__(self, inputs):
        batch_size = inputs.shape[0]
        y1 = self.switchnet0(inputs[:, :, :, :, 0])
        y2 = self.switchnet1(inputs[:, :, :, :, 1])
        y3 = self.switchnet2(inputs[:, :, :, :, 2])
        
        y = jnp.concatenate([y1, y2, y3], axis=-1)

        for conv_layer in self.convs:
            tmp = conv_layer(y)
            tmp = jax.nn.relu(tmp)
            y = jnp.concatenate([y, tmp], axis = -1)
        
        y = self.final_conv(y)
        y = jax.nn.relu(y)
        y = y.reshape((batch_size, self.L2x*self.L2y, 2))
        y = self.DMLayer(y)
        y = y.reshape((batch_size, self.L2x, self.L2y))
        
        return y
