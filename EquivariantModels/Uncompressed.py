import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import sparse
import flax.linen as nn
            
class Fstar(nn.Module):
    nx: int
    neta: int
    cart_mat: jnp.ndarray
    r_index: np.ndarray 

    def setup(self):
        kernel_shape = (self.nx, self.nx)
        p_shape = (1, self.nx)
        
        self.pre1 = self.param('pre1', nn.initializers.glorot_uniform(), p_shape)
        self.pre2 = self.param('pre2', nn.initializers.glorot_uniform(), p_shape)
        self.pre3 = self.param('pre3', nn.initializers.glorot_uniform(), p_shape)
        self.pre4 = self.param('pre4', nn.initializers.glorot_uniform(), p_shape)

        self.post1 = self.param('post1', nn.initializers.glorot_uniform(), p_shape)
        self.post2 = self.param('post2', nn.initializers.glorot_uniform(), p_shape)
        self.post3 = self.param('post3', nn.initializers.glorot_uniform(), p_shape)
        self.post4 = self.param('post4', nn.initializers.glorot_uniform(), p_shape)
        
        self.cos_kernel1 = self.param('cos_kernel1', nn.initializers.glorot_uniform(), kernel_shape)
        self.sin_kernel1 = self.param('sin_kernel1', nn.initializers.glorot_uniform(), kernel_shape)
        self.cos_kernel2 = self.param('cos_kernel2', nn.initializers.glorot_uniform(), kernel_shape)
        self.sin_kernel2 = self.param('sin_kernel2', nn.initializers.glorot_uniform(), kernel_shape)
        self.cos_kernel3 = self.param('cos_kernel3', nn.initializers.glorot_uniform(), kernel_shape)
        self.sin_kernel3 = self.param('sin_kernel3', nn.initializers.glorot_uniform(), kernel_shape)
        self.cos_kernel4 = self.param('cos_kernel4', nn.initializers.glorot_uniform(), kernel_shape)
        self.sin_kernel4 = self.param('sin_kernel4', nn.initializers.glorot_uniform(), kernel_shape)

    def __call__(self, inputs):
        # Separate real and imaginary parts of inputs
        R, I = inputs[:, 0, :], inputs[:, 1, :]
        
        rdata = lambda d: jnp.take(d, self.r_index)
        
        Rs = jax.vmap(rdata)(R)
        Rs = jnp.reshape(Rs, [-1, self.nx, self.nx])
        Is = jax.vmap(rdata)(I)
        Is = jnp.reshape(Is, [-1, self.nx, self.nx])
        
        def helper(pre, post, kernel2, kernel1, data):
            return jnp.matmul(post, jnp.multiply(kernel2, jnp.matmul(jnp.multiply(data, pre), kernel1)))  
        
        output_polar = helper(self.pre1, self.post1, self.cos_kernel1, self.cos_kernel2, Rs) \
                     + helper(self.pre2, self.post2, self.sin_kernel1, self.sin_kernel2, Rs) \
                     + helper(self.pre3, self.post3, self.cos_kernel3, self.sin_kernel3, Is) \
                     + helper(self.pre4, self.post4, self.sin_kernel4, self.cos_kernel4, Is)
        
        output_polar = jnp.reshape(output_polar, (-1, self.nx**2, 1))
        
        # Convert from polar to Cartesian coordinates
        def polar_to_cart(x):
            x = self.cart_mat @ x
            return jnp.reshape(x, (self.neta, self.neta, 1))
        
        return jax.vmap(polar_to_cart)(output_polar)

class UncompressedModel(nn.Module):
    nx: int
    neta: int
    cart_mat: jnp.ndarray
    r_index: np.ndarray 
    
    def setup(self):
        self.fstar_layer = Fstar(nx=self.nx, neta=self.neta, cart_mat=self.cart_mat, r_index=self.r_index)
        self.convs = [nn.Conv(features=6, kernel_size=(3, 3), padding='SAME') for _ in range(9)]
        self.final_conv = nn.Conv(features=1, kernel_size=(3, 3), padding='SAME')

    def __call__(self, inputs):
        y1 = self.fstar_layer(inputs[:, :, :, 0])
        y2 = self.fstar_layer(inputs[:, :, :, 1])
        y3 = self.fstar_layer(inputs[:, :, :, 2])
        
        y = jnp.concatenate([y1, y2, y3], axis=-1)

        for conv_layer in self.convs:
            tmp = conv_layer(y)
            tmp = jax.nn.relu(tmp)
            y = jnp.concatenate([y, tmp], axis = -1)
        
        y = self.final_conv(y)

        return y[:,:,:,0]
