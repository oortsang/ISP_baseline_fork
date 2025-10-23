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
        """
        Args:
            inputs (jnp.ndarray): Input tensor with shape (batch, 2, features), where the first axis of each sample
                                  corresponds to [real, imaginary] components.
        Returns:
            jnp.ndarray: Output tensor in Cartesian coordinates with shape (batch, neta**2, 1)
        """
        # Separate real and imaginary parts of inputs
        R, I = inputs[:, 0, :], inputs[:, 1, :]

        rdata = lambda d: jnp.take(d, self.r_index)

        Rs = jax.vmap(rdata)(R)
        Rs = jnp.reshape(Rs, [-1, self.nx, self.nx])
        Is = jax.vmap(rdata)(I)
        Is = jnp.reshape(Is, [-1, self.nx, self.nx])

        def helper(pre, post, kernel2, kernel1, data):
            """
            Args:
                pre: Pre-processing parameter.
                post: Post-processing parameter.
                kernel2: Kernel parameter
                kernel1: Kernel parameter
                data: Input data tensor.
            Returns:
                jnp.ndarray: Result of the matrix multiplications and element-wise operations.
            """
            return jnp.matmul(post, jnp.multiply(kernel2, jnp.matmul(jnp.multiply(data, pre), kernel1)))

        output_polar = helper(self.pre1, self.post1, self.cos_kernel1, self.cos_kernel2, Rs) \
                     + helper(self.pre2, self.post2, self.sin_kernel1, self.sin_kernel2, Rs) \
                     + helper(self.pre3, self.post3, self.cos_kernel3, self.sin_kernel3, Is) \
                     + helper(self.pre4, self.post4, self.sin_kernel4, self.cos_kernel4, Is)

        output_polar = jnp.reshape(output_polar, (-1, self.nx**2, 1))

        def polar_to_cart(x):
            """
            Args:
                x (jnp.ndarray): Input polar coordinate data.
            Returns:
                jnp.ndarray: Cartesian coordinate data reshaped to (neta, neta, 1).
            """
            x = self.cart_mat @ x
            return jnp.reshape(x, (self.neta, self.neta, 1))

        return jax.vmap(polar_to_cart)(output_polar)

        # (OOT, 2025-10-09) Can we un-vectorize it to reduce VRAM usage?
        # No, that doesn't seem to do anything
        # return jnp.stack([polar_to_cart(output_i) for output_i in output_polar], axis=0)


class UncompressedModel(nn.Module):
    nx: int
    neta: int
    cart_mat: jnp.ndarray
    r_index: np.ndarray

    def setup(self):
        self.fstar_layer0 = Fstar(nx=self.nx, neta=self.neta, cart_mat=self.cart_mat, r_index=self.r_index)
        self.fstar_layer1 = Fstar(nx=self.nx, neta=self.neta, cart_mat=self.cart_mat, r_index=self.r_index)
        self.fstar_layer2 = Fstar(nx=self.nx, neta=self.neta, cart_mat=self.cart_mat, r_index=self.r_index)
        self.convs = [nn.Conv(features=6, kernel_size=(3, 3), padding='SAME') for _ in range(9)]
        self.final_conv = nn.Conv(features=1, kernel_size=(3, 3), padding='SAME')

    def __call__(self, inputs):
        """
        Args:
            inputs (jnp.ndarray): Input tensor with at least 4 dimensions, where the last dimension represents different channels.
        Returns:
            jnp.ndarray: Output tensor after processing through Fstar layers and convolutional layers, with the final channel squeezed.
        """
        y0 = self.fstar_layer0(inputs[:, :, :, 0])
        y1 = self.fstar_layer1(inputs[:, :, :, 1])
        y2 = self.fstar_layer2(inputs[:, :, :, 2])

        y = jnp.concatenate([y0, y1, y2], axis=-1)

        for conv_layer in self.convs:
            tmp = conv_layer(y)
            tmp = jax.nn.relu(tmp)
            y = jnp.concatenate([y, tmp], axis=-1)

        y = self.final_conv(y)

        return y[:, :, :, 0]

# # Discretization of Omega (n_eta * n_eta).
# neta = (2**L)*s

# # Number of sources/detectors (n_sc).
# # Discretization of the domain of alpha in polar coordinates (n_theta * n_rho).
# # For simplicity, these values are set equal (n_sc = n_theta = n_rho), facilitating computation.
# nx = (2**L)*s



### (OOT, 2025-10-06) Below this, I've introduced an alternate interface
# that I hope is a bit more flexible (i.e., set number of frequencies and hyperparameters)

class UncompressedModelFlexible(nn.Module):
    """Modified interface of the Uncompressed model to accept more frequencies and
    more control over the hyperparameter choices
    """
    nx: int     # number of source/receivers; we usually say N_r or N_s
    neta: int   # number of gridpoints for the scatterint potentials; our code uses N_x for this
    cart_mat: jnp.ndarray
    r_index: np.ndarray

    # New parameters
    nk: int = 3 # number of frequencies; this is new
    # Architecture
    N_cnn_layers: int = 9
    N_cnn_channels: int = 6
    kernel_size: int = 3

    # I/O normalization?
    in_norm:  bool = False
    out_norm: bool = False
    in_mean:  jnp.array = None
    in_std:   jnp.array = None
    out_mean: jnp.array = None
    out_std:  jnp.array = None

    def setup(self):
        # Do I need to register these things with Jax for proper functioning?
        self.fstar_layers = [
            Fstar(nx=self.nx, neta=self.neta, cart_mat=self.cart_mat, r_index=self.r_index)
            for _ in range(self.nk)
        ]
        kernel_shape_2d = (self.kernel_size, self.kernel_size)
        self.convs = [
            nn.Conv(
                features=self.N_cnn_channels,
                kernel_size=kernel_shape_2d,
                padding='SAME'
            )
            for _ in range(self.N_cnn_layers)
        ]
        self.final_conv = nn.Conv(features=1, kernel_size=kernel_shape_2d, padding='SAME')

    def __call__(self, inputs):
        """
        Args:
            inputs (jnp.ndarray): Input tensor with at least 4 dimensions, where the last dimension represents different channels.
        Returns:
            jnp.ndarray: Output tensor after processing through Fstar layers and convolutional layers, with the final channel squeezed.
        """
        if self.in_norm:
            inputs = (inputs - self.in_mean) / self.in_std # will the axes work out?

        # Process each channel separately using Fstar layers
        # and concatenate outputs along channel dimension
        y = jnp.concatenate(
            [
                self.fstar_layers[i](inputs[:, :, :, i])
                for i in range(self.nk)
            ],
            axis=-1,
        )
        for conv_layer in self.convs:
            tmp = conv_layer(y)
            tmp = jax.nn.relu(tmp)
            y = jnp.concatenate([y, tmp], axis=-1)

        y = self.final_conv(y)
        output = y[:, :, :, 0]

        if self.out_norm:
            output = (output - self.out_mean) / self.out_std # will the axes work out?

        return output

