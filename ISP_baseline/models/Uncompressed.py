import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import sparse
import flax.linen as nn

import jax.sharding as jshard # above
from  jax.sharding import PartitionSpec as P

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
            # # Original code:
            # return jnp.matmul(post, jnp.multiply(kernel2, jnp.matmul(jnp.multiply(data, pre), kernel1)))

            # term1 = jnp.multiply(data, pre)
            # term2 = jnp.matmul(term1, kernel1)
            # term3 = jnp.multiply(kernel2, term2)
            # term4 = jnp.matmul(post, term3)

            term1 = jnp.einsum("bij,ij->bij", data, pre) # multiply
            term2 = jnp.einsum("bij,jk->bik", term1, kernel1) # matmul
            term3 = jnp.einsum("ik,bik->bik", kernel2, term2) # multiply
            term4 = jnp.einsum("hi,bik->bhk", post, term3) # matmul
            # jax.debug.print(f"data: {data.shape} t1:{term1.shape} t2:{term2.shape} t3:{term3.shape} t4:{term4.shape}")
            return term4

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

        output_cart = jax.vmap(polar_to_cart)(output_polar)
        # jax.debug.print(f"output_polar: {output_polar.shape}, output_cart: {output_cart.shape}")
        return output_cart

class UncompressedModel(nn.Module):
    nx: int
    neta: int
    cart_mat: jnp.ndarray
    r_index: np.ndarray
    grad_checkpoint: bool = True

    def setup(self):
        # Checkpoint if requested using flax/linen
        Fstar_chkpt = (
            nn.remat(Fstar)
            if self.grad_checkpoint
            else Fstar
        )
        self.fstar_layer0 = Fstar_chkpt(nx=self.nx, neta=self.neta, cart_mat=self.cart_mat, r_index=self.r_index)
        self.fstar_layer1 = Fstar_chkpt(nx=self.nx, neta=self.neta, cart_mat=self.cart_mat, r_index=self.r_index)
        self.fstar_layer2 = Fstar_chkpt(nx=self.nx, neta=self.neta, cart_mat=self.cart_mat, r_index=self.r_index)
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

def vm_helper(pre, post, kernel2, kernel1, data):
    """Vmapped helper function for MultifreqFstar
    Args:
        pre  (nk, 1, nx):  Pre-processing parameter.
        post (nk, 1, nx): Post-processing parameter.
        kernel2 (nk, nx, nx): Kernel parameter
        kernel1 (nk, nx, nx): Kernel parameter
        data (nk, nbatch*nx(??), nx, nx): Input data tensor.
    Returns:
        jnp.ndarray: Result of the matrix multiplications and element-wise operations.
            Shape: (nk, nbatch*nx, 1, nx)
    """
    # jax.debug.print(f"data: {data.shape}")
    # un-einsummed
    # def internal_helper(data):
    #     term1 = jnp.multiply(data, pre)
    #     term2 = jnp.matmul(term1, kernel1)
    #     term3 = jnp.multiply(kernel2, term2)
    #     term4 = jnp.matmul(post, term3)
    # return jax.vmap(jax.vmap(internal_helper))(data)
    # term1 = jnp.einsum("fbij,fij->fbij", data, pre) # multiply
    # term2 = jnp.einsum("fbij,fjk->fbik", term1, kernel1) # matmul
    # term3 = jnp.einsum("fik,fbik->fbik", kernel2, term2) # multiply
    # term4 = jnp.einsum("fhi,fbik->fbhk", post, term3) # matmul
    # out = term4
    # term12 = jnp.einsum(
    #     "fbij,fij,fjk->fbik",
    #     data, pre, kernel1,
    # )
    # term34 = jnp.einsum(
    #     "fhi,fik,fbik->fbhk",
    #     post, kernel2, term12,
    # )
    # out = term34
    term1234 = jnp.einsum(
        "fhi,fik,  fbij,fij,fjk->fbhk",
        post, kernel2, data, pre, kernel1,
    )
    out = term1234

    # jax.debug.print(f"data: {data.shape} t1:{term1.shape} t2:{term2.shape} t3:{term3.shape} t4:{term4.shape}")
    return out

class MultifreqFstar(nn.Module):
    """Variant of Fstar that can use multiple GPUs"""
    nx: int
    neta: int
    cart_mat: jnp.ndarray
    r_index: np.ndarray
    nk: int
    # mesh: jshard.Mesh=jshard.make_mesh(jax.device_count(), ("freq",))
    shard: jshard.NamedSharding=jshard.NamedSharding(
        jax.make_mesh((jax.device_count(),), ("freq",)),
        P("freq", None, None, None)
    )

    def setup(self):
        kernel_shape = (self.nk, self.nx, self.nx)
        p_shape = (self.nk, 1, self.nx)

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
        inputs = jnp.transpose(inputs, (3, 0, 1, 2)) # move frequency axis to the first slot
        # inputs = jnp.permute_dims(inputs, (3, 0, 1, 2)) # move frequency axis to the first slot
        # jax.debug.print(f"inputs: {inputs.shape}")
        inputs = jax.device_put(inputs, self.shard)
        # Separate real and imaginary parts of inputs
        R, I = inputs[:, :, 0, :], inputs[:, :, 1, :]

        rdata = lambda d: jnp.take(d, self.r_index)

        Rs = jax.vmap(jax.vmap(rdata))(R)
        Rs = jnp.reshape(Rs, [self.nk, -1, self.nx, self.nx])
        Is = jax.vmap(jax.vmap(rdata))(I)
        Is = jnp.reshape(Is, [self.nk, -1, self.nx, self.nx])

        output_polar = vm_helper(self.pre1, self.post1, self.cos_kernel1, self.cos_kernel2, Rs) \
                     + vm_helper(self.pre2, self.post2, self.sin_kernel1, self.sin_kernel2, Rs) \
                     + vm_helper(self.pre3, self.post3, self.cos_kernel3, self.sin_kernel3, Is) \
                     + vm_helper(self.pre4, self.post4, self.sin_kernel4, self.cos_kernel4, Is)

        output_polar = jnp.reshape(output_polar, (self.nk, -1, self.nx**2, 1)) # (nk, nbatch, nx^2, 1)

        def polar_to_cart(x):
            """
            Args:
                x (jnp.ndarray): Input in polar coordinates
            Returns:
                jnp.ndarray: Cartesian coordinate data reshaped to (neta, neta, 1).
            """
            out_tmp = self.cart_mat @ x
            return jnp.reshape(out_tmp, (self.neta, self.neta, 1))
        
        output_cart = jax.vmap(jax.vmap(polar_to_cart))(output_polar).reshape(self.nk, -1, self.neta, self.neta)
        output_cart = jnp.transpose(output_cart, (1,2,3, 0)) # freq dim last
        return output_cart

# # Discretization of Omega (n_eta * n_eta).
# neta = (2**L)*s

# # Number of sources/detectors (n_sc).
# # Discretization of the domain of alpha in polar coordinates (n_theta * n_rho).
# # For simplicity, these values are set equal (n_sc = n_theta = n_rho), facilitating computation.
# nx = (2**L)*s

# ### (OOT, 2025-10-06) Below this, I've introduced an alternate interface
# # that I hope is a bit more flexible (i.e., set number of frequencies and hyperparameters)
        
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
    grad_checkpoint: bool = True

    # I/O normalization?
    in_norm:  bool = False
    out_norm: bool = False
    in_mean:  jnp.array = None
    in_std:   jnp.array = None
    out_mean: jnp.array = None
    out_std:  jnp.array = None

    def setup(self):
        # Checkpoint if requested using flax/linen
        # Fstar_chkpt = (
        #     nn.remat(Fstar)
        #     if self.grad_checkpoint
        #     else Fstar
        # )
        # self.fstar_layers = [
        #     Fstar_chkpt(
        #         nx=self.nx,
        #         neta=self.neta,
        #         cart_mat=self.cart_mat,
        #         r_index=self.r_index,
        #     )
        #     for _ in range(self.nk)
        # ]
        MultifreqFstar_chkpt = (
            nn.remat(MultifreqFstar)
            if self.grad_checkpoint
            else MultifreqFstar
        )
        self.mf_fstar = MultifreqFstar_chkpt(
            nx=self.nx,
            neta=self.neta,
            cart_mat=self.cart_mat,
            r_index=self.r_index,
            nk=self.nk,
        )

        # apply_model = lambda data, params: fstar(params)(data)
        # fstar_params = [jax.tree_util.tree_flatten(elem) for elem in fstar_layers_parameter_list]
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
        # y = jnp.concatenate(
        #     [
        #         self.fstar_layers[i](inputs[:, :, :, i])
        #         for i in range(self.nk)
        #     ],
        #     axis=-1,
        # )
        # jax.debug.print(f"Uncompressed apply inputs: {inputs.shape}")
        y = self.mf_fstar(inputs[:, :, :, :])
        # jax.debug.print(f"y shape: {y.shape}")

        for conv_layer in self.convs:
            tmp = conv_layer(y)
            tmp = jax.nn.relu(tmp)
            y = jnp.concatenate([y, tmp], axis=-1)

        y = self.final_conv(y)
        output = y[:, :, :, 0]

        if self.out_norm:
            output = (output + self.out_mean) * self.out_std # will the axes work out?

        return output
