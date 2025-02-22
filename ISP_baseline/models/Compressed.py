import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import sparse
import flax.linen as nn

# Precomputing indices used for grouping neighboring blocks prior to applying Layer Hs.
def build_permutation_indices(L, l):
    delta = 2**(L-l-1)
    tmp = np.tile(np.arange(2)*delta, delta)
    tmp += np.repeat(np.arange(delta), 2)
    tmp = np.tile(tmp, 2**l)
    tmp += np.repeat(np.arange(2**l)*(2**(L-l)), 2**(L-l))
    return jnp.asarray(tmp)

# Precomputing indices used for redistributing blocks according to the transformation represented by x -> M*xM.
def build_switch_indices(L):
    L = L // 2
    tmp = np.arange(2**L)*(2**L)
    tmp = np.tile(tmp, 2**L)
    tmp += np.repeat(np.arange(2**L), 2**L)
    return jnp.asarray(tmp)

class V(nn.Module):
    r: int

    @nn.compact
    def __call__(self, x):
        
        n, s = x.shape[1], x.shape[2]

        init_fn = nn.initializers.glorot_uniform()
        vr1 = self.param('vr1', init_fn, (n, s, self.r))
        vi1 = self.param('vi1', init_fn, (n, s, self.r))
        vr2 = self.param('vr2', init_fn, (n, s, self.r))
        vi2 = self.param('vi2', init_fn, (n, s, self.r))
        vr3 = self.param('vr3', init_fn, (n, s, self.r))
        vi3 = self.param('vi3', init_fn, (n, s, self.r))
        vr4 = self.param('vr4', init_fn, (n, s, self.r))
        vi4 = self.param('vi4', init_fn, (n, s, self.r))

        x_re, x_im = x[..., 0], x[..., 1]

        y_re_1 = jnp.einsum('...iaj,ajk->...iak', x_re, vr1)
        y_re_1 = jnp.einsum('abj...i,bjk->abk...i', y_re_1, vr1)
        y_re_2 = jnp.einsum('...iaj,ajk->...iak', x_re, vi1)
        y_re_2 = jnp.einsum('abj...i,bjk->abk...i', y_re_2, vi1)
        y_re_3 = jnp.einsum('...iaj,ajk->...iak', x_im, vi2)
        y_re_3 = jnp.einsum('abj...i,bjk->abk...i', y_re_3, vr2)
        y_re_4 = jnp.einsum('...iaj,ajk->...iak', x_im, vr2)
        y_re_4 = jnp.einsum('abj...i,bjk->abk...i', y_re_4, vi2)
        y_re = y_re_1+y_re_2+y_re_3+y_re_4
        
        y_im_1 = jnp.einsum('...iaj,ajk->...iak', x_im, vr3)
        y_im_1 = jnp.einsum('abj...i,bjk->abk...i', y_im_1, vr3)
        y_im_2 = jnp.einsum('...iaj,ajk->...iak', x_im, vi3)
        y_im_2 = jnp.einsum('abj...i,bjk->abk...i', y_im_2, vi3)
        y_im_3 = jnp.einsum('...iaj,ajk->...iak', x_re, vi4)
        y_im_3 = jnp.einsum('abj...i,bjk->abk...i', y_im_3, vr4)
        y_im_4 = jnp.einsum('...iaj,ajk->...iak', x_re, vr4)
        y_im_4 = jnp.einsum('abj...i,bjk->abk...i', y_im_4, vi4)
        y_im = y_im_1+y_im_2+y_im_3+y_im_4
        
        y = jnp.stack([y_re, y_im], axis=-1)
        
        return y

class H(nn.Module):
    perm_idx: jnp.ndarray
    
    @nn.compact
    def __call__(self, x):
        # Placeholder for actual input shape dependent variables
        m = x.shape[1] // 2
        s = x.shape[2] * 2

        # Define weights
        init_fn = nn.initializers.glorot_uniform()
        hr1 = self.param('hr1', init_fn, (m, s, s))
        hi1 = self.param('hi1', init_fn, (m, s, s))
        hr2 = self.param('hr2', init_fn, (m, s, s))
        hi2 = self.param('hi2', init_fn, (m, s, s))
        hr3 = self.param('hr3', init_fn, (m, s, s))
        hi3 = self.param('hi3', init_fn, (m, s, s))
        hr4 = self.param('hr4', init_fn, (m, s, s))
        hi4 = self.param('hi4', init_fn, (m, s, s))

        # Apply permutations
        x = x.take(self.perm_idx, axis=1).take(self.perm_idx, axis=3)
        
        # Reshape operation
        x = x.reshape((-1, m, s, m, s, 2))
        # Split real and imaginary parts for processing
        x_re, x_im = x[..., 0], x[..., 1]
        
        y_re_1 = jnp.einsum('...iaj,ajk->...iak', x_re, hr1)
        y_re_1 = jnp.einsum('abj...i,bjk->abk...i', y_re_1, hr1)
        y_re_2 = jnp.einsum('...iaj,ajk->...iak', x_re, hi1)
        y_re_2 = jnp.einsum('abj...i,bjk->abk...i', y_re_2, hi1)
        y_re_3 = jnp.einsum('...iaj,ajk->...iak', x_im, hi2)
        y_re_3 = jnp.einsum('abj...i,bjk->abk...i', y_re_3, hr2)
        y_re_4 = jnp.einsum('...iaj,ajk->...iak', x_im, hr2)
        y_re_4 = jnp.einsum('abj...i,bjk->abk...i', y_re_4, hi2)
        y_re = y_re_1+y_re_2+y_re_3+y_re_4
        
        y_im_1 = jnp.einsum('...iaj,ajk->...iak', x_im, hr3)
        y_im_1 = jnp.einsum('abj...i,bjk->abk...i', y_im_1, hr3)
        y_im_2 = jnp.einsum('...iaj,ajk->...iak', x_im, hi3)
        y_im_2 = jnp.einsum('abj...i,bjk->abk...i', y_im_2, hi3)
        y_im_3 = jnp.einsum('...iaj,ajk->...iak', x_re, hi4)
        y_im_3 = jnp.einsum('abj...i,bjk->abk...i', y_im_3, hr4)
        y_im_4 = jnp.einsum('...iaj,ajk->...iak', x_re, hr4)
        y_im_4 = jnp.einsum('abj...i,bjk->abk...i', y_im_4, hi4)
        y_im = y_im_1+y_im_2+y_im_3+y_im_4
        
        y = jnp.stack([y_re, y_im], axis=-1)

        n = m * 2
        r = s // 2
        y = y.reshape((-1, n, r, n, r, 2))

        return y

class M(nn.Module):
    @nn.compact
    def __call__(self, x):
        n, r = x.shape[1], x.shape[2]

        # Initialize weights
        init_fn = nn.initializers.glorot_uniform()
        mr1 = self.param('mr1', init_fn, (n, r, r))
        mi1 = self.param('mi1', init_fn, (n, r, r))
        mr2 = self.param('mr2', init_fn, (n, r, r))
        mi2 = self.param('mi2', init_fn, (n, r, r))
        mr3 = self.param('mr3', init_fn, (n, r, r))
        mi3 = self.param('mi3', init_fn, (n, r, r))
        mr4 = self.param('mr4', init_fn, (n, r, r))
        mi4 = self.param('mi4', init_fn, (n, r, r))

        x_re, x_im = x[..., 0], x[..., 1]

        y_re_1 = jnp.einsum('...iaj,ajk->...iak', x_re, mr1)
        y_re_1 = jnp.einsum('abj...i,bjk->abk...i', y_re_1, mr1)
        y_re_2 = jnp.einsum('...iaj,ajk->...iak', x_re, mi1)
        y_re_2 = jnp.einsum('abj...i,bjk->abk...i', y_re_2, mi1)
        y_re_3 = jnp.einsum('...iaj,ajk->...iak', x_im, mi2)
        y_re_3 = jnp.einsum('abj...i,bjk->abk...i', y_re_3, mr2)
        y_re_4 = jnp.einsum('...iaj,ajk->...iak', x_im, mr2)
        y_re_4 = jnp.einsum('abj...i,bjk->abk...i', y_re_4, mi2)
        y_re = y_re_1+y_re_2+y_re_3+y_re_4
        
        y_im_1 = jnp.einsum('...iaj,ajk->...iak', x_im, mr3)
        y_im_1 = jnp.einsum('abj...i,bjk->abk...i', y_im_1, mr3)
        y_im_2 = jnp.einsum('...iaj,ajk->...iak', x_im, mi3)
        y_im_2 = jnp.einsum('abj...i,bjk->abk...i', y_im_2, mi3)
        y_im_3 = jnp.einsum('...iaj,ajk->...iak', x_re, mi4)
        y_im_3 = jnp.einsum('abj...i,bjk->abk...i', y_im_3, mr4)
        y_im_4 = jnp.einsum('...iaj,ajk->...iak', x_re, mr4)
        y_im_4 = jnp.einsum('abj...i,bjk->abk...i', y_im_4, mi4)
        y_im = y_im_1+y_im_2+y_im_3+y_im_4
        
        y = jnp.stack([y_re, y_im], axis=-1)

        return y

class G(nn.Module):
    perm_idx: jnp.ndarray

    @nn.compact
    def __call__(self, x):
        # Dimensions need to be dynamically inferred from 'x'
        m = x.shape[1] // 2
        s = x.shape[2] * 2

        # Initialize weights
        init_fn = nn.initializers.glorot_uniform()
        gr1 = self.param('gr1', init_fn, (m, s, s))
        gi1 = self.param('gi1', init_fn, (m, s, s))
        gr2 = self.param('gr2', init_fn, (m, s, s))
        gi2 = self.param('gi2', init_fn, (m, s, s))
        gr3 = self.param('gr3', init_fn, (m, s, s))
        gi3 = self.param('gi3', init_fn, (m, s, s))
        gr4 = self.param('gr4', init_fn, (m, s, s))
        gi4 = self.param('gi4', init_fn, (m, s, s))

        # Reshape and perform operations
        x = x.reshape((-1, m, s, m, s, 2))
        x_re, x_im = x[..., 0], x[..., 1]

        y_re_1 = jnp.einsum('...iaj,ajk->...iak', x_re, gr1)
        y_re_1 = jnp.einsum('abj...i,bjk->abk...i', y_re_1, gr1)
        y_re_2 = jnp.einsum('...iaj,ajk->...iak', x_re, gi1)
        y_re_2 = jnp.einsum('abj...i,bjk->abk...i', y_re_2, gi1)
        y_re_3 = jnp.einsum('...iaj,ajk->...iak', x_im, gi2)
        y_re_3 = jnp.einsum('abj...i,bjk->abk...i', y_re_3, gr2)
        y_re_4 = jnp.einsum('...iaj,ajk->...iak', x_im, gr2)
        y_re_4 = jnp.einsum('abj...i,bjk->abk...i', y_re_4, gi2)
        y_re = y_re_1+y_re_2+y_re_3+y_re_4
        
        y_im_1 = jnp.einsum('...iaj,ajk->...iak', x_im, gr3)
        y_im_1 = jnp.einsum('abj...i,bjk->abk...i', y_im_1, gr3)
        y_im_2 = jnp.einsum('...iaj,ajk->...iak', x_im, gi3)
        y_im_2 = jnp.einsum('abj...i,bjk->abk...i', y_im_2, gi3)
        y_im_3 = jnp.einsum('...iaj,ajk->...iak', x_re, gi4)
        y_im_3 = jnp.einsum('abj...i,bjk->abk...i', y_im_3, gr4)
        y_im_4 = jnp.einsum('...iaj,ajk->...iak', x_re, gr4)
        y_im_4 = jnp.einsum('abj...i,bjk->abk...i', y_im_4, gi4)
        y_im = y_im_1+y_im_2+y_im_3+y_im_4

        y = jnp.stack([y_re, y_im], axis=-1)

        # Final reshape and permutation
        n, r = m * 2, s // 2
        y = y.reshape((-1, n, r, n, r, 2))
        y = y.take(self.perm_idx, axis=1).take(self.perm_idx, axis=3)

        return y

class U(nn.Module):
    s: int  # Size parameter

    @nn.compact
    def __call__(self, x):
        # Extracting the shapes for weight initialization
        n, r = x.shape[1], x.shape[2]
        nx = n*self.s
        
        # Weight initialization
        init_fn = nn.initializers.glorot_uniform()
        ur1 = self.param('ur1', init_fn, (n, r, self.s))
        ui1 = self.param('ui1', init_fn, (n, r, self.s))
        ur2 = self.param('ur2', init_fn, (n, r, self.s))
        ui2 = self.param('ui2', init_fn, (n, r, self.s))
        ur3 = self.param('ur3', init_fn, (n, r, self.s))
        ui3 = self.param('ui3', init_fn, (n, r, self.s))
        ur4 = self.param('ur4', init_fn, (n, r, self.s))
        ui4 = self.param('ui4', init_fn, (n, r, self.s))

        # Splitting real and imaginary parts
        x_re, x_im = x[..., 0], x[..., 1]

        # Performing the einsum operations
        y_re_1 = jnp.einsum('...iaj,ajk->...iak', x_re, ur1)
        y_re_1 = jnp.einsum('abj...i,bjk->abk...i', y_re_1, ur1)
        y_re_2 = jnp.einsum('...iaj,ajk->...iak', x_re, ui2)
        y_re_2 = jnp.einsum('abj...i,bjk->abk...i', y_re_2, ui2)
        y_re_3 = jnp.einsum('...iaj,ajk->...iak', x_im, ui3)
        y_re_3 = jnp.einsum('abj...i,bjk->abk...i', y_re_3, ur3)
        y_re_4 = jnp.einsum('...iaj,ajk->...iak', x_im, ur4)
        y_re_4 = jnp.einsum('abj...i,bjk->abk...i', y_re_4, ui4)
        # Final sum of y_re components
        y_re = y_re_1 + y_re_2 + y_re_3 + y_re_4

        return y_re.reshape((-1, nx, nx))

class Fstar(nn.Module):
    L: int
    s: int
    r: int
    NUM_RESNET: int
    cart_mat: jnp.ndarray
    r_index: jnp.ndarray
    
    def setup(self):
        self.n = 2**self.L
        self.nx = (2**self.L)*self.s
        self.neta = (2**self.L)*self.s
        self.V = V(self.r)
        self.Hs = [H(build_permutation_indices(self.L, l)) for l in range(self.L-1, self.L//2-1, -1)]
        self.Ms = [M() for _ in range(self.NUM_RESNET)]
        self.Gs = [G(build_permutation_indices(self.L, l)) for l in range(self.L//2, self.L)]
        self.U = U(self.s)
        self.switch_idx = build_switch_indices(self.L)

    def __call__(self, inputs):
        y = inputs.take(self.r_index, axis=1)
        y = jnp.reshape(y, (-1, self.n, self.s, self.n, self.s, 2))
        
        y = self.V(y)
        for h in self.Hs:
            y = h(y)
        y = y.take(self.switch_idx, axis=1).take(self.switch_idx, axis=3)
        #for m in range(self.NUM_RESNET):
        #    y = y + self.Ms[2*m+1](nn.relu(self.Ms[2*m](y)))
        #    if not (m+1)==self.NUM_RESNET:
        #        y = nn.relu(y)
        for m in self.Ms:
            y = m(y) if m is self.Ms[-1] else y + nn.relu(m(y))
            
        for g in self.Gs:
            y = g(y)
        y = self.U(y)
        
        y = jnp.diagonal(y, axis1 = 1, axis2 = 2)
        output_polar = jnp.reshape(y, (-1, self.nx**2, 1))
        
        #def helper(input):
        #    y = jnp.take(input, self.r_index, axis=0)
        #    y = jnp.reshape(y, (-1, self.n, self.s, self.n, self.s, 2))
        #    
        #    y = self.V(y)
        #    
        #    for h in self.Hs:
        #        y = h(y)
        #             
        #    y = y.take(self.switch_idx, axis=1).take(self.switch_idx, axis=3)
        #    for m in self.Ms:
        #        y = m(y) if m is self.Ms[-1] else y + nn.relu(m(y))
        #
        #    for g in self.Gs:
        #        y = g(y)
        #
        #    y = self.U(y)
        #    
        #    y = jnp.diagonal(y, axis1 = 1, axis2 = 2)
        #    return jnp.reshape(y, (-1, self.nx**2, 1))   # Convert from polar to Cartesian coordinates
        #
        #output_polar = jax.vmap(helper)(inputs)
        def polar_to_cart(x):
            x = self.cart_mat @ x
            return jnp.reshape(x, (self.neta, self.neta, 1))
        
        return jax.vmap(polar_to_cart)(output_polar)

class CompressedModel(nn.Module):
    L: int
    s: int
    r: int
    NUM_RESNET: int
    NUM_CONV: int
    cart_mat: jnp.ndarray
    r_index: jnp.ndarray
    
    def setup(self):
        self.fstar_layer0 = Fstar(L=self.L, s=self.s, r=self.r, NUM_RESNET = self.NUM_RESNET, cart_mat=self.cart_mat, r_index=self.r_index)
        self.fstar_layer1 = Fstar(L=self.L, s=self.s, r=self.r, NUM_RESNET = self.NUM_RESNET, cart_mat=self.cart_mat, r_index=self.r_index)
        self.fstar_layer2 = Fstar(L=self.L, s=self.s, r=self.r, NUM_RESNET = self.NUM_RESNET, cart_mat=self.cart_mat, r_index=self.r_index)
        self.convs = [nn.Conv(features=6, kernel_size=(3, 3), padding='SAME') for _ in range(self.NUM_CONV)]
        self.final_conv = nn.Conv(features=1, kernel_size=(3, 3), padding='SAME')

    def __call__(self, inputs):
        y0 = self.fstar_layer0(inputs[:, :, :, 0])
        y1 = self.fstar_layer1(inputs[:, :, :, 1])
        y2 = self.fstar_layer2(inputs[:, :, :, 2])
        
        y = jnp.concatenate([y0, y1, y2], axis = -1)

        for conv_layer in self.convs:
            tmp = conv_layer(y)
            tmp = nn.relu(tmp)
            y = jnp.concatenate([y, tmp], axis = -1)
        
        y = self.final_conv(y)

        return y[:,:,:,0]
