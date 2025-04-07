import jax
import jaxkern as jk
import jax.numpy as jnp
import jax.random as jrnd
from typing import Callable, Tuple, Union, NamedTuple, Dict, Any, Optional
from jaxtyping import Array, Float
from jax.nn import softmax
from jaxkern.computations import (
    DenseKernelComputation,
)

from typing import Dict, List, Optional

class Matern12_CP(jk.base.AbstractKernel):

    def __init__(self, base_kernel, x0 = [], temp = 0) -> None:
        self.base_kernel = base_kernel
        self.temp = temp
        self.x0 = x0
        self._stationary = True
        self.name = 'Matern12'
    
    #
    def __call__(
        self, params: Dict, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        if x.shape != ():
            return self.cross_covariance(params, x, y)
        if x.shape == ():
            return self.check_side_mult(x, y, params)

    #
    def cross_covariance(self, params: Dict, x, y):
        """Computes the discontinuous cross-covariance.

        The bread-and-butter of the discontinuity analysis removes all 
        correlations between observations on different sides of the threshold 
        x0.

        Args:
            params: Parameters of the base kernel.
            x, y: points to determine covariance for
        Returns:
            an nxm matrix of cross covariances (n = len(x), m = len(y))
        """

        '''New version'''        
        
        K = jax.vmap(lambda x_, params: jax.vmap(lambda y_: self.check_side_mult(x_, y_, params))(y), in_axes=(0, None))(x, params)

        new_K = K + self.temp*jnp.eye(x.shape[0], y.shape[0])

        return new_K
        
    def check_side_mult(self, x_, y_, params: Dict):

        
        def returnxcp(xcp, params, x_, y_):
            params = dict(lengthscale = params['lengthscale'][xcp],
                          variance = params['variance'][xcp])
            # tau = euclidean_distance(x_, y)
            cov = params["variance"]* jnp.exp(-jnp.abs(x_ - y_)/params['lengthscale'])
            return cov
        
        def zero_func(xcp, params, x_, y_):
            return 0.
        
        xcp = jnp.sum(jnp.greater(x_, params['num']))
        ycp = jnp.sum(jnp.greater(y_, params['num']))
        
        val = jax.lax.cond(xcp == ycp, returnxcp, zero_func, xcp, params, x_, y_)
        return val
    #
    def init_params(self, key: jrnd.KeyArray) -> dict:
        self.base_kernel.init_params(key)