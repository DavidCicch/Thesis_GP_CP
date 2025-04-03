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

class White_CP(jk.base.AbstractKernel):

    def __init__(self, base_kernel, x0 = [], temp = 0) -> None:
        self.base_kernel = base_kernel
        self.temp = temp
        self.x0 = x0
        self._stationary = True
        self.name = 'White'

    def __call__(
        self, params: Dict, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        if x.shape != ():
            return self.cross_covariance(params, x, y)
        if x.shape == ():
            return self.check_side_mult(x, y, params)
    
    def cross_covariance(self, params: Dict, x, y, alt = False):
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with variance :math:`\\sigma`

        .. math::
            k(x, y) = \\sigma^2 \\delta(x-y)

        Args:
            params (Dict): Parameter set for which the kernel should be evaluated on.
            x (Float[Array, "1 D"]): The left hand argument of the kernel function's call.
            y (Float[Array, "1 D"]): The right hand argument of the kernel function's call.

        Returns:
            Float[Array, "1"]: The value of :math:`k(x, y)`.
        """
        if alt:
            K = jax.vmap(lambda x_, params: jax.vmap(lambda y_: self.check_side_mult_alt(x_, y_, params))(y), in_axes=(0, None))(x, params)
            new_K = K + self.temp*jnp.eye(x.shape[0], y.shape[0])
            return new_K
        else:
            K = jax.vmap(lambda x_, params: jax.vmap(lambda y_: self.check_side_mult(x_, y_, params))(y), in_axes=(0, None))(x, params)
            new_K = K + self.temp*jnp.eye(x.shape[0], y.shape[0])
            return new_K
        
    def check_side_mult(self, x_, y_, params):

        def returnxcp(xcp, params, x, y):
            new_params = dict(
                          variance = params['variance'][xcp],
                          )
            cov = jnp.all(jnp.equal(x, y)) * new_params["variance"]
            return cov.squeeze()
        
        def zero_func(xcp, params, x_, y_):
            return 0.
        
        xcp = jnp.sum(jnp.greater(x_, params["num"]))
        ycp = jnp.sum(jnp.greater(y_, params["num"]))
        
        val = jax.lax.cond(xcp == ycp, returnxcp, zero_func, xcp, params, x_, y_)
        
        return val
    
    def check_side_mult_alt(self, x_, y_, params):

        def returnxcp():
            return 1.
        
        def zero_func():
            return 0.
        
        xcp = jnp.sum(jnp.greater(x_, params["num"]))
        ycp = jnp.sum(jnp.greater(y_, params["num"]))
        
        val = jax.lax.cond(xcp == ycp, returnxcp, zero_func)
        
        return val

    def init_params(self, key: jrnd.KeyArray) -> Dict:
        return {"variance": jnp.array([1.0])}