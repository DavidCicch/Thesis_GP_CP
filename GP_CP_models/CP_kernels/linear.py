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

class Linear_CP(jk.base.AbstractKernel):
    """The linear kernel with change points.

    Key reference is MacKay 1998 - "Introduction to Gaussian processes".
    """

    def __init__(self, base_kernel, x0 = [], temp = 0) -> None:
        self.base_kernel = base_kernel
        self.temp = temp
        self.x0 = x0
        self._stationary = True
        self.name = 'Linear'

    def __call__(
        self, params: Dict, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        if x.shape != ():
            return self.cross_covariance(params, x, y)
        if x.shape == ():
            return self.check_side_mult(x, y, params)
    
    def cross_covariance(self, params: Dict, x, y):
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with variance parameter :math:`\\sigma`

        .. math::
            k(x, y) = \\sigma^2 x^{T}y

        Args:
            x (jax.Array): The left hand argument of the kernel function's call.
            y (jax.Array): The right hand argument of the kernel function's call
            params (dict): Parameter set for which the kernel should be evaluated on.
        Returns:
            Array: The value of :math:`k(x, y)`
        """

        K = jax.vmap(lambda x_, params: jax.vmap(lambda y_: self.check_side_mult(x_, y_, params))(y), in_axes=(0, None))(x, params)
        new_K = K + self.temp*jnp.eye(x.shape[0], y.shape[0])
        return new_K
    
    def check_side_mult(self, x_, y_, params):

        def returnxcp(xcp, params, x, y):
            new_params = dict(
                        bias = params['bias'][xcp],
                          variance = params['variance'][xcp],
                          )
            cov = new_params['bias'] + new_params["variance"] * x * y
            return cov.squeeze()
        
        def zero_func(xcp, params, x_, y_):
            return 0.
        
        xcp = jnp.sum(jnp.greater(x_, params["num"]))
        ycp = jnp.sum(jnp.greater(y_, params["num"]))
        
        val = jax.lax.cond(xcp == ycp, returnxcp, zero_func, xcp, params, x_, y_)
        
        return val

    def init_params(self, key: jrnd.KeyArray) -> Dict:
        return {"bias": jnp.array([1.0]),
                "variance": jnp.array([1.0])}
    

class Linear_CP_mult_output(jk.base.AbstractKernel):
    """The linear kernel with multiple change points, 
    enabled for multiple outputs dependent on one domain variable.

    Key reference is MacKay 1998 - "Introduction to Gaussian processes".
    """

    def __init__(self, base_kernel, x0 = [], temp = 0, n = 1) -> None:
        self.base_kernel = base_kernel
        self.temp = temp
        self.x0 = x0
        self._stationary = True
        self.name = 'Linear'
        self.n = n

    def __call__(
        self, params: Dict, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        if x.shape != ():
            return self.cross_covariance(params, x, y)
        if x.shape == ():
            return self.check_side_mult(x, y, params)
    
    def cross_covariance(self, params: Dict, x, y):
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with variance parameter :math:`\\sigma`

        .. math::
            k(x, y) = \\sigma^2 x^{T}y

        Args:
            x (jax.Array): The left hand argument of the kernel function's call.
            y (jax.Array): The right hand argument of the kernel function's call
            params (dict): Parameter set for which the kernel should be evaluated on.
        Returns:
            Array: The value of :math:`k(x, y)`
        """

        self.y_len = len(x)/self.n

        K = jax.vmap(lambda x_, params: jax.vmap(lambda y_: self.check_side_mult(x_, y_, params))(y), in_axes=(0, None))(x, params)
        new_K = K + self.temp*jnp.eye(x.shape[0], y.shape[0])
        return new_K
    
    def check_output_dim(self, x_, y_, params):

        xcp = x_//self.y_len
        ycp = y_//self.y_len
        
        val = jax.lax.cond(xcp == ycp, self.check_side_mult, self.zero_func, xcp, params, x_, y_, xcp)

        return val
    
    def check_side_mult(self, x_, y_, params, i):

        xcp = jnp.sum(jnp.greater(x_, params[f"num_{i}"]))
        ycp = jnp.sum(jnp.greater(y_, params[f"num_{i}"]))
        
        val = jax.lax.cond(xcp == ycp, self.returnxcp, self.zero_func, xcp, params, x_, y_, i)
        
        return val
    
    def returnxcp(self, xcp, params, x, y, i):
            new_params = dict(
                        bias = params[f'bias_{i}'][xcp],
                        variance = params[f'variance_{i}'][xcp],
                        )
            cov = new_params['bias'] + new_params["variance"] * x * y
            return cov.squeeze()
        
    def zero_func(self, xcp, params, x_, y_, i):
        return 0.  

    def init_params(self, key: jrnd.KeyArray) -> Dict:
        return {"bias": jnp.array([1.0]),
                "variance": jnp.array([1.0])}