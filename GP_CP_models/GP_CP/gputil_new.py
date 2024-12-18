# Copyright 2023- The Uncertainty in Complex Systems contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from GP_CP_models.inference_new import update_correlated_gaussian, update_metropolis
from GP_CP_models.GP_CP.meanfunctions import Zero
from GP_CP_models.GP_CP.likelihoods import AbstractLikelihood, Gaussian

from typing import Callable, Union, Dict, Any, Optional, Iterable, Mapping, Tuple
import jax
from jaxtyping import Float, Array
from jax.random import PRNGKey
import jax.numpy as jnp
import jax.random as jrnd
import jaxkern as jk
import distrax as dx



from jax.tree_util import tree_flatten, tree_unflatten
from distrax._src.distributions.distribution import Distribution
from distrax._src.bijectors.bijector import Bijector

__all__ = ['sample_predictive', 
           'sample_prior', 
           'update_gaussian_process', 
           'update_gaussian_process_cov_params', 
           'update_gaussian_process_mean_params', 
           'update_gaussian_process_obs_params']

jitter = 1e-5

def sample_predictive(key: PRNGKey,
                      x: Array,
                      z: Array,
                      target: Array,
                      cov_fn: Callable,
                      mean_params: Dict = None,
                      cov_params: Dict = None,
                      mean_fn: Callable = Zero(),
                      obs_noise = None):
    """Sample latent f for new points x_pred given one posterior sample.

    See Rasmussen & Williams. We are sampling from the posterior predictive for
    the latent GP f, at this point not concerned with an observation model yet.

    We have [f, f*]^T ~ N(0, KK), where KK is a block matrix:

    KK = [[K(x, x), K(x, x*)], [K(x, x*)^T, K(x*, x*)]]

    This results in the conditional

    f* | x, x*, f ~ N(mu, cov), where

    mu = K(x*, x)K(x,x)^-1 f
    cov = K(x*, x*) - K(x*, x) K(x, x)^-1 K(x, x*)

    Args:
        key: The jrnd.PRNGKey object
        x_pred: The prediction locations x*
        state_variables: A sample from the posterior

    Returns:
        A single posterior predictive sample f*

    """

    if obs_noise is not None:
        if jnp.isscalar(obs_noise) or jnp.ndim(obs_noise) == 0:
            diagonal_noise = obs_noise**2 * jnp.eye(x.shape[0],)
        else:
            diagonal_noise = jnp.diagflat(obs_noise**2)
    else:
        diagonal_noise = 0

    mean = mean_fn.mean(params=mean_params, x=z)
    Kxx = cov_fn.cross_covariance(params=cov_params, x=x, y=x)
    Kzx = cov_fn.cross_covariance(params=cov_params, x=z, y=x)
    Kzz = cov_fn.cross_covariance(params=cov_params, x=z, y=z)

    Kxx += jitter * jnp.eye(*Kxx.shape)
    Kzx += jitter * jnp.eye(*Kzx.shape)
    Kzz += jitter * jnp.eye(*Kzz.shape)

    L = jnp.linalg.cholesky(Kxx + diagonal_noise)
    v = jnp.linalg.solve(L, Kzx.T)
    predictive_var = Kzz - jnp.dot(v.T, v)
    predictive_var += jitter * jnp.eye(*Kzz.shape)
    C = jnp.linalg.cholesky(predictive_var)
    def get_sample(u_, target_):
        alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, target_))
        predictive_mean = mean + jnp.dot(Kzx, alpha)
        return predictive_mean + jnp.dot(C, u_)

    #
    if jnp.ndim(target) == 3:            
        _, nu, d = target.shape
        u = jrnd.normal(key, shape=(len(z), nu, d))
        samples = jax.vmap(jax.vmap(get_sample, in_axes=1), in_axes=1)(u, target)
        return samples.transpose([2, 0, 1])
    elif jnp.ndim(target) == 1:
        u = jrnd.normal(key, shape=(len(z),))
        return get_sample(u, target)
    else:
        raise NotImplementedError(f'Shape of target must be (n,) or (n, nu, d)',
        f'but {target.shape} was provided.')

#
def sample_prior(key: PRNGKey,
                 x: Array,
                 cov_params: Dict,
                 cov_fn: Callable,
                 mean_params: Dict = None,
                 mean_fn: Callable = Zero(),
                 nd: Tuple[int, ...] = None,
                 jitter: Float = 1e-6):
    """Draw a sample f ~ GP(m, k)

    If `nd` is provided, the resulting sample is of shape (n, ) + nd. The mean
    and covariance are broadcasted over the first dimension.

    """
    n = x.shape[0]
    output_shape = (n, )
    if nd is not None:
        output_shape += nd

    mu = mean_fn.mean(params=mean_params, x=x)
    cov = cov_fn.cross_covariance(params=cov_params,
                                    x=x,
                                    y=x) + jitter * jnp.eye(n)
    L = jnp.linalg.cholesky(cov)
    z = jrnd.normal(key, shape=output_shape)
    V = jnp.tensordot(L, z, axes=(1, 0))
    f = jnp.moveaxis(jnp.add(mu,
                             jnp.moveaxis(V, 0, -1)),
                     -1, 0)
    return f

#
def update_gaussian_process(key: PRNGKey, 
                            f_current: Array, 
                            loglikelihood_fn: Callable, 
                            X: Array,
                            mean_fn: Callable = Zero(),
                            cov_fn: Callable = jk.RBF(),
                            mean_params: Dict = None,
                            cov_params: Dict = None):
    n = f_current.shape[0]
    mean = mean_fn.mean(params=mean_params, x=X)
    cov = cov_fn.cross_covariance(params=cov_params, x=X, y=X) + jitter * jnp.eye(n)
    if jnp.ndim(f_current) > 1:
        _, nu, d = f_current.shape
        num_el = n*nu*d
        f_current = jnp.reshape(f_current, num_el)
        f_new, f_info = update_correlated_gaussian(key,
                                                   f_current,
                                                   loglikelihood_fn,
                                                   mean,
                                                   cov,
                                                   (nu, d))
        return jnp.reshape(f_new, (n, nu, d)), f_info
    else:
        return update_correlated_gaussian(key,
                                          f_current,
                                          loglikelihood_fn,
                                          mean,
                                          cov)

#
def update_gaussian_process_hyper_params(key: PRNGKey,
                                          X: Array,
                                          num_params,
                                          num_priors, 
                                          hyper_params: Dict = None,
                                          hyperpriors: Dict = None, 
                                          temp: float = 0):
    priors_flat, priors_treedef = tree_flatten(hyperpriors, lambda l: isinstance(l, (Distribution, Bijector)))

    def logdensity_fn_(hyper_params_):
        log_pdf1 = 0     
        log_pdf2 = 0           
        values_flat, _ = tree_flatten(hyper_params_)
        for value, dist in zip(values_flat, priors_flat):
            log_pdf1 += jnp.sum(dist.log_prob(value))

        log_pdf2 = num_priors.log_prob(num_params, values_flat[0])

        log_pdf = log_pdf1+log_pdf2

        return log_pdf

    #
    return update_metropolis(key, logdensity_fn_, hyper_params, stepsize=0.01)


def update_gaussian_process_cov_params(key: PRNGKey,
                                          X: Array,
                                          f: Array,
                                          mean_fn: Callable = Zero(),
                                          cov_fn: Callable = jk.RBF(),
                                          mean_params: Dict = None,
                                          cov_params: Dict = None,
                                          hyperpriors: Dict = None, 
                                          temp: float = 0):
    """Updates the parameters of a Gaussian process covariance function.

    """

    n = X.shape[0]
    mu = mean_fn.mean(params=mean_params, x=X)
    priors_flat, priors_treedef = tree_flatten(hyperpriors, lambda l: isinstance(l, (Distribution, Bijector)))

    def logdensity_fn_(cov_params_):
        jitter = 1e-5
        log_pdf1 = 0     
        log_pdf2 = 0           
        values_flat, _ = tree_flatten(cov_params_)
        for value, dist in zip(values_flat, priors_flat):
            log_pdf1 += jnp.sum(dist.log_prob(value))

        cov_ = cov_fn.cross_covariance(params=cov_params_, x=X, y=X) + jitter * jnp.eye(n)
        if jnp.ndim(f) == 1:
            log_pdf2 += dx.MultivariateNormalFullCovariance(mu, cov_).log_prob(f)
        elif jnp.ndim(f) == 3:
            log_pdf2 += jnp.sum(jax.vmap(jax.vmap(dx.MultivariateNormalFullCovariance(mu, cov_).log_prob, in_axes=1), in_axes=1)(f))
        else:
            raise NotImplementedError(f'Expected f to be of size (n,) or (n, nu, d),',
                                      f'but size {f.shape} was provided.')
        
        log_pdf = log_pdf1+log_pdf2
        return log_pdf

    #
    return update_metropolis(key, logdensity_fn_, cov_params, stepsize=0.01)


def update_gaussian_process_cov_params_num_hyper(key: PRNGKey,
                                          X: Array,
                                          f: Array,
                                          mean_fn: Callable = Zero(),
                                          cov_fn: Callable = jk.RBF(),
                                          mean_params: Dict | list = None,
                                          cov_params: Dict | list = None,
                                          hyper_params: Dict | list = None,
                                          hyperpriors: Dict | list = None, 
                                          temp: float = 0,
                                          kernel_index = None):
    """Updates the parameters of a Gaussian process covariance function.

    """

    n = X.shape[0]
    mu = mean_fn.mean(params=mean_params, x=X)

    hyper_flat, _ = tree_flatten(hyper_params) 
    
    cov_params_subset = dict(num = cov_params['num'])
    new_priors = dict(num = hyperpriors['num'])
    
    priors_flat, priors_treedef = tree_flatten(new_priors, lambda l: isinstance(l, (Distribution, Bijector)))
                                        
    def logdensity_fn_(cov_params_):
        jitter = 1e-5
        log_pdf1 = 0     
        log_pdf2 = 0           
        values_flat, _ = tree_flatten(cov_params_)

        for value, dist in zip(values_flat, priors_flat):
            log_pdf1 += jnp.nansum(dist.log_prob(value, hyper_flat[0]))
        
        
        scale = 1000000
        
        add_temp = 0.1 * ((scale**(1-temp)-1)/(scale-1))
        cov_ = cov_fn.cross_covariance(params=cov_params, x=X, y=X) + (jitter + add_temp) * jnp.eye(n)
        if jnp.ndim(f) == 1:
            log_pdf2 += dx.MultivariateNormalFullCovariance(mu, cov_).log_prob(f)
        elif jnp.ndim(f) == 3:
            log_pdf2 += jnp.sum(jax.vmap(jax.vmap(dx.MultivariateNormalFullCovariance(mu, cov_).log_prob, in_axes=1), in_axes=1)(f))
        else:
            raise NotImplementedError(f'Expected f to be of size (n,) or (n, nu, d),',
                                      f'but size {f.shape} was provided.')
        
        log_pdf = log_pdf1+log_pdf2
        

        return log_pdf
   
    
    return update_metropolis(key, logdensity_fn_, cov_params_subset, stepsize=0.01)

def update_gaussian_process_cov_params_lv_hyper(key: PRNGKey,
                                          X: Array,
                                          f: Array,
                                          mean_fn: Callable = Zero(),
                                          cov_fn: Callable = jk.RBF(),
                                          mean_params: Dict | list = None,
                                          cov_params: Dict | list = None,
                                          hyper_params: Dict | list = None,
                                          hyperpriors: Dict | list = None, 
                                          temp: float = 0):
    """Updates the parameters of a Gaussian process covariance function.

    """

    n = X.shape[0]
    mu = mean_fn.mean(params=mean_params, x=X)
    
    hyper_flat, _ = tree_flatten(hyper_params)


    cov_params_subset = {i:cov_params[i] for i in cov_params if i!='num'}
    new_priors = {i:hyperpriors[i] for i in hyperpriors if i!='num'}

    priors_flat, priors_treedef = tree_flatten(new_priors, lambda l: isinstance(l, (Distribution, Bijector)))
    
    def logdensity_fn_(cov_params_):
        jitter = 1e-5
        log_pdf1 = 0     
        log_pdf2 = 0           
        values_flat, _ = tree_flatten(cov_params_)

       
        for value, dist in zip(values_flat, priors_flat):    
            log_pdf1 += jnp.nansum(dist.log_prob(value)) + jnp.nansum(hyperpriors['num'].log_prob(cov_params['num'], hyper_flat[0]))

        scale = 1000000
        
        add_temp = 0.1 * ((scale**(1-temp)-1)/(scale-1))
        
        cov_ = cov_fn.cross_covariance(params=cov_params, x=X, y=X) + (jitter + add_temp) * jnp.eye(n)
        if jnp.ndim(f) == 1:
            log_pdf2 += dx.MultivariateNormalFullCovariance(mu, cov_).log_prob(f)
        elif jnp.ndim(f) == 3:
            log_pdf2 += jnp.sum(jax.vmap(jax.vmap(dx.MultivariateNormalFullCovariance(mu, cov_).log_prob, in_axes=1), in_axes=1)(f))
        else:
            raise NotImplementedError(f'Expected f to be of size (n,) or (n, nu, d),',
                                      f'but size {f.shape} was provided.')
        
        log_pdf = log_pdf1+log_pdf2
        
        return log_pdf
    
    return update_metropolis(key, logdensity_fn_, cov_params_subset, stepsize=0.01)

#
def update_gaussian_process_mean_params(key: PRNGKey,
                                        X: Array,
                                        f: Array,
                                        mean_fn: Callable = Zero(),
                                        cov_fn: Callable = jk.RBF(),
                                        mean_params: Dict = None,
                                        cov_params: Dict = None,
                                        hyperpriors: Dict = None):
    """Updates the parameters of a Gaussian process mean function.

    TODO: use same tree-flattening approach as for cov_params

    """

    n = X.shape[0]
    cov = cov_fn.cross_covariance(params=cov_params, x=X, y=X) + jitter * jnp.eye(n)
    def logdensity_fn_(mean_params_):
        log_pdf = 0
        for param, val in mean_params_.items():
            log_pdf += jnp.sum(hyperpriors[param].log_prob(val))
        mean_ = mean_fn.mean(params=mean_params_, x=X)
        if jnp.ndim(f) == 1:
            log_pdf += dx.MultivariateNormalFullCovariance(mean_, cov).log_prob(f)
        elif jnp.ndim(f) == 3:            
            log_pdf += jnp.sum(jax.vmap(jax.vmap(dx.MultivariateNormalFullCovariance(mean_, cov).log_prob, in_axes=1), in_axes=1)(f))
        else:
            raise NotImplementedError(f'Expected f to be of size (n,) or (n, nu, d),',
                                      f'but size {f.shape} was provided.')
        return log_pdf
    #
    return update_metropolis(key, logdensity_fn_, mean_params, stepsize=0.01)

#
def update_gaussian_process_obs_params(key: PRNGKey, y: Array,
                                       f: Array,
                                       temperature: Float = 1.0,
                                       likelihood: AbstractLikelihood = Gaussian(),
                                       obs_params: Dict = None,
                                       hyperpriors: Dict = None):
    """Updates the parameters of the observation model.

    TODO: use same tree-flattening approach as for cov_params

    """
    def print_log(log_pdf):
        jax.debug.print("log_prob(obs) = {}", log_pdf)

    def nothing(log_pdf):
        pass

    def logdensity_fn_(obs_params_):
        log_pdf = 0
        for param, val in obs_params_.items():
            log_pdf += jnp.sum(hyperpriors[param].log_prob(val))
        log_pdf += temperature*jnp.sum(likelihood.log_prob(params=obs_params_, f=f, y=y))
        # jax.lax.cond(temperature == 1, print_log, nothing, log_pdf)
        return log_pdf
    #
    key, subkey = jrnd.split(key)
    return update_metropolis(subkey, logdensity_fn_, obs_params, stepsize=0.01)

# 