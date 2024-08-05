from jax import Array
from jaxtyping import Float
from jax.random import PRNGKeyArray as PRNGKey
from typing import Callable, Union, Dict, Any, Optional, Iterable, Mapping
import matplotlib.pyplot as plt
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]
from jax.tree_util import tree_flatten, tree_unflatten
from GP_CP_models.GP_CP.gputil_new import sample_prior, sample_predictive

from distrax._src.distributions.distribution import Distribution
from distrax._src.bijectors.bijector import Bijector
from GP_CP_models.GP_CP.meanfunctions import Zero
from GP_CP_models.fullgp import FullLatentGPModelhyper_mult
from GP_CP_models.fullgp import FullMarginalGPModelhyper_mult, FullMarginalGPModelhyper_mult_poisson
import copy 

import jax 
import distrax as dx
import jaxkern as jk
import jax.numpy as jnp
from jax.random import PRNGKey
import jax.random as jrnd
from blackjax import elliptical_slice, rmh
    
class GP_CP_Marginal():
    def __init__(self, X, y: Optional[Array]=None,
                 cov_fn: Optional[Callable]=None,
                 mean_fn: Callable = None,
                 priors: Dict = None, 
                 num_particles: int = None,
                 num_mcmc_steps: int = None,
                 ground_truth: Dict = None,
                 poisson = True):
            if jnp.ndim(X) == 1:
                X = X[:, jnp.newaxis]  
            if cov_fn is None:
                raise ValueError(
                    f'Provide a covariance function for the GP!')
            # Validate arguments
            if y is not None and X.shape[0] > len(y):
                raise ValueError(
                    f'X and y should have the same leading dimension, '
                    f'but X has shape {X.shape} and y has shape {y.shape}.',
                    f'Use the `FullLatentGPModelRepeatedObs` model for repeated inputs.')
            self.X, self.y = X, y        
            self.n = self.X.shape[0]        
            if mean_fn is None:
                mean_fn = Zero()
            self.mean_fn = mean_fn
            self.cov_fn = cov_fn
            self.param_priors = priors
            self.num_particles = num_particles
            self.num_mcmc_steps = num_mcmc_steps
            self.particles = None
            if poisson:
                self.gp_fit = FullMarginalGPModelhyper_mult_poisson(self.X, self.y, cov_fn=self.cov_fn, priors=self.param_priors)
            else:
                self.gp_fit = FullMarginalGPModelhyper_mult(self.X, self.y, cov_fn=self.cov_fn, priors=self.param_priors)

            self.likelihood = None
            if isinstance(self.cov_fn, jk.base.CombinationKernel):
                self.kernel_name = [kernel.name for kernel in cov_fn.kernel_set]
            else:
                self.kernel_name = [cov_fn.name]
            self.ground_truth = ground_truth


    def model_GP(self, key):
        print('Running Marginal GP')
        
        gp_marginal = self.gp_fit  # Implies likelihood=Gaussian()
        key, gpm_key = jrnd.split(key)
        mgp_particles, _, mgp_marginal_likelihood = gp_marginal.inference(gpm_key,
                                                                        mode='gibbs-in-smc',
                                                                        sampling_parameters=dict(num_particles=self.num_particles, num_mcmc_steps=self.num_mcmc_steps),
                                                                        )
        self.particles = mgp_particles
        self.gp_fit = gp_marginal
        self.likelihood = mgp_marginal_likelihood
        
    def plot_post(self, ground_truth=None):
        ''' Only plots up to a maximum of 5 posteriors per default'''
            
        if isinstance(self.particles.particles['kernel'], dict):
            isdict = True
            num_kernels = 1
           
        else:
            isdict = False
            num_kernels = len(self.particles.particles['kernel'])
        for k in range(num_kernels):
            if isdict:
                num_CPs = jnp.max(jnp.sum(~jnp.isnan(self.particles.particles['kernel']['num']), axis = 1))
                tr = copy.deepcopy(self.particles.particles['kernel'])
                del tr['num'] 
                trainables = [name for name in tr.keys()]
            else:
                num_CPs = jnp.max(jnp.sum(~jnp.isnan(self.particles.particles['kernel'][k]['num']), axis = 1))
                tr = copy.deepcopy(self.particles.particles['kernel'][k])
                del tr['num'] 
                trainables = [name for name in tr.keys()]
            if trainables == []:
                raise ValueError(
                    f'No posteriors to plot!')
            
            num_params = len(trainables)

            symbols = [fr'{name[0]}' for name in trainables]
            
            num_CP = jnp.minimum(num_CPs, 5).tolist()
            _, axes = plt.subplots(nrows=num_params, ncols=num_CP+1, constrained_layout=True,
                                figsize=(16, 6))
            
            if num_CP == 0:
                axes = axes[:, jnp.newaxis]
            elif num_params == 1:
                axes = axes[:, jnp.newaxis].T

            for j, var in enumerate(trainables):
                    pd = tr[var]
                    for i in range(num_CP+1):
                        # There are some outliers that skew the axes
                        # pd_u, pd_l = jnp.nanpercentile(pd[:, i], q=99.9), jnp.nanpercentile(pd[:, i], q=0.1)
                        # pd_filtered = jnp.extract(pd[:, i]>pd_l, pd[:, i])
                        # pd_filtered = jnp.extract(pd_filtered<pd_u, pd_filtered)
                        axes[j, i].hist(pd[~jnp.isnan(pd[:, i]), i], bins=30, density=True, color='tab:blue')
                        if ground_truth is not None:
                            if isdict:
                                if len(ground_truth['kernel'][var]) > i:
                                    axes[j, i].axvline(x=ground_truth['kernel'][var][i], ls=':', c='k')
                            else:
                                if len(ground_truth['kernel'][k][var]) > i:
                                    axes[j, i].axvline(x=ground_truth['kernel'][k][var][i], ls=':', c='k')
                        
                        axes[j, i].set_xlabel(r'${:s}$'.format(f'{symbols[j]}_{i}'))


                    axes[j, 0].set_ylabel(var, rotation=0, ha='right')
                
            plt.suptitle(f'Posterior estimate of Bayesian Marginal GP {self.kernel_name[k]} kernel ({self.num_particles} particles)')
            plt.show();

    def _plot_fit(self, key, predict=True, f_true = None, ground_truth = None, particles = None):
        if predict:
            x_pred = jnp.linspace(-0.25, 1.25, num=int(1.5 * len(self.y)))
        else:
            x_pred = jnp.linspace(0, 1, num=len(self.y))

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 4), sharex=True,
                                    sharey=True, constrained_layout=True)
        
    
        key, key_pred = jrnd.split(key)
        if particles is None:
            if isinstance(self.particles, dict):
                parts = self.particles
            else:
                parts = self.particles.particles
            f_pred = self.gp_fit.predict_f(key_pred, x_pred)
        else:
            parts = particles
            f_pred = self.gp_fit.predict_f_particle(key_pred, x_pred, particles)

        num_particles = self.num_particles
        ax = axes[0]
        for i in jnp.arange(0, num_particles, step=10):
            ax.plot(x_pred, f_pred[i, :], alpha=0.1, color='tab:blue')


        colors = plt.cm.jet(jnp.linspace(0.3,1, len(self.kernel_name)))

        ax2 = ax.twinx()
        if isinstance(parts['kernel'], dict):
            pd = parts['kernel']['num']
            new_pd = pd[jnp.logical_not(jnp.isnan(pd))]
            ax2.hist(new_pd, bins=30, density=True, color='tab:blue', alpha=0.5)
            if ground_truth is not None:
                if 'num' in ground_truth['kernel'].keys():
                    for CP in ground_truth['kernel']['num']:
                        ax2.axvline(x=CP, ls=':', c='black')
        else:
            for i, pd in enumerate(parts['kernel']):
                new_pd = pd['num'][jnp.logical_not(jnp.isnan(pd['num']))]
                ax2.hist(new_pd, bins=30, density=True, color=colors[i], label = self.kernel_name[i], alpha=0.5)
                if ground_truth is not None:
                    if 'num' in ground_truth['kernel'][i].keys():
                        for CP in ground_truth['kernel'][i]['num']:
                            ax2.axvline(x=CP, ls=':', c=colors[i])
            ax2.legend()

        ax = axes[1]
        f_mean = jnp.nanmean(f_pred, axis=0)
        if particles is None:
            y_pred = self.gp_fit.predict_y(key_pred, x_pred)
        else:
            y_pred = self.gp_fit.predict_y(key_pred, x_pred, particles)
        
        f_hdi_lower = jnp.nanpercentile(y_pred, q=2.5, axis=0)
        f_hdi_upper = jnp.nanpercentile(y_pred, q=97.5, axis=0)
        
        ax.plot(x_pred, f_mean, color='tab:blue', lw=2)
        ax.fill_between(x_pred, f_hdi_lower, f_hdi_upper,
                        alpha=0.2, color='tab:blue', lw=0)
        
        ax2 = ax.twinx()
        if isinstance(parts['kernel'], dict):
            pd = parts['kernel']['num']
            new_pd = pd[jnp.logical_not(jnp.isnan(pd))]
            ax2.hist(new_pd, bins=30, density=True, color='tab:blue', alpha=0.5)
            if ground_truth is not None:
                if 'num' in ground_truth['kernel'].keys():
                    for CP in ground_truth['kernel']['num']:
                        ax2.axvline(x=CP, ls=':', c='black')
        else:
            for i, pd in enumerate(parts['kernel']):
                new_pd = pd['num'][jnp.logical_not(jnp.isnan(pd['num']))]
                ax2.hist(new_pd, bins=30, density=True, color=colors[i], label = self.kernel_name[i], alpha=0.5)
                if ground_truth is not None:
                    if 'num' in ground_truth['kernel'][i].keys():
                        for CP in ground_truth['kernel'][i]['num']:
                            ax2.axvline(x=CP, ls=':', c=colors[i])
            ax2.legend()
        ax2.set_ylabel('CP probability', fontsize=16)

        for ax in axes:
            if f_true is not None:
                ax.plot(self.X.squeeze(), f_true, 'k', label=r'$f$')
            ax.plot(self.X.squeeze(), self.y, 'rx', label='obs')
            if predict:
                ax.set_xlim([-0.25, 1.25])
            else:
                ax.set_xlim([0., 1.])
            ax.set_ylim([jnp.min(self.y)-.5, jnp.max(self.y)+.5])
            ax.set_xlabel(r'$x$', fontsize=12)
            ax.set_ylabel(r'$y$', fontsize=12)
        return axes

    def plot_fit(self, key, predict=False, f_true = None, ground_truth = None, particles = None):
        axes = self._plot_fit(key, predict, f_true, ground_truth, particles)
        axes[0].set_title('SMC particles', fontsize=16)
        axes[0].set_ylabel('Marginal GP', rotation=0, ha='right', fontsize=14)
        axes[1].set_title('Posterior 95% HDI', fontsize=16)
        plt.show()

    def _plot_num(self):
        fig = plt.figure(figsize=(12, 6))
        colors = plt.cm.jet(jnp.linspace(0.3,1, len(self.kernel_name)))
        counts = jnp.zeros((self.num_particles, len(self.kernel_name)))

        if isinstance(self.particles.particles['kernel'], dict):
            pd = self.particles.particles['kernel']['num']
            counts = counts.at[:, 0].set(jnp.count_nonzero(~jnp.isnan(pd), axis = 1))
            uni_vals = jnp.sort(jnp.concatenate([jnp.unique(counts)-0.5, jnp.unique(counts)+0.5]))
        else:
            for i, pd in enumerate(self.particles.particles['kernel']):
                num_val = pd['num']
                counts = counts.at[:, i].set(jnp.count_nonzero(~jnp.isnan(num_val), axis = 1))
                uni_vals = jnp.sort(jnp.concatenate([jnp.unique(counts)-0.5, jnp.unique(counts)+0.5]))
            
        plt.hist(counts.T, bins=uni_vals, rwidth = 0.5, color=colors, label=self.kernel_name)

    def plot_num(self):
        self._plot_num()
        plt.xlabel("Amount of change points")
        plt.ylabel("Amount of particles")
        plt.title('Amount of Change Points in Marginal GP')
        plt.legend()
        plt.show()

    def number_metric(self, ground_truth):

        if isinstance(ground_truth['kernel'], dict):
            true_number = len(ground_truth['kernel']['num'])
            if isinstance(self.particles.particles['kernel'], dict):
                num_val = self.particles.particles['kernel']['num']
                counts = jnp.mean(jnp.count_nonzero(~jnp.isnan(num_val), axis = 1))
                max_num = jnp.maximum(num_val.shape[1] - true_number, true_number)
                return (counts - true_number)/max_num
            else:
                counts = 0
                for i, kernel in enumerate(self.particles.particles['kernel']):
                    num_val = self.particles.particles['kernel'][i]['num']
                    counts += jnp.mean(jnp.count_nonzero(~jnp.isnan(num_val), axis = 1))
                max_num = jnp.maximum(num_val.shape[1] - true_number, true_number)
                return (counts - true_number)/max_num
        
        metric = jnp.zeros(len(self.particles.particles['kernel']))
        for i, kernel in enumerate(self.particles.particles['kernel']):
            true_number = len(ground_truth['kernel'][i]['num'])
            num_val = kernel['num']
            counts = jnp.mean(jnp.count_nonzero(~jnp.isnan(num_val), axis = 1))
            max_num = jnp.maximum(num_val.shape[1] - true_number, true_number)
            metric = metric.at[i].set((counts - true_number)/max_num)
        return dict(zip(self.kernel_name, metric.tolist()))
    
    def _loc_calculation(self, locs, diffs, true_locations, max_num, max_dist):
        for i, loc in enumerate(locs):       
            if jnp.count_nonzero(~jnp.isnan(loc)) == 0:
                dist = jnp.zeros((len(true_locations), 1))
            else:
                dist = jnp.zeros((len(true_locations), jnp.maximum(1, jnp.count_nonzero(~jnp.isnan(loc)))))
                for j, true_loc in enumerate(true_locations):
                    true_locs = jnp.sort(self.X.squeeze()[jnp.argsort(jnp.abs(self.X.squeeze() - true_loc))[:2]])
                    dist1 = (loc[~jnp.isnan(loc)] - true_locs[0])
                    dist2 = (loc[~jnp.isnan(loc)] - true_locs[1])
                    dist_comp = dist1 * dist2
                    dist_min = jnp.minimum(jnp.abs(dist1), jnp.abs(dist2))
                    dist_min = dist_min.at[dist_comp < 0].set(0)
                    dist = dist.at[j, :].set(jnp.abs(dist_min.squeeze()))
            num_diff = jnp.abs(len(true_locations) - jnp.count_nonzero(~jnp.isnan(loc)))
            sorted_min_dist = jnp.sort(jnp.min(dist, axis = 1))
           
            if num_diff == 0:
                diffs = diffs.at[i].set(jnp.sum(sorted_min_dist)/max_num)
            else: 
                sorted_min_dist = sorted_min_dist.at[-num_diff:].set(max_dist)
                diffs = diffs.at[i].set(jnp.sum(sorted_min_dist)/max_num)
        return jnp.mean(diffs)


    def location_metric(self, ground_truth):
        if isinstance(self.particles.particles['kernel'], dict):
            true_locations = ground_truth['kernel']['num']
            locs = self.particles.particles['kernel']['num']
            diffs = jnp.zeros(len(locs))
            max_num = locs.shape[1]
            max_dist = 1
            return self._loc_calculation(locs, diffs, true_locations, max_num, max_dist)
            
        
        metric = jnp.zeros(len(self.particles.particles['kernel']))
        for i, kernel in enumerate(self.particles.particles['kernel']):
            true_locations = ground_truth['kernel'][i]['num']
            locs = kernel['num']
            diffs = jnp.zeros(len(locs))
            max_num = locs.shape[1]
            max_dist = 1
            metric = metric.at[i].set(self._loc_calculation(locs, diffs, true_locations, max_num, max_dist))
        return dict(zip(self.kernel_name, metric.tolist()))
    

    def likelihood_metric(self, key, particles = None):
        size = len(self.y)
        x_pred = jnp.linspace(-0, 1, num=size)
        key, key_pred = jrnd.split(key)
        if self.gp_fit is not None:
            if particles is None:
                f_pred = self.gp_fit.predict_f(key_pred, x_pred)
            else:
                f_pred = self.gp_fit.predict_f_particle(key_pred, x_pred, particles)
        else: 
            raise ValueError('No GP trained yet!')
        
        f_mean = jnp.nanmean(f_pred, axis=0)
        cov = jnp.zeros(size) + jnp.mean(self.particles.particles['likelihood']['obs_noise'])
        MVN = dx.MultivariateNormalDiag(f_mean, cov)
        return MVN.log_prob(self.y)


    def jaccard_metric(self, ground_truth):
        if isinstance(ground_truth['kernel'], dict):
            true_number = len(ground_truth['kernel']['num'])
            true_locations = jnp.sort(ground_truth['kernel']['num'])
        else:
            true_locations = []
            true_number = 0
            for i, kernel in enumerate(ground_truth['kernel']):
                true_number += len(kernel['num'])
                true_locations.append(jnp.sort(kernel['num']))
            true_locations = jnp.array(true_locations).squeeze()

        def zero_one_matrix(params, x, y):
            def check_side_mult_alt(x_, y_, params):

                def returnxcp():
                    return 1.
                
                def zero_func():
                    return 0.
                
                xcp = jnp.sum(jnp.greater(x_, params["num"]))
                ycp = jnp.sum(jnp.greater(y_, params["num"]))
                
                val = jax.lax.cond(xcp == ycp, returnxcp, zero_func)
                
                return val
            
            K = jax.vmap(lambda x_, params: jax.vmap(lambda y_: check_side_mult_alt(x_, y_, params))(y), in_axes=(0, None))(x, params)
            return K
    
        
        true_locations = dict(num = true_locations)
        true_matrix = zero_one_matrix(params = true_locations, x= self.X, y=self.X)

        if isinstance(self.particles.particles['kernel'], dict):
            kernel = self.particles.particles['kernel']
            num_params = dict(num = kernel['num'])
            cov_param_in_axes = jax.tree_map(lambda l: 0, num_params)
            est_K = jax.vmap(lambda a: 
                             zero_one_matrix(params = a, x= self.X, y=self.X),
                              in_axes=(cov_param_in_axes, ))(num_params)
            diff_K = 1 - jnp.abs(true_matrix - est_K)
            sum_diff = jnp.sum(diff_K.flatten())/(len(self.X)**2 * self.num_particles)
            
            return jnp.mean(sum_diff)
        
        num_params = jnp.zeros(self.num_particles)
        num_params = num_params[:, None]
        for i, kernel in enumerate(self.particles.particles['kernel']):
            
            num_params = jnp.concatenate((num_params, kernel['num']), axis = 1)
        num_params = num_params[:, 1:]
        
        num_params = dict(num = num_params)
        cov_param_in_axes = jax.tree_map(lambda l: 0, num_params)
        est_K = jax.vmap(lambda a: 
                             zero_one_matrix(params = a, x= self.X, y=self.X),
                              in_axes=(cov_param_in_axes, ))(num_params)
        diff_K = 1 - jnp.abs(true_matrix - est_K)
        sum_diff = jnp.sum(diff_K.flatten())/(len(self.X)**2 * self.num_particles)
        
        return jnp.mean(sum_diff)

class GP_CP_Latent(GP_CP_Marginal):
    def __init__(self, X, y: Optional[Array]=None,
                 cov_fn: Optional[Callable]=None,
                 mean_fn: Callable = None,
                 priors: Dict = None, 
                 num_particles: int = None,
                 num_mcmc_steps: int = None,
                 likelihood = None,
                 **kwargs):
        super().__init__(X, y, cov_fn, mean_fn, priors, num_particles, num_mcmc_steps, **kwargs)   
        if [likelihood] is not None:
            self.gp_fit = FullLatentGPModelhyper_mult(self.X, self.y, cov_fn=self.cov_fn, priors=self.param_priors, likelihood=likelihood)
        else:
            self.gp_fit = FullLatentGPModelhyper_mult(self.X, self.y, cov_fn=self.cov_fn, priors=self.param_priors)
    
    def model_GP(self, key, poisson=False):
        print('Running Latent GP')
        kernel = self.cov_fn

        priors = self.param_priors
        gp_latent = self.gp_fit # Implies likelihood=Gaussian()
        key, gpm_key = jrnd.split(key)
        lgp_particles, _, lgp_marginal_likelihood = gp_latent.inference(gpm_key,
                                                                        mode='gibbs-in-smc',
                                                                        sampling_parameters=dict(num_particles=self.num_particles, num_mcmc_steps=self.num_mcmc_steps),
                                                                        poisson = True)
        self.particles = lgp_particles
        self.gp_fit = gp_latent
        self.likelihood = lgp_marginal_likelihood

    def plot_fit(self, key, predict=False, f_true = None, ground_truth = None, particles = None):
        axes = self._plot_fit(key, predict, f_true, ground_truth, particles)
        axes[0].set_title('SMC particles', fontsize=16)
        axes[0].set_ylabel('Latent GP', rotation=0, ha='right', fontsize=14)
        axes[1].set_title('Posterior 95% HDI', fontsize=16)
        plt.show()

    def plot_num(self):
        self._plot_num()
        plt.xlabel("Amount of change points")
        plt.ylabel("Amount of particles")
        plt.title('Amount of Change Points in Latent GP')
        plt.legend()
        plt.show()

    def plot_post(self, ground_truth=None):
        ''' Only plots up to a maximum of 5 posteriors per default'''
            
        if isinstance(self.particles.particles['kernel'], dict):
            isdict = True
            num_kernels = 1
            
        else:
            isdict = False
            num_kernels = len(self.particles.particles['kernel'])
        for k in range(num_kernels):
            if isdict:
                num_CPs = jnp.max(jnp.sum(~jnp.isnan(self.particles.particles['kernel']['num']), axis = 1))
                tr = copy.deepcopy(self.particles.particles['kernel'])
                del tr['num'] 
                trainables = [name for name in tr.keys()]
            else:
                num_CPs = jnp.max(jnp.sum(~jnp.isnan(self.particles.particles['kernel'][k]['num']), axis = 1))
                tr = copy.deepcopy(self.particles.particles['kernel'][k])
                del tr['num'] 
                trainables = [name for name in tr.keys()]
            if trainables == []:
                raise ValueError(
                    f'No posteriors to plot!')
            
            num_params = len(trainables)

            symbols = [fr'{name[0]}' for name in trainables]
            
            num_CP = jnp.minimum(num_CPs, 5).tolist()
            _, axes = plt.subplots(nrows=num_params, ncols=num_CP+1, constrained_layout=True,
                                figsize=(16, 6))
            
            if num_CP == 0:
                axes = axes[:, jnp.newaxis]
            elif num_params == 1:
                axes = axes[:, jnp.newaxis].T

            for j, var in enumerate(trainables):
                    pd = tr[var]
                    for i in range(num_CP+1):
                        # There are some outliers that skew the axes
                        # pd_u, pd_l = jnp.nanpercentile(pd[:, i], q=99.9), jnp.nanpercentile(pd[:, i], q=0.1)
                        # pd_filtered = jnp.extract(pd[:, i]>pd_l, pd[:, i])
                        # pd_filtered = jnp.extract(pd_filtered<pd_u, pd_filtered)
                        axes[j, i].hist(pd[~jnp.isnan(pd[:, i]), i], bins=30, density=True, color='tab:blue')
                        # axes[j, i].hist(pd[:, i][~jnp.isnan(pd[:, i])], bins=30, density=True, color='tab:blue')
                        if ground_truth is not None:
                            if isdict:
                                if len(ground_truth['kernel'][var]) > i:
                                    axes[j, i].axvline(x=ground_truth['kernel'][var][i], ls=':', c='k')
                            else:
                                if len(ground_truth['kernel'][k][var]) > i:
                                    axes[j, i].axvline(x=ground_truth['kernel'][k][var][i], ls=':', c='k')
                        
                        axes[j, i].set_xlabel(r'${:s}$'.format(f'{symbols[j]}_{i}'))                        

                    axes[j, 0].set_ylabel(var, rotation=0, ha='right')
                
            plt.suptitle(f'Posterior estimate of Bayesian Latent GP {self.kernel_name[k]} kernel ({self.num_particles} particles)')
            plt.show();

    

