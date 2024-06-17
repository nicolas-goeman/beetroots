import time
from typing import Dict, List, Optional, Union

try:
    import cupy as xp
except:
    import numpy as xp

from beetroots.inversion.results.results_mcmc import ResultsExtractorMCMC
from beetroots.inversion.results.results_optim_map import ResultsExtractorOptimMAP
from beetroots.inversion.run.run_mcmc import RunMCMC
from beetroots.inversion.run.run_optim_map import RunOptimMAP
from beetroots.modelling.likelihoods.approx_censored_add_mult import (
    MixingModelsLikelihood,
)
from beetroots.modelling.likelihoods.gaussian_censored import CensoredGaussianLikelihood
from beetroots.modelling.target_distribution.posterior import Posterior
from beetroots.modelling.target_distribution.full_conditional import FullConditional
from beetroots.modelling.priors.l22_laplacian_prior import L22LaplacianSpatialPrior
from beetroots.modelling.priors.smooth_indicator_prior import SmoothIndicatorPrior
from beetroots.modelling.priors.spatial_prior_params import SpatialPriorParams
from beetroots.sampler.my_gibbs_sampler import MyGibbsSampler
from beetroots.sampler.saver.my_saver import MySaver
from beetroots.sampler.utils.sampler_params import MyGibbsSamplerParams
from beetroots.simulations.astro.sampler_setup.abstract_sampler_approach import (
    SimulationTargetDistributionType,
)
from beetroots.space_transform.transform import MyScaler

import importlib

class SimulationHierarchical(SimulationTargetDistributionType):
    def setup_target_distribution(
        self,
        scaler,
        forward_map,
        y,
        sigma_a,
        sigma_m,
        omega,
        syn_map,
        params_component_distributions: dict[str, dict],
        dict_target_distributions_match_components: dict[str, list],
    ) -> None:
        # component distributions
        component_distributions_names = params_component_distributions.keys()
        component_distributions = dict()

        if "theta_indicator_prior" in component_distributions_names:
            dic = params_component_distributions["theta_indicator_prior"]
            module_ = importlib.import_module(dic['module'])
            class_ = getattr(module_, dic['class_name'])
            kwargs = dic['params']
            lower_bounds_lin = xp.array(kwargs['lower_bounds_lin']) # Linear space
            upper_bounds_lin = xp.array(kwargs['upper_bounds_lin']) # Linear space

            kwargs.update({"D": self.D_sampling, "L": self.L, "N": self.N, "list_idx_sampling": self.list_idx_sampling})

            if dic['do_scaling']:
                lower_bounds = scaler.from_lin_to_scaled(
                    lower_bounds_lin.reshape((1, self.D)),
                ).flatten()
                upper_bounds = scaler.from_lin_to_scaled(
                    upper_bounds_lin.reshape((1, self.D)),
                ).flatten()
            kwargs['lower_bounds'] = lower_bounds
            kwargs['upper_bounds'] = upper_bounds

            component_distributions["theta_indicator_prior"] = class_(**kwargs)
        else: 
            raise ValueError("'theta_indicator_prior' must be a component distribution")
        
        if "theta_spatial_prior" in component_distributions_names:
            dic = params_component_distributions["theta_spatial_prior"]
            module_ = importlib.import_module(dic['module'])
            class_ = getattr(module_, dic['class_name'])
            kwargs = dic['params']
            kwargs['cloud_name'] = self.cloud_name
            kwargs['list_idx_sampling'] = self.list_idx_sampling
            kwargs['df'] = syn_map
            kwargs['spatial_prior_params'] = SpatialPriorParams(kwargs['name'], kwargs['use_next_nearest_neighbors'] , kwargs['initial_regu_weights'])
            kwargs.update({"D": self.D_sampling, "L": self.L, "N": self.N})
            component_distributions["theta_spatial_prior"] = class_(**kwargs)

            dict_sites = component_distributions["theta_spatial_prior"].dict_sites
            list_edges = component_distributions["theta_spatial_prior"].list_edges
            weights = component_distributions["theta_spatial_prior"].weights
        else:
            dict_sites, list_edges, weights = None, None, None
            raise Warning("no spatial prior for theta will be used as no component distribution with the key 'theta_spatial_prior' has been provided.")
        
        if "aux_given_theta" in component_distributions_names:
            dic = params_component_distributions["aux_given_theta"]
            module_ = importlib.import_module(dic['module'])
            class_ = getattr(module_, dic['class_name'])
            kwargs = dic['params']
            kwargs['forward_map'] = forward_map
            kwargs.update({"D": self.D_sampling, "L": self.L, "N": self.N})
            kwargs['sigma_m'] = sigma_m

            component_distributions["aux_given_theta"] = class_(**kwargs)
        else:
            raise ValueError("'aux_given_theta' must be a component distribution")
        
        if "obs_given_aux" in component_distributions_names:
            dic = params_component_distributions["obs_given_aux"]
            module_ = importlib.import_module(dic['module'])
            class_ = getattr(module_, dic['class_name'])
            kwargs = dic['params']
            kwargs['y'] = y
            kwargs['sigma_a'] = sigma_a
            kwargs['omega'] = omega
            kwargs.update({"D": self.D_sampling, "L": self.L, "N": self.N})
            component_distributions["obs_given_aux"] = class_(**kwargs)
        else:
            raise ValueError("'obs_given_aux' must be a component distribution")
                
        # target distributions
        target_distributions = {}

        for i, (target_dist_name, list_component_dists) in enumerate(dict_target_distributions_match_components.items()):
            component_distributions_target = {component_dist_name: component_distributions[component_dist_name] for component_dist_name in list_component_dists}
            if 'theta' in target_dist_name:
                var_shape = (self.N, self.D_sampling)
            elif 'aux' in target_dist_name:
                var_shape = (self.N, self.L)
            else:
                raise ValueError(f"target_dist_name must contain 'theta' or 'aux' but got {target_dist_name}")
            
            target_distribution = FullConditional(
                    D=self.D_sampling,
                    L=self.L,
                    N=self.N,
                    distribution_components=component_distributions_target,
                    var_name=target_dist_name,
                    var_shape=var_shape,
                    dict_sites=dict_sites,
                )
            target_distributions[target_dist_name] = target_distribution

        dict_models = {"hierarchical_aux_theta": target_distributions} # We could instantiate several versions of the problem (different hyperparams for example similarly to the posterior model with the approximation) 

        params_plot_setup = {
            "dict_sites_": dict_sites,
            "y": y*1,
            "sigma_a": sigma_a*1,
            "omega": omega*1,
            "lower_bounds_lin": lower_bounds_lin,
            "upper_bounds_lin": upper_bounds_lin,
        }
 
        proposal_distribution_params = {
            "list_edges": list_edges,
            "weights": weights,
            "upper_bounds": upper_bounds,
            "lower_bounds": lower_bounds,
            "indicator_margin_scale": params_component_distributions["theta_indicator_prior"]['params']['indicator_margin_scale']
            }
        kwargs_proposal_distributions = {
            "full_conditional_theta": proposal_distribution_params,
            "full_conditional_auxiliary": proposal_distribution_params,
        }
        return dict_models, scaler, params_plot_setup, kwargs_proposal_distributions

    def inversion_optim_mle(self):
        pass

    def inversion_optim_map(
        self,
        dict_models: Dict[str, Posterior],
        scaler: MyScaler,
        my_sampler_params: MyGibbsSamplerParams,
        N_MCMC: int,
        T_MC: int,
        T_BI: int,
        batch_size: int = 10,
        freq_save: int = 1,
        start_from: Optional[str] = None,
        can_run_in_parallel: bool = True,
    ) -> None:
        raise NotImplementedError("This method does not work currently. The optimization process for the Gibbs sampler is not implemented yet.")
        tps_init = time.time()

        sampler_ = MyGibbsSampler(my_sampler_params, self.D_sampling, self.L, self.N)

        saver_ = MySaver(
            N=self.N,
            D=self.D,
            D_sampling=self.D_sampling,
            L=self.L,
            scaler=scaler,
            batch_size=batch_size,
            freq_save=freq_save,
            list_idx_sampling=self.list_idx_sampling,
        )

        run_optim_map = RunOptimMAP(self.path_data_csv_out, self.max_workers)
        run_optim_map.main(
            dict_models,
            sampler_,
            saver_,
            scaler,
            N_MCMC,
            T_MC,
            path_raw=self.path_raw,
            start_from=start_from,
            freq_save=freq_save,
            can_run_in_parallel=can_run_in_parallel,
        )

        results_optim_map = ResultsExtractorOptimMAP(
            self.path_data_csv_out_optim_map,
            self.path_img,
            self.path_raw,
            N_MCMC,
            T_MC,
            T_BI,
            freq_save,
            self.max_workers,
        )
        for model_name, posterior in dict_models.items():
            results_optim_map.main(
                posterior=posterior,
                model_name=model_name,
                scaler=scaler,
                #
                list_idx_sampling=self.list_idx_sampling,
                list_fixed_values=self.list_fixed_values,
                #
                estimator_plot=self.plots_estimator,
                Theta_true_scaled=self.Theta_true_scaled,
            )

        duration = time.time() - tps_init  # is seconds
        duration_str = time.strftime("%H:%M:%S", time.gmtime(duration))
        msg = "Simulation and analysis finished. Total duration : "
        msg += f"{duration_str} s\n"
        print(msg)

        list_model_names = list(dict_models.keys())
        return list_model_names

    def inversion_mcmc(
        self,
        dict_models: Dict[str, Posterior],
        scaler: MyScaler,
        my_sampler_params: MyGibbsSamplerParams,
        N_MCMC: int,
        T_MC: int,
        T_BI: int,
        plot_1D_chains: bool,
        plot_2D_chains: bool,
        plot_ESS: bool,
        plot_comparisons_yspace: bool,
        #
        batch_size: int = 10,
        freq_save: int = 1,
        start_from: Optional[Union[str, dict]] = None,
        #
        regu_spatial_N0: Union[int, float] = xp.infty,
        regu_spatial_scale: Optional[float] = 1.0,
        regu_spatial_vmin: Optional[float] = 1e-8,
        regu_spatial_vmax: Optional[float] = 1e8,
        #
        y_valid: Optional[xp.ndarray] = None,
        sigma_a_valid: Optional[xp.ndarray] = None,
        omega_valid: Optional[xp.ndarray] = None,
        sigma_m_valid: Optional[xp.ndarray] = None,
        #
        can_run_in_parallel: bool = True,
        point_challenger: Dict = {},
        list_CI: List[int] = [68, 90, 95, 99],
    ) -> None:
        tps_init = time.time()

        sampler_ = MyGibbsSampler(my_sampler_params, self.D_sampling, self.L, self.N)

        saver_ = MySaver(
            self.N,
            self.D,
            self.D_sampling,
            self.L,
            scaler,
            batch_size=batch_size,
            freq_save=freq_save,
            list_idx_sampling=self.list_idx_sampling,
        )

        run_mcmc = RunMCMC(self.path_data_csv_out, self.max_workers)
        run_mcmc.main(
            dict_models,
            sampler_,
            saver_,
            scaler,
            N_MCMC,
            T_MC,
            T_BI=T_BI,
            path_raw=self.path_raw,
            path_csv_mle=self.path_data_csv_out_optim_mle,
            path_csv_map=self.path_data_csv_out_optim_map,
            start_from=start_from,
            freq_save=freq_save,
            #
            regu_spatial_N0=regu_spatial_N0,
            regu_spatial_scale=regu_spatial_scale,
            regu_spatial_vmin=regu_spatial_vmin,
            regu_spatial_vmax=regu_spatial_vmax,
            #
            can_run_in_parallel=can_run_in_parallel,
        )

        results_mcmc = ResultsExtractorMCMC(
            self.path_data_csv_out_mcmc,
            self.path_img,
            self.path_raw,
            N_MCMC,
            T_MC,
            T_BI,
            freq_save,
            self.max_workers,
        )
        for model_name, posterior in dict_models.items():
            results_mcmc.main(
                posterior=posterior,
                model_name=model_name,
                scaler=scaler,
                list_names=self.list_names_plots,
                list_idx_sampling=self.list_idx_sampling,
                list_fixed_values=self.list_fixed_values,
                #
                plot_1D_chains=plot_1D_chains,
                plot_2D_chains=plot_2D_chains,
                plot_ESS=plot_ESS,
                plot_comparisons_yspace=plot_comparisons_yspace,
                #
                estimator_plot=self.plots_estimator,
                analyze_regularization_weight=xp.isfinite(regu_spatial_N0),
                list_lines_fit=self.list_lines_fit,
                Theta_true_scaled=self.Theta_true_scaled,
                list_lines_valid=self.list_lines_valid,
                y_valid=y_valid,
                sigma_a_valid=sigma_a_valid,
                omega_valid=omega_valid,
                sigma_m_valid=sigma_m_valid,
                point_challenger=point_challenger,
                list_CI=list_CI,
            )

        duration = time.time() - tps_init  # is seconds
        duration_str = time.strftime("%H:%M:%S", time.gmtime(duration))
        msg = "Simulation and analysis finished. Total duration : "
        msg += f"{duration_str} s\n"
        print(msg)

        list_model_names = list(dict_models.keys())
        return list_model_names
