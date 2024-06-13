import os
from typing import Dict, List, Optional, Union

import numpy as np

from beetroots.modelling.priors.spatial_prior_params import SpatialPriorParams
from beetroots.sampler.utils.sampler_params import MySamplerParams
from beetroots.simulations.abstract_simulation import Simulation
from beetroots.simulations.astro import data_validation
from beetroots.simulations.astro.forward_map_setup.abstract_nn import SimulationNN
from beetroots.simulations.astro.observation_setup.abstract_toy_case import SimulationToyCase
from beetroots.simulations.astro.sampler_setup.abstract_sampler_hierarchical import SimulationHierarchical


class SimulationToyCaseNNHierachical(SimulationNN, SimulationToyCase, SimulationHierarchical):
    __slots__ = (
        "path_output_sim",
        "path_img",
        "path_raw",
        "path_data_csv",
        "path_data_csv_in",
        "path_data_csv_out",
        "path_data_csv_out_mcmc",
        "path_data_csv_out_optim_map",
        "path_data_csv_out_optim_mle",
        "N",
        "D",
        "D_no_kappa",
        "L",
        "list_names",
        "list_names_plot",
        "cloud_name",
        "max_workers",
        "list_lines_fit",
        "list_lines_valid",
        "Theta_true_scaled",
        "map_shaper",
        "plots_estimator",
    )

    def setup(
        self,
        forward_model_name: str,
        force_use_cpu: bool,
        fixed_params: Dict[str, Optional[float]],
        is_log_scale_params: Dict[str, bool],
        #
        sigma_a_float: float,
        sigma_m_float: float,
        omega_float: float,
        #
        params_component_distributions,
        dict_target_distributions_match_components,
    ):
        self.N = int(self.cloud_name.split("N")[1]) ** 2

        self.list_lines_valid = []

        scaler, forward_map = self.setup_forward_map(
            forward_model_name=forward_model_name,
            force_use_cpu=force_use_cpu,
            dict_fixed_params=fixed_params,
            dict_is_log_scale_params=is_log_scale_params,
        )

        sigma_a = sigma_a_float * np.ones((self.N, self.L))
        sigma_m = sigma_m_float * np.ones((self.N, self.L))
        omega = omega_float * np.ones((self.N, self.L))

        syn_map, y = self.setup_observation(
            scaler=scaler,
            forward_map=forward_map,
            sigma_a=sigma_a,
            sigma_m=sigma_m,
            omega=omega,
        )

        # run setup
        dict_models, scaler, params_plot_setup = self.setup_target_distribution(
            scaler=scaler,
            forward_map=forward_map,
            y=y,
            sigma_a=sigma_a,
            sigma_m=sigma_m,
            omega=omega,
            syn_map=syn_map,
            params_component_distributions=params_component_distributions,
            dict_target_distributions_match_components=dict_target_distributions_match_components,
        )


        return (
            dict_models,
            scaler,
            params_plot_setup,
        )

    def main(self, params: dict, path_data_cloud: str) -> None:

        (
            dict_models,
            scaler,
            params_plot_setup,
        ) = simulation.setup(
            **params["forward_model"],
            #
            sigma_a_float=params["sigma_a_float"],
            sigma_m_float=np.log(params["sigma_m_float_linscale"]),
            omega_float=3 * params["sigma_a_float"],
            #
            params_component_distributions=params['component_distributions'],
            dict_target_distributions_match_components = params['target_distributions'],
        )

        for model_name in list(dict_models.keys()):
            folder_path = f"{self.path_raw}/{model_name}"
            if not os.path.isdir(folder_path):
                os.mkdir(folder_path)

        simulation.save_and_plot_setup(
            **params_plot_setup,
            scaler=scaler,
        )
        # * Optim MAP
        if params["to_run_optim_map"]:
            simulation.inversion_optim_map(
                dict_models=dict_models,
                scaler=scaler,
                my_sampler_params=MySamplerParams(**params["sampling_params"]["map"]),
                can_run_in_parallel=params["forward_model"]["force_use_cpu"],
                **params["run_params"]["map"],
            )

        # * MCMC
        if params["to_run_mcmc"]:
            simulation.inversion_mcmc(
                dict_models=dict_models,
                scaler=scaler,
                my_sampler_params=MySamplerParams(**params["sampling_params"]["mcmc"]),
                can_run_in_parallel=params["forward_model"]["force_use_cpu"],
                **params["run_params"]["mcmc"],
            )
        return


if __name__ == "__main__":
    yaml_file, path_data, path_models, path_outputs = SimulationToyCaseNNHierachical.parse_args()

    params = SimulationToyCaseNNHierachical.load_params(path_data, yaml_file)

    SimulationToyCaseNNHierachical.check_input_params_file(
        params,
        data_validation.schema_astro_hierarchical,
    )

    simulation = SimulationToyCaseNNHierachical(
        **params["simu_init"],
        yaml_file=yaml_file,
        path_data=path_data,
        path_outputs=path_outputs,
        path_models=path_models,
        forward_model_fixed_params=params["forward_model"]["fixed_params"],
    )

    simulation.main(
        params=params,
        path_data_cloud=path_data,
    )
