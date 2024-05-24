from typing import Optional

try:
    import cupy as xp
except:
    import numpy as xp

from beetroots.sampler.saver.abstract_saver import Saver


class HierarchicalSaver(Saver):
    def initialize_memory(
        self,
        T_MC: int,
        t: int,
        current: dict = dict(),
        additional_sampling_log: dict = dict(),
    ) -> None:
        """initializes the memory with the correct shapes

        Parameters
        ----------
        T_MC : int
            size of the markov chain to be sampled
        t : int
            current iteration index
        """
        # TODO: replace everything to the hierarchical approach with unknown number of variables.

        if self.batch_size is None:
            self.batch_size = T_MC

        self.t_last_init = t * 1
        self.next_batch_size = min(self.batch_size, (T_MC - t + 1) // self.freq_save)
        # print(t, self.next_batch_size)
        self.final_next_batch_size = self.next_batch_size

        for key in current.keys():
            self.memory[key] = np.zeros((self.final_next_batch_size, *vars[key].shape))

            if self.save_forward_map_evals:
                for k, v in forward_map_evals.items():
                    if np.all(["grad" not in k, "hess_diag" not in k]):
                        self.memory[f"list_{k}"] = np.zeros(
                            (self.final_next_batch_size,) + v.shape
                        )

                for k, v in forward_map_evals_u.items():
                    if np.all(["grad" not in k, "hess_diag" not in k]):
                        self.memory[f"list_u_{k}"] = np.zeros(
                            (self.final_next_batch_size,) + v.shape
                        )

            for k, v in nll_utils.items():
                if np.all(["nll_" not in k, "grad" not in k, "hess_diag" not in k]):
                    self.memory[f"list_{k}"] = np.zeros(
                        (self.final_next_batch_size,) + v.shape
                    )

            for k, v in nll_utils_u.items():
                if np.all(["nll_" not in k, "grad" not in k, "hess_diag" not in k]):
                    self.memory[f"list_u_{k}"] = np.zeros(
                        (self.final_next_batch_size,) + v.shape
                    )

            for k, v in dict_objective.items():
                self.memory[f"list_{k}"] = np.zeros((self.final_next_batch_size,) + v.shape)

            for k, v in dict_objective_u.items():
                self.memory[f"list_u_{k}"] = np.zeros(
                    (self.final_next_batch_size,) + v.shape
                )

            for k, v in additional_sampling_log.items():
                if isinstance(v, np.ndarray):
                    self.memory[f"list_{k}"] = np.zeros(
                        (self.final_next_batch_size,) + v.shape
                    )
                else:
                    self.memory[f"list_{k}"] = np.zeros((self.final_next_batch_size,))

            for k, v in additional_sampling_log_u.items():
                if isinstance(v, np.ndarray):
                    self.memory[f"list_u_{k}"] = np.zeros(
                        (self.final_next_batch_size,) + v.shape
                    )
                else:
                    self.memory[f"list_u_{k}"] = np.zeros((self.final_next_batch_size,))

            self.memory["list_rng_state"] = np.zeros(
                (self.final_next_batch_size, 32),
                dtype=np.uint8,
            )
            self.memory["list_rng_inc"] = np.zeros(
                (self.final_next_batch_size, 32),
                dtype=np.uint8,
            )

    def update_memory(
        self,
        t: int,
        current: dict,
        dict_objective: dict = dict(),
        additional_sampling_log: dict = dict(),
        rng_state_array: Optional[xp.ndarray] = None,
        rng_inc_array: Optional[xp.ndarray] = None,
    ) -> None:
        # TODO: replace everything to the hierarchical approach with unknown number of variables.

        """updates the memory with new information. All of the potential entries are optional except for the current iterate."""
        t_save = (t - self.t_last_init) // self.freq_save

        self.memory["list_Theta"][t_save, :, :] = self.scaler.from_scaled_to_lin(x)
        self.memory["list_U"][t_save, :, :] = u

        if self.save_forward_map_evals:
            for k, v in forward_map_evals.items():
                if np.all(["grad" not in k, "hess_diag" not in k]):
                    self.memory[f"list_{k}"][t_save] = v

            for k, v in forward_map_evals_u.items():
                if np.all(["grad" not in k, "hess_diag" not in k]):
                    self.memory[f"list_{k}"][t_save] = v

        for k, v in nll_utils.items():
            if np.all(["nll_" not in k, "grad" not in k, "hess_diag" not in k]):
                self.memory[f"list_{k}"][t_save] = v

        for k, v in nll_utils_u.items():
            if np.all(["nll_" not in k, "grad" not in k, "hess_diag" not in k]):
                self.memory[f"list_u_{k}"][t_save] = v

        for k, v in dict_objective.items():
            if k not in ["m_a", "s_a", "m_m", "s_m"]:
                self.memory[f"list_{k}"][t_save] = v

        for k, v in dict_objective_u.items():
            if k not in ["m_a", "s_a", "m_m", "s_m"]:
                self.memory[f"list_u_{k}"][t_save] = v

        for k, v in additional_sampling_log.items():
            self.memory[f"list_{k}"][t_save] = v

        for k, v in additional_sampling_log_u.items():
            self.memory[f"list_u_{k}"][t_save] = v

        if (rng_state_array is not None) and (rng_inc_array is not None):
            self.memory["list_rng_state"][t_save] = rng_state_array
            self.memory["list_rng_inc"][t_save] = rng_inc_array