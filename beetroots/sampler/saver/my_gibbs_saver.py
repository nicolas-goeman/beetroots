from typing import Optional

try:
    import cupy as xp
except:
    import numpy as xp

from beetroots.sampler.saver.abstract_saver import Saver


class MyGibbsSaver(Saver):
    def initialize_memory(
        self,
        T_MC: int,
        t: int,
        current: dict = dict(),
        nlpdf_utils: dict = dict(),
        dict_objective: dict = dict(),
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
        if self.batch_size is None:
            self.batch_size = T_MC

        self.t_last_init = t * 1
        self.next_batch_size = min(self.batch_size, (T_MC - t + 1) // self.freq_save)
        # print(t, self.next_batch_size)
        self.final_next_batch_size = self.next_batch_size

        for key in current.keys():
            self.memory[key] = {'list_var': xp.zeros((self.final_next_batch_size, *current[key]['var'].shape))}

            if self.save_forward_map_evals and 'forward_map_evals' in current[key].keys():
                for k, v in current[key]['forward_map_evals'].items():
                    if xp.all(["grad" not in k, "hess_diag" not in k]):
                        self.memory[f"list_{k}"] = xp.zeros(
                            (self.final_next_batch_size,) + v.shape
                        )

            for k, v in nlpdf_utils[key].items():
                if xp.all(["nll_" not in k, "grad" not in k, "hess_diag" not in k]):
                    self.memory[key][f"list_{k}"] = xp.zeros(
                        (self.final_next_batch_size,) + v.shape
                    )

            for k, v in dict_objective[key].items():
                self.memory[key][f"list_{k}"] = xp.zeros((self.final_next_batch_size,) + v.shape)

            for k, v in additional_sampling_log[key].items():
                if isinstance(v, xp.ndarray):
                    self.memory[key][f"list_{k}"] = xp.zeros(
                        (self.final_next_batch_size,) + v.shape
                    )
                else:
                    self.memory[key][f"list_{k}"] = xp.zeros((self.final_next_batch_size,))

        self.memory["list_rng_state"] = xp.zeros(
            (self.final_next_batch_size, 32),
            dtype=xp.uint8,
        )
        self.memory["list_rng_inc"] = xp.zeros(
            (self.final_next_batch_size, 32),
            dtype=xp.uint8,
        )

    def update_memory(
        self,
        t: int,
        current: dict,
        nlpdf_utils: dict = dict(),
        dict_objective: dict = dict(),
        additional_sampling_log: dict = dict(),
        rng_state_array: Optional[xp.ndarray] = None,
        rng_inc_array: Optional[xp.ndarray] = None,
    ) -> None:

        """updates the memory with new information. All of the potential entries are optional except for the current iterate."""
        t_save = (t - self.t_last_init) // self.freq_save

        for key in current.keys():
            self.memory[key]['list_var'][t_save] = self.scaler.from_scaled_to_lin(current[key]['var']) # FIXME: we assume here that every variables need to be scaled back to the linear space.

            if self.save_forward_map_evals and 'forward_map_evals' in current[key].keys():
                for k, v in current[key]['forward_map_evals'].items():
                    if xp.all(["grad" not in k, "hess_diag" not in k]):
                        self.memory[f"list_{k}"][t_save] = v

            for k, v in nlpdf_utils[key].items():
                if xp.all(["nll_" not in k, "grad" not in k, "hess_diag" not in k]):
                    self.memory[key][f"list_{k}"][t_save] = v

            for k, v in dict_objective[key].items():
                if k not in ["m_a", "s_a", "m_m", "s_m"]:
                    self.memory[key][f"list_{k}"][t_save] = v

            for k, v in additional_sampling_log[key].items():
                self.memory[key][f"list_{k}"][t_save] = v

            if (rng_state_array is not None) and (rng_inc_array is not None):
                self.memory["list_rng_state"][t_save] = rng_state_array
                self.memory["list_rng_inc"][t_save] = rng_inc_array