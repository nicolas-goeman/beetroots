import abc


class SimulationTargetDistributionType(abc.ABC):
    @abc.abstractmethod
    def setup_target_distribution(self):
        pass

    @abc.abstractmethod
    def inversion_mcmc(self):
        pass

    @abc.abstractmethod
    def inversion_optim_map(self):
        pass

    @abc.abstractmethod
    def inversion_optim_mle(self):
        pass
