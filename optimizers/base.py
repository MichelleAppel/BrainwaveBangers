from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
    """Common interface the main loop expects."""
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def start_epoch(self, current_distance: float):
        """Called once per epoch with the current distance."""
        ...

    @abstractmethod
    def propose(self, phase: int):
        """
        Return next params dict for the given phase.
        phase: 1 => '+'; 2 => '-' (SPSA uses both; BO uses only phase 1)
        Return None to skip a phase.
        """
        ...

    @abstractmethod
    def report(self, phase: int, A_list, V_list):
        """Called at the end of a phase with the collected A/V samples."""
        ...

    @abstractmethod
    def step(self):
        """Called at the end of the epoch to update internal state."""
        ...

    @abstractmethod
    def current_params_dict(self):
        """Return current center parameters as a dict."""
        ...
