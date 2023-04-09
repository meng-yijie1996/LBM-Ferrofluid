from abc import ABC, abstractmethod


class AbstractLBMPropagation(ABC):
    type_flags = ["x", "X", "y", "Y", "z", "Z"]

    @property
    @abstractmethod
    def rank(self) -> int:
        ...

    @abstractmethod
    def propagation(self, f):
        ...

    @abstractmethod
    def rebounce_obstacle(self, f, flags):
        ...
