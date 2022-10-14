from abc import ABC, abstractmethod


class AbstractLBMMagnetic(ABC):
    type_flags = ["x", "X", "y", "Y", "z", "Z"]

    @property
    @abstractmethod
    def rank(self) -> int:
        ...

    @abstractmethod
    def get_H_int(self):
        ...
