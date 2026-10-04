import abc
from dataclasses import dataclass
from enum import StrEnum


class VariableDirection(StrEnum):
    X = "x"
    Y = "y"


@dataclass
class Variable:
    name: str
    direction: VariableDirection
    type: type
    value: int | float | str | list


@dataclass
class System(abc.ABC):
    x: list[Variable]
    y: list[Variable]

    @abc.abstractmethod
    def f(self):
        pass

