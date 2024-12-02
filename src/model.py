# -*- encoding: utf-8 -*-

"""
Alternate Base Class Defination for Identification of the Material

The material (primary cluster) and its smallest possible cluster which
is going to be identified is based on the base model defined in the
:mod:``dataway`` module which is available freely to the public.
However, each specific module inherits the same core base model and
extends it to its own needs.
"""

import numpy as np

from abc import abstractmethod
from typing import Iterable, List

import TradeETL as etl # noqa: F401, F403 # pyright: ignore[reportMissingImports]

class ModelBaseName(etl.models.BaseModel):
    def __init__(self, material : str, hsc_codes : Iterable[str], grades : Iterable[dict] = []) -> None:
        super().__init__(material, hsc_codes, grades)


    def fit(self, descriptions : np.ndarray) -> None:
        descriptions = [ self._normalize(description).upper() for description in descriptions ]


    def predict(self, thresh : float, *args, **kwargs) -> Iterable[List]:
        pass


    def modelname(self) -> str:
        return self.__class__.__name__
