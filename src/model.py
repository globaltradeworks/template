# -*- encoding: utf-8 -*-

"""
Alternate Base Class Defination for Identification of the Material

The material (primary cluster) and its smallest possible cluster which
is going to be identified is based on the base model defined in the
:mod:``dataway`` module which is available freely to the public.
However, each specific module inherits the same core base model and
extends it to its own needs.
"""

import math

import numpy as np
import pandas as pd

from typing import Iterable, List

import TradeETL as etl # noqa: F401, F403 # pyright: ignore[reportMissingImports]

class ModelBaseName(etl.models.BaseModel):
    def __init__(self, material : str, hs_codes : Iterable[str], grades : Iterable[dict] = []) -> None:
        super().__init__(material, hs_codes, grades)


    def fit(self, descriptions : np.ndarray, *args, **kwargs) -> None:
        self.descriptions = self._normalize(descriptions)

        # ? in case of multi-modal approach add the additional details
        hs_codes = kwargs.get("hs_codes", np.array([]))

        # ? the manufacturer are added and set as class attribute as
        # the manufacturer are further required for finding/scoring the secondary cluster
        self.manufacturers = kwargs.get("manufacturers", np.array([]))

        # todo: after processing and fitting the model, set scores
        # scores attribute is available as variable once the model fits
        self.scores = None

        return None


    def predict(self, thresh : float, *args, **kwargs) -> Iterable[List]:
        bins = [0, 60, math.floor((thresh - 0.2 * thresh)), math.floor((thresh - 0.1 * thresh)), thresh, 100]

        # todo: also create additional control metrics and set as class property
        # like the fuzzy score remarks which can be captured by the module mixer
        self.fuzzy_remarks = pd.cut(self.scores, bins = bins, labels = ["R", "L", "M", "N", "H"])

        # typically there is a true-value and the false-value is always none
        # the true value represents the primary and secondary cluster (if exists) unique code
        trueval, falseval = ("code", None), (None, None)

        return np.array([ trueval if score > thresh else falseval for score in self.scores ])
