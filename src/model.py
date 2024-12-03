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
        self.descriptions = descriptions # assert ndim == 1

        # ? in case of multi-modal approach add the additional details
        # since the class object has to be initialized with the hs_codes
        # we can take privileage of the same and add dummy hs code which is
        # the first value from the ``self.hs_codes`` thus, the score can be
        # always triggered with hs code in the self.hs_code attribute
        _n_rows, _dummy_hs_code = self.descriptions.shape[0], self.hs_codes[0]
        hs_codes = kwargs.get("hs_codes", np.array([_dummy_hs_code] * _n_rows))

        # ? the manufacturer are added and set as class attribute as
        # the manufacturer are further required for finding/scoring the secondary cluster
        self.manufacturers = kwargs.get("manufacturers", np.array([]))

        # todo: frame the logic in such a way that the fit function works
        # when data records are passed w/o hs_codes or when with the data
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
