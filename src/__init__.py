# -*- encoding: utf-8 -*-

"""
The Source Directory for the AI-ML Model for Prediction of Material

The material (primary cluster) and its smallest possible cluster which
is typically the grade of the commodity material being exported. name
of the module has to be refactored and set accordingly for each
specific material classifical problem based on this template.

@author: Debmalya Pramanik
@copywright: 2024; Copywright Authors
"""

# ? package follows https://peps.python.org/pep-0440/
# ? https://python-semver.readthedocs.io/en/latest/advanced/convert-pypi-to-semver.html
__version__ = "v0.0.1.dev0"

# initialize the directory and include the settings file
# settings/configuration files are available module wide
import os

ROOT = os.path.abspath(os.path.dirname(__file__))
CONFIG = os.path.join(ROOT, "config")

# the model config is agnostic and is available as global variable
# to all end user, this reduces `os.path.join(...)` repeated syntax
import yaml

with open(os.path.join(CONFIG, "model.yaml"), "r") as f:
    MODEL_CONFIG = yaml.load(f, Loader = yaml.FullLoader)


# init-time options registrations
from src.model import ModelBaseName
