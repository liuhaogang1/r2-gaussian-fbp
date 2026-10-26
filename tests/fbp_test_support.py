"""Test-only TIGRE import stand-in; production dependency loading is unchanged."""
import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest import mock


def import_prepare():
    tigre = ModuleType("tigre")
    tigre.geometry = lambda **kwargs: SimpleNamespace(**kwargs)
    algorithms = ModuleType("tigre.algorithms")
    tigre.algorithms = algorithms
    with mock.patch.dict(sys.modules, {"tigre": tigre, "tigre.algorithms": algorithms}):
        return importlib.import_module("prepare_fbp_tiff")
