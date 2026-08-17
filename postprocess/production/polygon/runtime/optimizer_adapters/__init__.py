"""Explicit compatibility adapters around the frozen numerical kernel."""

from .artifacts import install_artifact_adapters
from .geometry import install_geometry_adapters
from .native_dp import install_native_dp_adapters
from .python_dp import install_python_dp_adapter
from .resources import install_resource_adapters

__all__ = (
    "install_artifact_adapters",
    "install_geometry_adapters",
    "install_native_dp_adapters",
    "install_python_dp_adapter",
    "install_resource_adapters",
)
