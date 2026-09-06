"""Leakage-controlled Bitcoin forecasting research package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lstm-bitcoin-research")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0+unknown"

__all__ = ["__version__"]
