"""Pytest configuration: headless matplotlib and quiet logging."""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
logging.getLogger("ts_forecast").setLevel(logging.WARNING)
