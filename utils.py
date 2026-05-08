"""Utility functions."""

from pathlib import Path


def get_project_root():
    """Return the project root directory (the directory containing this file)."""
    return Path(__file__).resolve().parent
