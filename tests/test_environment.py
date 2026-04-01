import pytest


def test_core_dependencies_import():
    """
    Verifies that all core project dependencies can be imported successfully.
    This acts as an environment sanity check for the team.
    """
    try:
        import torch
        import torchvision
        import numpy as np
        import pandas as pd
        import matplotlib
        import sklearn
        import mediapipe as mp

        # If we get here, nothing crashed
        imports_successful = True
    except ImportError as e:
        print(f"Import failed: {e}")
        imports_successful = False

    assert (
        imports_successful
    ), "One or more core dependencies failed to import. Check your environment setup."
