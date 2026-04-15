def test_core_dependencies_import():
    """
    Verifies that all core project dependencies can be imported successfully.
    This acts as an environment sanity check for the team.
    """
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        import numpy as np  # noqa: F401
        import pandas as pd  # noqa: F401
        import matplotlib  # noqa: F401
        import sklearn  # noqa: F401
        import mediapipe as mp  # noqa: F401

        # If we get here, nothing crashed
        imports_successful = True
    except ImportError as e:
        print(f"Import failed: {e}")
        imports_successful = False

    assert (
        imports_successful
    ), "One or more core dependencies failed to import. Check your environment setup."
