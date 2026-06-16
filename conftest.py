import sys
from pathlib import Path

# Allow imports like `from pipeline.config import ...` used inside bst_x. Keeps tests in tests dir
sys.path.insert(0, str(Path(__file__).parent / "src" / "bst_x"))
# Allow imports like `from model.tempose import ...` used inside stroke_classification
sys.path.insert(0, str(Path(__file__).parent / "src" / "bst_x" / "stroke_classification"))
# Allow imports like `from shared.temporal import ...` for BRIC packages.
sys.path.insert(0, str(Path(__file__).parent / "src"))
