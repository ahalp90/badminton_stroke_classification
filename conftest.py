import sys
from pathlib import Path

# Allow imports like `from pipeline.config import ...` used inside bst_refactor. Keeps tests in tests dir
sys.path.insert(0, str(Path(__file__).parent / "src" / "bst_refactor"))
# Allow imports like `from model.tempose import ...` used inside stroke_classification
sys.path.insert(0, str(Path(__file__).parent / "src" / "bst_refactor" / "stroke_classification"))
# Allow imports like `from perception.temporal import ...` for BRIC packages.
sys.path.insert(0, str(Path(__file__).parent / "src"))
