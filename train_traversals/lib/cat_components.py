"""Import the sibling CAT's shared layers without shadowing Traversals' models.

CAT is part of this repository, but its scripts import an unrelated top-level
``models`` namespace. Load only the shared representation module as a private
package; its relative imports reuse CAT's attention, RoPE and SwiGLU verbatim.
"""
import importlib.util
from pathlib import Path
import sys


_NAME = "_traversals_cat_transformer"
if _NAME not in sys.modules:
    _root = Path(__file__).resolve().parents[2] / "CAT" / "models"
    _spec = importlib.util.spec_from_file_location(
        _NAME, _root / "transformer.py", submodule_search_locations=[str(_root)]
    )
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_NAME] = _module
    try:
        _spec.loader.exec_module(_module)
    except Exception:
        del sys.modules[_NAME]
        raise

TransformerBlock = sys.modules[_NAME].TransformerBlock
VisionRotaryEmbeddingFast = sys.modules[_NAME].VisionRotaryEmbeddingFast
get_2d_sincos_pos_embed = sys.modules[_NAME].get_2d_sincos_pos_embed
