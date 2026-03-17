"""Launch script for the frontend application."""

import sys
from pathlib import Path

project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from clt_forge.frontend.app import main
from clt_forge.frontend.config.settings import AppConfig
STORAGE_ROOT = Path(clt_forge.__file__).resolve().parents[2] / "storage" # symlink to scratch

if __name__ == "__main__":
    MODEL = "gpt2"
    config = AppConfig(
        attr_graph_path=STORAGE_ROOT / "attribution" / MODEL / "attribution_graph.pt",
        dict_base_folder= STORAGE_ROOT / "autointerp" / MODEL / "dict",
        clt_checkpoint= STORAGE_ROOT / "checkpoints" / MODEL / "d1s3fw30/middle_22137856", # only needed for interventions
        model_name= MODEL, # only needed for interventions
        model_class_name="HookedTransformer",
        host="0.0.0.0",
        port=8157,
        debug=False,
    )

    main(config)
