import pytest
from pathlib import Path

from clt_forge.clt import CLT
from clt_forge.config import CLTConfig
from tests.utils import build_clt_training_runner_cfg, build_autointerp_cfg
from clt_forge.autointerp.pipeline_new import AutoInterp

# Minimal testing, just looking at the auto-interp outputs is a strong test in itself.

@pytest.fixture
def tiny_saved_clt(tmp_path: Path):
    """
    Create and save a tiny CLT so AutoInterp can load it.
    """
    cfg_runner = build_clt_training_runner_cfg(
        d_in=64,
        d_latent=4*64,
        context_size=16,
        train_batch_size_tokens=4096,
    )

    clt_cfg = cfg_runner.create_sub_config(CLTConfig, n_layers=1)
    clt = CLT(clt_cfg)

    save_dir = tmp_path / "clt_ckpt"
    save_dir.mkdir()

    clt.save_model(str(save_dir))

    return save_dir

def test_autointerp_runs(tmp_path, tiny_saved_clt):
    cfg = build_autointerp_cfg(
        base_dir=tmp_path,
        clt_path=tiny_saved_clt,
    )

    autointerp = AutoInterp(cfg)

    autointerp.run(
        job_id=0,
        total_jobs=4,
        save_dir=Path(cfg.latent_cache_path),
        generate_explanations=False,
    )

    db_path = Path(cfg.latent_cache_path) / "parquet" / "job_0.parquet"
    assert db_path.exists()
