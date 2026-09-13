"""Check resolved metadata and immutable launch records with small real objects."""
from argparse import Namespace
from pathlib import Path

import torch
import yaml

from lib.aux import save_experiment_config, snapshot_source
from lib.TraversalPDE import TraversalPDE
from lib.utils import build_adamw, CosineScheduleWithWarmup


def test_resolved_config_and_resume(tmp_path):
    net = TraversalPDE(2, 3, 4, n_hidden=8, lambdas={"BB": .25})
    recognizer = torch.nn.Linear(4, 2)
    optimizer = build_adamw(recognizer, lr=.003, weight_decay=.02, betas=(0., .99))
    scheduler = CosineScheduleWithWarmup(optimizer, 2, 20)
    kwargs = dict(models={"recognizer": recognizer}, traversals={"pde": net},
                  optimizers={"recognizer": optimizer}, schedulers={"recognizer": scheduler})
    path = save_experiment_config(tmp_path, Namespace(batch_size=4), **kwargs)
    original = path.read_bytes()
    cfg = yaml.safe_load(original)
    assert cfg["args"]["batch_size"] == 4
    groups = cfg["optimizers"]["recognizer"]["param_groups"]
    assert {g["weight_decay"] for g in groups} == {0., .02}
    assert all(g["betas"] == [0., .99] for g in groups)
    assert {n for g in groups for n in g["parameters"]} == {"recognizer.weight", "recognizer.bias"}
    assert cfg["schedulers"]["recognizer"]["state"]["min_lr_factor"] == .1
    pde = cfg["traversals"]["pde"]
    assert pde["modules"]["F"]["settings"]["n_hidden"] == 8
    assert pde["settings"]["_pde_cfg"]["eps_norm2"] > 0
    assert any(loss["lam"] == .25 for loss in pde["losses"])
    source = Path(__file__).resolve().parents[1] / "lib" / "TraversalPDE.py"
    assert (tmp_path / "lib" / source.name).read_bytes() == source.read_bytes()
    # Simulate an existing snapshot from a different source revision.
    (tmp_path / "lib" / source.name).write_text("original source")
    optimizer.param_groups[0]["weight_decay"] = .07
    second = save_experiment_config(tmp_path, Namespace(batch_size=8), **kwargs)
    assert second.parent.parent == tmp_path / "runs"
    assert path.read_bytes() == original
    assert (tmp_path / "lib" / source.name).read_text() == "original source"
    assert (second.parent / "lib" / source.name).read_bytes() == source.read_bytes()
    assert yaml.safe_load(second.read_text())["optimizers"]["recognizer"]["param_groups"][0]["weight_decay"] == .07
    assert not list(tmp_path.rglob("*.pyc"))


def test_snapshot_is_not_overwritten(tmp_path):
    snapshot_source(tmp_path)
    marker = tmp_path / "lib" / "recognizer.py"
    marker.write_text("old recognizer")
    snapshot_source(tmp_path)
    assert marker.read_text() == "old recognizer"
