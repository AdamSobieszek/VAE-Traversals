"""Exercise shell orchestration with lightweight stand-ins for expensive training."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
VP = ROOT.parent / 'VP-metric-pytorch'
pytestmark = pytest.mark.skipif(not VP.is_dir(), reason='VP sister project is unavailable')


@pytest.mark.parametrize('model,runner', [('SNGAN', 'sngan'), ('StyleGAN', 'stylegan2'),
                                         ('GAT', 'gat'), ('BigGAN', 'biggan'), ('ProgGAN', 'proggan')])
def test_generation_then_vp(tmp_path, model, runner):
    experiment = tmp_path / 'experiment with spaces'
    experiment.mkdir()
    stub = tmp_path / 'python-stub'
    stub.write_text(f'#!{sys.executable}\n' + '''import json, os, runpy, sys
from pathlib import Path
import numpy as np
if sys.argv[1] == '-c':
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
with open(os.environ['PIPELINE_LOG'], 'a') as stream:
    stream.write(json.dumps(sys.argv[1:]) + '\\n')
if sys.argv[1] == 'lib/val_utils.py':
    if os.environ.get('FAIL_GENERATION'):
        sys.exit(3)
    output = Path(sys.argv[sys.argv.index('--exp') + 1]) / 'vp_pairs'
    output.mkdir()
    np.save(output / 'labels.npy', np.eye(7))
else:
    # Validate forwarded options against the actual VP CLI without training.
    parser = runpy.run_path(sys.argv[1])['build_parser']()
    args = parser.parse_args(sys.argv[2:])
    assert args.out_dim == 7
    assert Path(args.data_dir, 'labels.npy').is_file()
''')
    stub.chmod(0o755)
    log = tmp_path / 'calls.jsonl'
    env = {**os.environ, 'PAIR_PYTHON': str(stub), 'VP_PYTHON': str(stub),
           'VP_ROOT': str(VP), 'PIPELINE_LOG': str(log)}
    for key in ('N_SAMPLES', 'SEED', 'BIDIRECTIONAL', 'VP_EPOCHS', 'FAIL_GENERATION'):
        env.pop(key, None)
    subprocess.run(['bash', str(ROOT / f'scripts/{model}_genpairs.sh'), str(experiment)],
                   cwd=tmp_path, env=env, check=True)
    generation, metric = [json.loads(line) for line in log.read_text().splitlines()]
    assert generation[0] == 'lib/val_utils.py'
    assert generation[generation.index('--n-samples') + 1] == '20000'
    assert generation[generation.index('--seed') + 1] == '123'
    assert '--bidirectional' in generation
    assert metric[metric.index('--data-dir') + 1] == str(experiment / 'vp_pairs')
    assert metric[metric.index('--result-dir') + 1] == str(experiment / 'vp_results')
    assert metric[-6:] == ['--out-dim', '7', '--seed', '123', '--epochs', '300']
    assert metric[metric.index('--run-name') + 1].lower() == runner
    log.unlink()
    failed = subprocess.run(['bash', str(ROOT / f'scripts/{model}_genpairs.sh'), str(experiment)],
                            cwd=tmp_path, env={**env, 'FAIL_GENERATION': '1'})
    assert failed.returncode == 3
    assert len(log.read_text().splitlines()) == 1  # never run VP after failed generation
