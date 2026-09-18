"""Exercise native argument parsing without constructing a model or dataset."""
import sys
from types import ModuleType
from starVLA.rl.flow_grpo.cli import main
import pytest


@pytest.mark.parametrize('arguments', [
    ['--set', 'runtime.activation_checkpointing=false', '--set', 'runtime.max_updates=2'],
    ['--set', 'runtime.activation_checkpointing=false', 'runtime.max_updates=2'],
])
def test_all_override_groups_reach_native_resolver(monkeypatch, arguments):
    seen = {}
    config = ModuleType('starVLA.rl.flow_grpo.config')
    def resolve(*args):
        seen['overrides'] = args[-1]
        return {'sentinel': 123}, None
    config.resolve_config = resolve
    config.split_tokens = lambda cfg: ([], [])
    trainer = ModuleType('starVLA.rl.flow_grpo.trainer')
    trainer.run = lambda cfg, sft, resume: seen.update(executed=cfg)
    monkeypatch.setitem(sys.modules, config.__name__, config)
    monkeypatch.setitem(sys.modules, trainer.__name__, trainer)
    monkeypatch.delenv('REWARD_WORKERS', raising=False)
    monkeypatch.setattr(sys, 'argv', ['flow-grpo', 'train', *arguments])
    main()
    assert seen == {'overrides': ['runtime.activation_checkpointing=false', 'runtime.max_updates=2'],
                    'executed': {'sentinel': 123}}
