"""Regression tests for the Matrix-Game-3 Ulysses attention adapter."""

import importlib.util
import sys
import types
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]
ULYSSES_PATH = ROOT / "wan" / "distributed" / "ulysses.py"


def _load_ulysses_with_attention_stub(attention):
    """Load the adapter with lightweight relative-import stubs."""

    package_names = (
        "matrix_game_test",
        "matrix_game_test.wan",
        "matrix_game_test.wan.distributed",
        "matrix_game_test.wan.modules",
    )
    for name in package_names:
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package

    attention_module = types.ModuleType("matrix_game_test.wan.modules.attention")
    attention_module.flash_attention = lambda *args, **kwargs: None
    attention_module.attention = attention
    sys.modules[attention_module.__name__] = attention_module

    util_module = types.ModuleType("matrix_game_test.wan.distributed.util")
    util_module.all_to_all = lambda value, **kwargs: value
    sys.modules[util_module.__name__] = util_module

    spec = importlib.util.spec_from_file_location(
        "matrix_game_test.wan.distributed.ulysses", ULYSSES_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_distributed_attention_forwards_version_keyword():
    calls = []

    def attention(q, k, v, *, k_lens, window_size, version):
        calls.append(
            {
                "k_lens": k_lens,
                "window_size": window_size,
                "version": version,
            }
        )
        return q

    module = _load_ulysses_with_attention_stub(attention)

    class SingleProcessDist:
        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def get_world_size():
            return 1

    module.dist = SingleProcessDist
    q = torch.zeros(1, 2, 4, 3)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    seq_lens = torch.tensor([2])

    output = module.distributed_attention(
        q, k, v, seq_lens, window_size=(3, 5), fa_version="3"
    )

    assert output is q
    assert calls == [
        {"k_lens": seq_lens, "window_size": (3, 5), "version": "3"}
    ]
