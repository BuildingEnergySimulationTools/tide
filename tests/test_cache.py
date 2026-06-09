import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tide.cache import CacheManager
from tide.plumbing import Plumber


@pytest.fixture
def time_index():
    return pd.date_range("2023-01-01", freq="h", periods=24, tz="UTC")


@pytest.fixture
def sample_data(time_index):
    return pd.DataFrame(
        {
            "temp__°C__zone1": np.random.randn(24) * 5 + 20,
            "humid__%HR__zone1": np.random.randn(24) * 5 + 50,
        },
        index=time_index,
    )


@pytest.fixture
def pipe_dict():
    return {
        "pre": {
            "°C": [["ReplaceThreshold", {"upper": 30}]],
            "%HR": [["ReplaceThreshold", {"upper": 100}]],
        },
        "common": [["Interpolate", ["linear"]]],
    }


# ---------------------------------------------------------------------------
# CacheManager unit tests
# ---------------------------------------------------------------------------


def test_cache_manager_memory_hit(sample_data):
    cm = CacheManager(persistence=False)
    cm.set("key1", sample_data)
    result = cm.get("key1")
    pd.testing.assert_frame_equal(result, sample_data)


def test_cache_manager_miss(sample_data):
    cm = CacheManager(persistence=False)
    assert cm.get("nonexistent") is None


def test_cache_manager_clear_memory(sample_data):
    cm = CacheManager(persistence=False)
    cm.set("key1", sample_data)
    cm.clear(disk=False)
    assert cm.get("key1") is None


def test_cache_manager_persistence(sample_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        cm = CacheManager(persistence=True, cache_dir=tmpdir)
        cm.set("mykey", sample_data)

        # New CacheManager instance pointing to same dir (simulates new session)
        cm2 = CacheManager(persistence=True, cache_dir=tmpdir)
        result = cm2.get("mykey")
        pd.testing.assert_frame_equal(result, sample_data)


def test_cache_manager_persistence_clear_disk(sample_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        cm = CacheManager(persistence=True, cache_dir=cache_dir)
        cm.set("k", sample_data)
        assert (cache_dir / "k.parquet").exists()
        cm.clear(disk=True)
        assert not cache_dir.exists()


def test_cache_manager_hash_dataframe_deterministic(sample_data):
    h1 = CacheManager.hash_dataframe(sample_data)
    h2 = CacheManager.hash_dataframe(sample_data)
    assert h1 == h2


def test_cache_manager_hash_dataframe_changes_with_data(sample_data):
    h1 = CacheManager.hash_dataframe(sample_data)
    modified = sample_data.copy()
    modified.iloc[0, 0] += 999.0
    h2 = CacheManager.hash_dataframe(modified)
    assert h1 != h2


def test_cache_manager_make_key_deterministic():
    k1 = CacheManager.make_key("abc", {"step1": [["Identity"]]})
    k2 = CacheManager.make_key("abc", {"step1": [["Identity"]]})
    assert k1 == k2


def test_cache_manager_make_key_differs_on_steps():
    k1 = CacheManager.make_key("abc", {"step1": [["Identity"]]})
    k2 = CacheManager.make_key("abc", {"step2": [["Identity"]]})
    assert k1 != k2


# ---------------------------------------------------------------------------
# Plumber JSON pipe_dict
# ---------------------------------------------------------------------------


def test_save_and_load_pipe_dict(sample_data, pipe_dict):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "pipe.json"
        plumber = Plumber(sample_data, pipe_dict)
        plumber.save_pipe_dict(path)

        plumber2 = Plumber(sample_data)
        plumber2.load_pipe_dict(path)
        assert plumber2.pipe_dict == pipe_dict


def test_load_pipe_dict_returns_self(sample_data, pipe_dict):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "pipe.json"
        plumber = Plumber(sample_data, pipe_dict)
        plumber.save_pipe_dict(path)

        plumber2 = Plumber(sample_data)
        result = plumber2.load_pipe_dict(path)
        assert result is plumber2


def test_from_json_classmethod(sample_data, pipe_dict):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "pipe.json"
        with open(path, "w") as f:
            json.dump(pipe_dict, f)

        plumber = Plumber.from_json(path, data=sample_data)
        assert plumber.pipe_dict == pipe_dict
        pd.testing.assert_frame_equal(plumber.data, Plumber(sample_data).data)


def test_save_pipe_dict_is_valid_json(sample_data, pipe_dict):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "pipe.json"
        Plumber(sample_data, pipe_dict).save_pipe_dict(path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == pipe_dict


# ---------------------------------------------------------------------------
# Plumber cache integration
# ---------------------------------------------------------------------------


def test_get_corrected_data_result_unchanged(sample_data, pipe_dict):
    """Cached execution must produce the same result as the original pipeline."""
    plumber_ref = Plumber(sample_data, pipe_dict)
    expected = plumber_ref.get_pipeline().fit_transform(sample_data.copy())

    plumber_cached = Plumber(sample_data, pipe_dict)
    result = plumber_cached.get_corrected_data()

    pd.testing.assert_frame_equal(result, expected)


def test_cache_hit_on_second_call(sample_data, pipe_dict):
    """Second identical call should be served from cache (no recompute)."""
    plumber = Plumber(sample_data, pipe_dict)
    first = plumber.get_corrected_data()
    second = plumber.get_corrected_data()
    pd.testing.assert_frame_equal(first, second)
    # Cache should contain entries for both steps
    assert len(plumber.cache_manager._memory) == 2


def test_cache_prefix_reuse(sample_data, pipe_dict):
    """After running all steps, adding a new final step should reuse the prefix."""
    plumber = Plumber(sample_data, pipe_dict)
    plumber.get_corrected_data()  # warms cache for "pre" and "common"

    extended_pipe = dict(pipe_dict)
    extended_pipe["extra"] = [["Identity"]]
    plumber.pipe_dict = extended_pipe

    # Cache for "pre" and "common" prefix should still be valid
    data = sample_data.copy()
    data_hash = CacheManager.hash_dataframe(
        data.loc[data.index[0] : data.index[-1], data.columns].copy()
    )
    prefix_two = dict(list(pipe_dict.items())[:2])
    key = CacheManager.make_key(data_hash, prefix_two)
    assert plumber.cache_manager.get(key) is not None


def test_cache_partial_steps(sample_data, pipe_dict):
    """steps=['common'] cached independently from the full pipeline."""
    plumber = Plumber(sample_data, pipe_dict)
    result_partial = plumber.get_corrected_data(steps="common")
    result_partial2 = plumber.get_corrected_data(steps="common")
    pd.testing.assert_frame_equal(result_partial, result_partial2)


def test_cache_persistence_across_instances(sample_data, pipe_dict):
    """Results cached to disk must be loaded by a new Plumber instance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p1 = Plumber(sample_data, pipe_dict, cache_persistence=True, cache_dir=tmpdir)
        first = p1.get_corrected_data()

        p2 = Plumber(sample_data, pipe_dict, cache_persistence=True, cache_dir=tmpdir)
        second = p2.get_corrected_data()

        pd.testing.assert_frame_equal(first, second)
        # p2 should have loaded from disk (memory was empty before get_corrected_data)
        assert len(p2.cache_manager._memory) > 0


def test_clear_cache(sample_data, pipe_dict):
    plumber = Plumber(sample_data, pipe_dict)
    plumber.get_corrected_data()
    assert len(plumber.cache_manager._memory) > 0
    plumber.clear_cache()
    assert len(plumber.cache_manager._memory) == 0


def test_clear_cache_disk(sample_data, pipe_dict):
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        plumber = Plumber(
            sample_data, pipe_dict, cache_persistence=True, cache_dir=cache_dir
        )
        plumber.get_corrected_data()
        assert cache_dir.exists()
        plumber.clear_cache(disk=True)
        assert not cache_dir.exists()
