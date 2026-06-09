import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


class CacheManager:
    """Step-level cache for Plumber pipeline results.

    Stores the output DataFrame after each pipeline step, keyed by a hash of the
    input data and the ordered steps configuration.  Two modes:

    - persistence=False (default): in-memory only, cleared when the Python session ends.
    - persistence=True: also writes Parquet files to ``cache_dir`` so results survive
      across sessions.  In-memory cache is always consulted first.

    Parameters
    ----------
    persistence : bool, default False
        Whether to persist cache entries to disk.
    cache_dir : str or Path, optional
        Directory for Parquet files.  Defaults to ``.tide_cache/`` in the current
        working directory.  Ignored when ``persistence=False``.
    """

    def __init__(self, persistence: bool = False, cache_dir: str | Path = None):
        self.persistence = persistence
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".tide_cache")
        self._memory: dict[str, pd.DataFrame] = {}

    def get(self, key: str) -> pd.DataFrame | None:
        """Return the cached DataFrame for *key*, or None on a miss."""
        if key in self._memory:
            return self._memory[key]
        if self.persistence:
            path = self.cache_dir / f"{key}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                # parquet does not preserve DatetimeIndex.freq — restore it
                try:
                    df.index.freq = pd.infer_freq(df.index)
                except Exception:
                    pass
                self._memory[key] = df
                return df
        return None

    def set(self, key: str, df: pd.DataFrame) -> None:
        """Store *df* under *key* (memory + optionally disk)."""
        self._memory[key] = df.copy()
        if self.persistence:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(self.cache_dir / f"{key}.parquet")

    def clear(self, disk: bool = True) -> None:
        """Clear the cache.

        Parameters
        ----------
        disk : bool, default True
            Also remove the on-disk cache directory when ``persistence=True``.
        """
        self._memory.clear()
        if disk and self.persistence and self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)

    @staticmethod
    def hash_dataframe(df: pd.DataFrame) -> str:
        """Return an MD5 hex digest of *df* (index + values)."""
        return hashlib.md5(
            pd.util.hash_pandas_object(df, index=True).values.tobytes()
        ).hexdigest()

    @staticmethod
    def make_key(data_hash: str, steps_prefix: dict) -> str:
        """Return a cache key for a given data hash and ordered steps prefix dict."""
        payload = json.dumps(
            {"d": data_hash, "s": steps_prefix}, sort_keys=True, default=str
        )
        return hashlib.md5(payload.encode()).hexdigest()
