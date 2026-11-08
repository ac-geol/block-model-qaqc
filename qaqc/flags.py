"""Block flags (spec 8).

A flag is a yes/no marker on each block, for example "this block has the same centre as
another block". :class:`FlagRegistry` stores them behind a small API so the storage can change
later. Each flag is a numpy bool array, created only when the flag is first set on at least one
block, so flags that never fire cost nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FlagInfo:
    """What a flag means and which module sets it."""

    name: str
    description: str
    module: str


class FlagRegistry:
    """Yes/no markers on the rows of one table (block model or composites)."""

    def __init__(self, n_rows: int) -> None:
        self.n_rows = int(n_rows)
        self._info: dict[str, FlagInfo] = {}
        self._masks: dict[str, np.ndarray] = {}

    def register(self, name: str, description: str, module: str) -> None:
        """Declare a flag before it is set. Registering the same flag again is allowed."""
        self._info.setdefault(name, FlagInfo(name, description, module))

    def _check(self, name: str) -> None:
        if name not in self._info:
            raise KeyError(f"Flag '{name}' has not been registered")

    def set(self, name: str, mask: np.ndarray) -> None:
        """Turn the flag on for the rows where ``mask`` is True (added to any earlier marks)."""
        self._check(name)
        m = np.asarray(mask, dtype=bool)
        if m.shape != (self.n_rows,):
            raise ValueError(f"Flag '{name}': mask has {m.size} rows, expected {self.n_rows}")
        if name in self._masks:
            self._masks[name] |= m
        elif m.any():
            self._masks[name] = m.copy()

    def get(self, name: str) -> np.ndarray:
        """The flag as a bool array (all False if it was never set)."""
        self._check(name)
        mask = self._masks.get(name)
        return mask.copy() if mask is not None else np.zeros(self.n_rows, dtype=bool)

    def any(self) -> np.ndarray:
        """True for rows with at least one flag."""
        out = np.zeros(self.n_rows, dtype=bool)
        for mask in self._masks.values():
            out |= mask
        return out

    def counts(self) -> dict[str, int]:
        """Number of flagged rows for every registered flag (0 if never set)."""
        return {
            name: int(self._masks[name].sum()) if name in self._masks else 0 for name in self._info
        }

    def info(self) -> dict[str, FlagInfo]:
        """Registered flags with their descriptions."""
        return dict(self._info)

    def to_frame(self, rows: np.ndarray, only_active: bool = True) -> pd.DataFrame:
        """0/1 (uint8) columns, one per flag, for the given row positions.

        Args:
            rows: Row positions (integers) or a bool mask over all rows.
            only_active: Leave out flags that are not set on any row of the whole table.
        """
        rows = np.asarray(rows)
        if rows.dtype == bool:
            rows = np.flatnonzero(rows)
        names = [n for n in self._info if not only_active or n in self._masks]
        data = {n: self.get(n)[rows].astype(np.uint8) for n in names}
        return pd.DataFrame(data, index=pd.RangeIndex(len(rows)))
