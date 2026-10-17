"""Regenerate tests/data/engine_golden.npz from the current engine.

Only run this when an intentional change to the engine's output is made;
the golden file is what proves refactors don't change the picture.

    python -m tools.make_engine_golden
"""

from __future__ import annotations

import os

import numpy as np

from core.engine import OrnamentEngine, load_tile
from tests.engine_cases import CASES, GOLDEN, run_case


def main() -> None:
    out = {}
    for name in CASES:
        for key, value in run_case(name, OrnamentEngine, load_tile).items():
            out[f"{name}_{key}"] = value
    os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
    np.savez_compressed(GOLDEN, **out)
    print(f"Wrote {GOLDEN}: " + ", ".join(f"{k}{list(v.shape)}" for k, v in out.items()))


if __name__ == "__main__":
    main()
