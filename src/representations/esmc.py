"""Load frozen ESMC-300M per-residue embedding shards.

Export first with ``python -m src.representations.esmc`` (see that module's
CLI in ``export_raw_esmc.py``). Shards are not redistributed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch


def load_shard(path: Path) -> List[Dict[str, Any]]:
    return torch.load(path, map_location="cpu", weights_only=False)
