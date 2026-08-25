"""LoRA-adapted ESMC-300M helpers.

Train adapters with ``python -m src.representations.train_lora_esmc``.
Export shards with ``python -m src.representations.export_lora_esmc``.
The adapter checkpoints are not shipped.
"""

from __future__ import annotations

DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ("out_proj", "layernorm_qkv.1")
