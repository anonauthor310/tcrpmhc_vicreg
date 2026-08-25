"""LoRA-adapted ESMC-300M embedding export helpers.

The LoRA adapters themselves are trained separately and are not shipped.
See ``src/representations/export_lora_esmc.py``.
"""

from __future__ import annotations

DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
