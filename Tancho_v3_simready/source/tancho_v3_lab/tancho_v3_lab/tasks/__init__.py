import os

from . import staged

# Legacy Flat/Rough/Fixed-Flat tasks (pre-staged experiments and the scripts in
# scripts/diagnostics, scripts/evaluation).  Opt in with TANCHO_LEGACY_TASKS=1.
if os.environ.get("TANCHO_LEGACY_TASKS") == "1":
    from .direct.tancho_v3 import register  # noqa: F401
