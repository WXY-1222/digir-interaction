"""
Wrapper training entrypoint for local sparse map-attention DIGIR.

This script reuses `train_digir_full.py` end-to-end and only swaps model import to:
    from models.digir_local_map import DIGIR

All CLI arguments and training/eval logic stay identical to train_digir_full.py.
"""
import os
import sys

import train_digir_full as base


def import_digir_model_local_map(digir_root: str):
    if digir_root:
        digir_root = os.path.abspath(os.path.expanduser(digir_root))
        if digir_root not in sys.path:
            sys.path.insert(0, digir_root)
    try:
        from models.digir_local_map import DIGIR
    except Exception as exc:
        hint = (
            "Cannot import `models.digir_local_map`. "
            "Set --digir_root (or DIGIR_ROOT env) to your DIGIR code directory."
        )
        raise RuntimeError(hint) from exc
    return DIGIR


base.import_digir_model = import_digir_model_local_map


if __name__ == "__main__":
    base.main()

