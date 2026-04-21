"""
Training entrypoint for DIGIR Joint Multi-Agent Interacting Queries variant.

Reuses train_digir_full.py end-to-end and only swaps model import:
    from models.digir_joint_game import DIGIR
"""
import os
import sys

import train_digir_full as base


def import_digir_model_joint_game(digir_root: str):
    if digir_root:
        digir_root = os.path.abspath(os.path.expanduser(digir_root))
        if digir_root not in sys.path:
            sys.path.insert(0, digir_root)
    try:
        from models.digir_joint_game import DIGIR
    except Exception as exc:
        hint = (
            "Cannot import `models.digir_joint_game`. "
            "Set --digir_root (or DIGIR_ROOT env) to your DIGIR code directory."
        )
        raise RuntimeError(hint) from exc
    return DIGIR


base.import_digir_model = import_digir_model_joint_game


if __name__ == "__main__":
    base.main()

