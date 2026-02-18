# src/__init__.py
# ---------------------------------------------------------------
# Package root for researchOS shared modules.
#
# Migration note:
#   This package currently lives at  src/
#   and is imported as               from src.config import ...
#   When we move to a proper package (pip install -e .),
#   rename this directory to         researchos/
#   and update imports to            from researchos.config import ...
#   All internal imports use relative form (from . / from .notion)
#   so only the top-level consumer imports need updating.
# ---------------------------------------------------------------
