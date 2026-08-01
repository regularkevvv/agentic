"""Make the flat handler modules importable however pytest is invoked.

Hugging Face Inference Endpoints load handler.py from the repository root of
the deployed model, so the modules here are flat rather than a package. That
layout is what gets deployed, so it is what the tests import.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
