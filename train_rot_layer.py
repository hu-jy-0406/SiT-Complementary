#!/usr/bin/env python3
"""Run the shared SiT trainer with the rotation-layer model implementation."""

import os
import runpy


# train.py imports its model registry at module load time. Set both variables
# first, and require the shared trainer to fail closed if another registry is
# ever selected accidentally.
os.environ["SIT_MODEL_MODULE"] = "models_rot_layer"
os.environ["SIT_EXPECTED_MODEL_MODULE"] = "models_rot_layer"

runpy.run_module("train", run_name="__main__")
