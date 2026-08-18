#!/usr/bin/env python3
"""Run the shared SiT trainer with the convolutional-layer model implementation."""

import os
import runpy


# Select models_conv.py before train.py imports the model registry, and require
# the shared trainer to fail closed if another model module is selected.
os.environ["SIT_MODEL_MODULE"] = "models_conv"
os.environ["SIT_EXPECTED_MODEL_MODULE"] = "models_conv"

runpy.run_module("train", run_name="__main__")
