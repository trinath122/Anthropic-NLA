"""Experiment pipelines, one module per plan step."""

import os

# Must be set before torch initialises CUDA; reduces fragmentation OOMs on long-sequence batches.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
