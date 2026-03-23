#!/bin/bash

uv pip install "jax[cuda12]>=0.6.2,<0.7.2"

uv run python -c "import jax; print(jax.devices())"
