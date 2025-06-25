#!/bin/bash

# 1. first create new venv
# just do it

# 2. install torch, this is how we did it:
pip install torch==2.2.2+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html

# 3. install other dependencies using setup.py:
pip install -e .
