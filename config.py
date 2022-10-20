import os
"""Defaults"""

input_dir = './'

# get config out of environment
input_dir = os.environ.get('UEB_MODEL_INPUT_DIR', input_dir)
