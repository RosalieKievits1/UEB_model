import os
"""Defaults"""

input_dir = '/Users/rosaliekievits/Desktop/Tiff bestanden MEP'
input_dir_knmi = '/Users/rosaliekievits/Desktop/SVFbestandenMEP'

# get config out of environment
input_dir = os.environ.get('UEB_MODEL_INPUT_DIR', input_dir)
input_dir_knmi = os.environ.get('UEB_MODEL_INPUT_DIR_knmi', input_dir_knmi)
