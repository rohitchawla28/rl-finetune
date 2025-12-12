# CELL: Generate All Plots
# Copy this cell into your notebook or run as a script

import sys
import os

# Make sure we're in the right directory
if not os.path.exists('generate_plots.py'):
    print("ERROR: generate_plots.py not found in current directory")
    print("Make sure you're in the rl-finetune directory")
else:
    # Execute the plotting script
    exec(open('generate_plots.py').read())

