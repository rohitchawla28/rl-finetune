# FIX_IMPORT_CELL.py
"""
Run this cell FIRST if you get import errors.
This forces Python to reload all modules.
"""

import sys
import importlib

# Force reload src modules
modules_to_reload = [
    'src',
    'src.eval_utils',
    'src.sft',
    'src.scst',
    'src.analysis',
    'src.rewards',
    'src.data',
]

for module_name in modules_to_reload:
    if module_name in sys.modules:
        print(f"Reloading {module_name}...")
        importlib.reload(sys.modules[module_name])
    else:
        print(f"Module {module_name} not yet loaded")

# Verify the import works
try:
    from src.eval_utils import get_dataset_keys
    test = get_dataset_keys("cnn_dailymail")
    print(f"✅ SUCCESS! get_dataset_keys works: {test}")
except Exception as e:
    print(f"❌ Still failing: {e}")
    print("\nTry:")
    print("1. Restart runtime/kernel")
    print("2. Verify src/eval_utils.py has get_dataset_keys function")
    print("3. Check you're in the correct directory")

