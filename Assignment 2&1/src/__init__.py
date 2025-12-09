"""
FILE: __init__.py
-----------------
This file is essentially a marker. It tells Python that this directory ('src') 
should be treated as a package. 

Without this file, imports between sibling files (like importing 'data_loader' 
into 'train_eval.py') can sometimes break depending on how you run the script.

I am leaving it mostly empty to keep the namespace clean, but I added a 
version variable just in case we need to track submissions.
"""

__version__ = "0.1.0"
__author__ = "Group Clarity"

# If we wanted to, we could expose functions here to make imports shorter,
# e.g., 'from src import run_baseline_A' instead of 'from src.model_baseline_A...'
# but for this assignment, I'll keep the imports explicit in the individual files.