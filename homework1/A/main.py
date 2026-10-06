import sys
import pandas as pd
import numpy as np
from pathlib import Path
homework1_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(homework1_dir))

# Import const.py from the utils package
from utils import const

df = pd.read_excel(const.INPUT_FILE_PATH, sheet_name=None)
