import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import skanalytics
from skanalytics.engineering.preprocessing import (
    nanscale, StandardNANScaler)
