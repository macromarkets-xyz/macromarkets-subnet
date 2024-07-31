from pathlib import Path
from typing import Type, Optional, Any, List, Tuple
import math


# ---------------------------------
# Project Constants.
# ---------------------------------


# The uid for this subnet.
SUBNET_UID = 70
# The start block of this subnet
SUBNET_START_BLOCK = 2635801
# The root directory of this project.
ROOT_DIR = Path(__file__).parent.parent

# ---------------------------------
# Miner/Validator Model parameters.
# ---------------------------------

weights_version_key = 1

# validator weight moving average term
alpha = 0.9
# validator scoring exponential temperature
temperature = 0.08
