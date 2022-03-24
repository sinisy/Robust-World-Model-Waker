# General tools.
from .config import *
from .counter import *
from .flags import *
from .logger import *
from .when import *
from .utils import *

# RL tools.
from .other import *
from .driver import *
from .replay import *

# TensorFlow tools.
from .tfutils import *
from .dists import *
from .nets import *


DOMAIN_TASK_IDS = {
    'terrainwalker_all': ['stand', 'walk', 'run', 'flip', 'walk-bwd'],
    'terrainhopper_all': ['hop', 'stand', 'hop-bwd'],
    'po