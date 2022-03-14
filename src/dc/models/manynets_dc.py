# -*- coding: utf-8 -*-
"""DC specific implementation of Manynets.
DEPRECATED; merged w/ compression version. Only here for initial pre-trained model compatibility"""

import torch
from torch import nn
import math
import random
import numpy as np
import time
import statistics
import logging
import sys

sys.path.append("..")
from compression.models import abstract_model
from compression.models import Balle2018PT_compressor
from compression.models import bitEstimator
from common.libs import pt_helpers
from common.libs import pt_ops
from common.libs import distinct_colors
from common.libs import utilities
from compression.models import manynets_compressor

logger = logging.getLogger("ImageDC")
try:
    import torchac
except ModuleNotFoundError:
    logger.info("manynets_dc: torchac not available; entropy coding is disabled")
try:
    import png
except ModuleNotFoundError:
    logger.info("manynets_dc: png is not available (currently used in encode/decode)")


class ManyPriors_DC(manynets_compressor.Balle2017ManyPriors_ImageCompressor):
    def __init__(self, **kwargs):
        super().__init__(lossf=None, **kwargs)
