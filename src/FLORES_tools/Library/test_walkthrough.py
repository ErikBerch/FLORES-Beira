from __future__ import (absolute_import, division, print_function, unicode_literals)
import sys
from pathlib import Path
import os
import matplotlib.pyplot as plt
import scipy.stats as ss
import numpy as np
import csv
from scipy.integrate import trapz
from timeit import default_timer as timer

path = Path().absolute()
print(path)
#!conda install --yes --prefix C:\\Program Files\\Anaconda3 geopandas
sys.executable

#Only use to change directory to parent directory
os.chdir(path.parent)
#Only use to change directory back to file
print(os.getcwd())

from  .Library.simulation_calculations_beira import run_hydraulic_calculation
from FLORES_tools.Library.simulation_definitions_beira import (Impact)
import FLORES_tools.Library.simulation_data_beira as data
from FLORES_tools.Library.flood_simulation_model import FloodSimModel

from ema_workbench import (RealParameter, ScalarOutcome, Constant, Model)

import rasterio
from matplotlib import pyplot
from IPython.display import Image

dir_name_data = Path().cwd().joinpath('Projects\FLORES_beira\input_data')
print(dir_name_data)
flores_sim = FloodSimModel()
