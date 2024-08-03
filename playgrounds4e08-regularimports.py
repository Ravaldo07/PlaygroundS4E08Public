# %% [code]

print(f"\n---> Importing commonly used libraries and packages in my model pipelines\n")
import subprocess
import sys

# Define the pip install command
command = [sys.executable, "-m", "pip", "install", "lightgbm==4.5.0"]
subprocess.check_call(command)
command = [sys.executable, "-m", "pip", "install", "polars==1.2.1"]
subprocess.check_call(command)
command = [sys.executable, "-m", "pip", "install", "ucimlrepo"]
subprocess.check_call(command)

from ucimlrepo import fetch_ucirepo 
from IPython.display import clear_output
from gc import collect

from warnings import filterwarnings
filterwarnings('ignore')

from os import path, walk, getpid
from psutil import Process
import re, shutil
from collections import Counter
from itertools import product

import ctypes
libc = ctypes.CDLL("libc.so.6")

from IPython.display import display_html, clear_output
from pprint import pprint
from functools import partial
from copy import deepcopy
import pandas as pd, numpy as np, os, joblib
import re

from warnings import filterwarnings
filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

from colorama import Fore, Style, init
from warnings import filterwarnings
filterwarnings('ignore')
from tqdm.notebook import tqdm

# Pipeline specific packages:-
from sklearn.model_selection import StratifiedKFold as SKF
from sklearn.metrics import roc_auc_score, log_loss, matthews_corrcoef 

import lightgbm as lgb, logging
from lightgbm import LGBMClassifier as LGBMC, log_evaluation, early_stopping
import catboost as cb, lightgbm as lgb
from catboost import CatBoostClassifier as CBC, Pool
from xgboost import XGBClassifier as XGBC, DMatrix

# Making sklearn pipeline outputs as dataframe:-
from sklearn import set_config
set_config(transform_output = "pandas")
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 50)
pd.set_option('display.precision', 3)
np.random.seed(42)

# Setting plots configuration:-
sns.set({"axes.facecolor"       : "#ffffff",
         "figure.facecolor"     : "#ffffff",
         "axes.edgecolor"       : "#000000",
         "grid.color"           : "#ffffff",
         "font.family"          : ['Cambria'],
         "axes.labelcolor"      : "#000000",
         "xtick.color"          : "#000000",
         "ytick.color"          : "#000000",
         "grid.linewidth"       : 0.75,  
         "grid.linestyle"       : "--",
         "axes.titlecolor"      : '#0099e6',
         'axes.titlesize'       : 8.5,
         'axes.labelweight'     : "bold",
         'legend.fontsize'      : 7.0,
         'legend.title_fontsize': 7.0,
         'font.size'            : 7.5,
         'xtick.labelsize'      : 7.5,
         'ytick.labelsize'      : 7.5,        
        })

grid_specs = {'visible': True, 'which': 'both', 'linestyle': '--', 
              'color': 'lightgrey', 'linewidth': 0.75
             };

title_specs = {'fontsize': 9, 'fontweight': 'bold', 'color': '#992600'};

# Color printing    
def PrintColor(text:str, color = Fore.BLUE, style = Style.BRIGHT):
    "Prints color outputs using colorama using a text F-string"
    print(style + color + text + Style.RESET_ALL)

class MyLogger:
    """
    This class helps to suppress logs in lightgbm and Optuna
    Source - https://github.com/microsoft/LightGBM/issues/6014
    """

    def init(self, logging_lbl: str):
        self.logger = logging.getLogger(logging_lbl)
        self.logger.setLevel(logging.ERROR)

    def info(self, message):
        pass

    def warning(self, message):
        pass

    def error(self, message):
        self.logger.error(message)
        
l = MyLogger()
l.init(logging_lbl = "lightgbm_custom")
lgb.register_logger(l)

print(f"\n---> Installation done\n")
print(f"\nLGBM = {lgb.__version__} | Catboost = {cb.__version__}\n")