################################################################
#  SETUP                                                       #
#  Function : Setup for geodem                                 #
#  Date : 2019-11-13                                           #
#  Author : Ben Stevens                                        #
################################################################

# import libraries
global project_id,path,pd,np,mt,dt,mp,plt,pdf,mse,r2s,sqrt,DataFrame,monthrange,isleap,sqldf,PdfPages,pp,randint,pysqldf,create_grid,cl,linear_model,sm, TheilSenRegressor, RANSACRegressor, ss, confusion_matrix, gc, pyodbc, multiprocessing, time, Pool, f1_score, itertools, os, gc, logging
import numpy as np
import pandas as pd
import math as mt
import datetime as dt
import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2s
from sklearn import linear_model as linear_model
from math import sqrt as sqrt
from pandas import DataFrame as DataFrame
from calendar import monthrange as monthrange
from calendar import isleap as isleap
from matplotlib.backends.backend_pdf import PdfPages as PdfPages
from random import randint as randint
from sklearn import cluster as cl
from sklearn import linear_model as linear_model
import statsmodels.api as sm
from sklearn.linear_model import TheilSenRegressor as TheilSenRegressor
from sklearn.linear_model import RANSACRegressor as RANSACRegressor
from scipy import stats as ss
from sklearn.metrics import confusion_matrix as confusion_matrix
import gc as gc
import pyodbc as pyodbc
import multiprocessing as multiprocessing
from time import time as time
from multiprocessing import Pool as Pool
from sklearn.metrics import f1_score as f1_score
import itertools as itertools
import os as os
import gc as gc
import logging as logging


# path for project
path = 'C:/work/yellow_zebra/geodem'



