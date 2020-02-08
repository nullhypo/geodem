################################################################
#  SETUP                                                       #
#  Function : Setup for geodem                                 #
#  Date : 2019-11-13                                           #
#  Author : Ben Stevens                                        #
################################################################

# import libraries
global path, pd, np, sc, mp, sns, plt, sk, mt, sqldf, linear_model, pyodbc, datetime, pdf, PdfPages, spatial, md, PCA, gc, figure, pp, cl, Axes3D, ss, os, psql
import pandas as pd
import numpy as np
import scipy as sc
import matplotlib as mp
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk
import math as mt
import matplotlib.backends.backend_pdf as pdf
from matplotlib.backends.backend_pdf import PdfPages as PdfPages
from scipy import spatial as spatial
from sklearn import manifold as md
from sklearn.decomposition import PCA as PCA
from pylab import figure
from sklearn import preprocessing as pp
from sklearn import cluster as cl
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats as ss
import os as os

import pandasql as psql

# path for project
path = 'C:/work/yellow_zebra/geodem'



