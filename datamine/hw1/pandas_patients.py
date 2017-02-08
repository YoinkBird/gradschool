import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import *

# OPEN FILE
# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
# replace '?' with NaN
# na_values : scalar, str, list-like, or dict, default None
#     Additional strings to recognize as NA/NaN. If dict passed, specific per-column NA values.
#     By default the following values are interpreted as NaN: '', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan',
# see also:
#    replace: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.replace.html
#    data = data.replace(to_replace='?', value=np.nan)
file_path_rel = "files_pset1/PatientData.csv"
data = pd.read_csv(file_path_rel, header=None, na_values='?') # ,delimiter=",")
########################################

print("""
Are there missing values? Replace them with the average of the corresponding feature column.
""")
# NOTE: data[13] has missing valeus
# examine with: data.fillna(data.mean())[13]
#  http://pandas.pydata.org/pandas-docs/stable/missing_data.html#filling-with-a-pandasobject
data.fillna(data.mean())

# dis for breakpoitns
print("")
