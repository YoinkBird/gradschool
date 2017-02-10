
# coding: utf-8

# In[ ]:




# In[ ]:




# In[ ]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import *

# coding guidelines:
# within "problems", use descriptive varnames
# within subroutines, try to use short varnames
# http://pandas.pydata.org/pandas-docs/stable/dsintro.html
# df: dataframe
# ds: (data) series (want to avoid one-char var)

# In[ ]:


# omit years ]1890,2005[
use_simplified_dataset=0
# read in all csv again, dump comprehensive file
re_read_all_csv=0
"""
return
where to put the gender?
        year1, year2, ....
<name>  freq
"""
def read_names_pandas__df(filepath):
  name_data = DataFrame()
  # next time use glob
  # https://docs.python.org/3/library/glob.html
  startyear = 1880
  endyear = 2015
  years = [(startyear,endyear)]
  if(use_simplified_dataset):
    years = [(1880,1890)]
    years.append((2005,2015))
  for yearrange in years:
    for year in range(yearrange[0],yearrange[1]+1):
      file_path_rel = "%s/yob%s.txt" % (filepath, year)
      #readme = open(file_path_rel, "r")
      header_names = ['name','gndr','freq']
      #header_names = ['name','gndr',year]
      tmpdf = pd.read_csv(file_path_rel,names=header_names)
      # tmpdf = tmpdf.drop('gndr',axis=1) # how to remove a column
      tmpdf['year'] = year
      name_data = name_data.append(tmpdf)
  ######################################## 

  return name_data

# OPEN FILE, MULTIPLE
# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
parent_dir_path_rel = "files_pset2/Names"
all_names_csv_path_rel = ("%s/all_yob.csv" % parent_dir_path_rel)
# TODO: temporarily working on reduced dataset
if(use_simplified_dataset):
  all_names_csv_path_rel = ("%s/some_yob.csv" % parent_dir_path_rel)
data = DataFrame()
if(re_read_all_csv):
  data = read_names_pandas__df(parent_dir_path_rel)
  data.to_csv(path_or_buf=all_names_csv_path_rel)
else:
  data = pd.read_csv(all_names_csv_path_rel)
# test
pt_freq = data.pivot_table('freq', index='year', columns='gndr')
print(pt_freq.tail())
#also vik
# name_data.groupby('gndr').freq.sum()
########################################


# In[ ]:

print("""
Write a program that on input k and XXXX, returns the top k names from year XXXX.
""")
# return avg freq from year
input_year = 1882
data.pivot_table('freq',index="year").loc[input_year]

# In[ ]:

print("""
Write a program that on input Name returns the frequency for men and women of the name Name.
""")



# In[ ]:

print("""
It could be that names are more diverse now than they were in 1880, so that a name
may be relatively the most popular, though its frequency may have been decreasing over
the years. Modify the above to return the relative frequency.
""")



# In[ ]:

print("""
Find all the names that used to be more popular for one gender, but then became more
popular for another gender.
""")



# In[ ]:

print("""
(Optional) Find something cool about this data set.
""")

