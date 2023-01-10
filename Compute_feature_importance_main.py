import pandas as pd
import collections
import itertools
from itertools import permutations
import numpy as np
from matplotlib import pyplot as plt
import json
import sys
from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('file1', None, '')
flags.DEFINE_string('file2', None, '')
flags.DEFINE_string('file3', None, '')
flags.DEFINE_string('th', None, '')
# flags.DEFINE_integer('th', None, '')


# Set variables

flags.FLAGS(sys.argv)
#print(FLAGS.file1, FLAGS.file2, float(FLAGS.th))
query = {'color': 'red', 'length': 'crop', 'type': 'top'}
file1 = FLAGS.file1
file2 = FLAGS.file2
file3 = FLAGS.file3
threshold = float(FLAGS.th)


# Upload the two files first
df1=pd.read_csv(file1)
df2=pd.read_csv(file2)
# df1=pd.read_csv("Copy of TrainingDataset - quality_dataset.csv")
# df2=pd.read_csv("Copy of TrainingDataset - parsed_titles.csv")

#df3=pd.read_csv("df3 - PARSED_QUERY_FOR_MINI_RETRIEVER.csv")

df2 = df2.replace('-', np.nan)

#Final columns
cols=[]
for i in df2.columns:
  if i in df1.columns and i != 'title':
    cols.append(i)

# using merge function by setting how='inner'
df = pd.merge(df1, df2, 
                   on='title', 
                   how='inner')

# take out columns with _x and _y
final_cols=[]
for col in cols:
  if (col+"_x") in df.columns and (col+"_y") in df.columns:
    final_cols.append(col)
    
# using merge function by setting how='inner'
df = pd.merge(df1, df2, 
                   on='title', 
                   how='inner')

# take out columns with _x and _y
final_cols=[]
for col in cols:
  if (col+"_x") in df.columns and (col+"_y") in df.columns:
    final_cols.append(col)

#data1.loc[i, 'other_id_phone'] = some_value

for id in df.index:
  for col in final_cols:
    if pd.isnull(df[col+"_x"][id]) and not pd.isnull(df[col+"_y"][id]):
      df.loc[id, col+"_x"] = df[col+"_y"][id]

#df is the final data

x=[]
for i in df.columns:
  if '_x' not in i and '_y' not in i:
    x.append(i)
df = df.drop(x, axis=1)

# Now, our df is ready
#print(df)

#--------------------------------------------------------------------------------------------------------------------

# To Calculate Importance of one feature value




#To calculate no. of time a value occurs in a particular feature in df
def count_freq(col, value):
  c=0
  for i in df[col]:
    if value in str(i):
      c+=1
  return c

#count_freq('color_x','blue')

# To Calculate importance of categories
imp_cat={}
for col in final_cols:
  a = len(df)-df[col+'_x'].isnull().sum()
  b = len(df)-df[col+'_y'].isnull().sum()
  imp_cat[col] = round(b/a,4)

#imp_cat

# Calclate importance of a single feature value

def calc_importance(col, val):
  a = df[col+"_y"].tolist().count(val) # Numerator i.e occurence of the value in title 
  b = count_freq(col+"_x", val)          # Denominator i.e occurence of the value in the dataset
  if b==0:
    #print("Den zeo: ", col, val)
    return 0
   
  return(a/b)


#print(calc_importance('color', 'green'))

# Calculate and store importance of all individual features : Around 550 features
dict={}
for col in cols:
  for val in df[col+"_y"].unique():
    if not pd.isnull(val):
      dict[val]=round(calc_importance(col, val),4)



# The final function
def score_single(val):
  try:
    return dict[val]
  except:
    return 0

#print(score_single('crop'))


#-----------------------------------------------------------------------------------------

# TO Deal with cross features: Total around 6k feature crosses


# Calculate the frequency dictionary for df1 data
d1={}

for index, row in df.iterrows():

  for i in range(len(final_cols)):
    cur_col = final_cols[i]
    cur_col+="_x"

    temp1 = row[cur_col]
    if pd.isnull(temp1): continue

    for v1 in temp1.split('|'):
      val1 = v1.strip()

      for j in range(i+1,len(final_cols)):
        next_col=final_cols[j]
        next_col+="_x"
        temp2 = row[next_col]
        if pd.isnull(temp2): continue
        
        for v2 in temp2.split('|'):
          val2 = v2.strip()
          if val1 not in d1.keys():
            d1[val1]={}
          if val2 not in d1[val1].keys():
            d1[val1][val2]=0
          
          d1[val1][val2] += 1

# Calculate the frequency dictionary for df2 data
d2={}

for index, row in df.iterrows():

  for i in range(len(final_cols)):
    cur_col = final_cols[i]
    cur_col+="_y"
    val1 = row[cur_col]
    if pd.isnull(val1): continue
    
    for j in range(i+1,len(final_cols)):
      next_col=final_cols[j]
      next_col+="_y"
      val2 = row[next_col]
      if pd.isnull(val2): continue
      
      if val1 not in d2.keys():
        d2[val1]={}
      if val2 not in d2[val1].keys():
        d2[val1][val2]=0
      
      d2[val1][val2] += 1
      

def in_dict2(i,j):
  if i not in d2.keys():
    return False
  if j not in d2[i].keys():
    return False
  return True

cross_dict={}
for i in d1.keys():
  cross_dict[i]={}
  for j in d1[i].keys():
    if not in_dict2(i,j): 
      continue 
    cross_dict[i][j] = round(d2[i][j]/d1[i][j],4)
  
#print(cross_dict)
    
# Final function
def score_cross(i):
  val1 = i[0]
  val2 = i[1]
  try:
    return cross_dict[val1][val2]
    
  except:
    try:
      return cross_dict[val2][val1]
    except:
      return 0

# To get Column name from a given value
val_to_col={}
for col in final_cols:
  for j in df2[col].unique():
    if pd.isnull(j): continue
    val_to_col[j] = col
  
    
final_dict={'feature_importance': [dict, cross_dict, val_to_col]}

json_object = json.dumps(final_dict, indent = 4) 
#print(json_object)
jsonFile = open(file3, "w")
jsonFile.write(json_object)
jsonFile.close()

#---------------------------------------------------------------------------------------

# TO Calculate Precision
    # Precision Calculation:  
#Precision = TruePositives / (TruePositives + FalsePositives)


df_rand = df.sample(n = int(0.3 * len(df)))

true_positives = 0
false_positives = 0

for index, row in df_rand.iterrows():
  for col in final_cols:
    val1 = row[col+"_x"]
    val2 = row[col+"_y"]
    if pd.isnull(val1): continue
    
    temp = val1.split('|')
    for val in temp:
      imp = score_single(val.strip())
      if imp > threshold:
        if val == val2:
          true_positives += 1
        else:
          false_positives += 1


print("True Positives: ",true_positives, "False Positives: ", false_positives)
print("Precision with threshold ",threshold, "is:  ", true_positives/(true_positives + false_positives))
