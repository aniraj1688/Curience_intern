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
flags.DEFINE_string('details_dataset', None, '')
flags.DEFINE_string('titles_parsed_dataset', None, '')
flags.DEFINE_string('output_file', None, '')
flags.DEFINE_float('threshold',0.0,0.0)

# Set variables
flags.FLAGS(sys.argv) 
  
details_dataset = FLAGS.details_dataset
titles_parsed_dataset = FLAGS.titles_parsed_dataset
output_file = FLAGS.output_file
threshold = float(FLAGS.threshold) 


def preprocess(details_dataset, titles_parsed_dataset):  # Function to preprocess the dataframes and returns the dataframes and columns
  # Upload the two files first
  df1=pd.read_csv(details_dataset)
  df2=pd.read_csv(titles_parsed_dataset)

  df2 = df2.replace('-', np.nan)

  #Final columns
  cols=[]
  for i in df2.columns:
    if i in df1.columns and i != 'title':
      cols.append(i)
  return df1,df2,cols

def merge_dataframes(df1,df2,cols): # FUnction to Inner join both dataframes
  # using merge function by setting how='inner'
  df = pd.merge(df1, df2, 
                    on='title', 
                    how='inner')
  
  # take out columns with _x and _y
  final_cols=[]
  for col in cols:
    if (col+"_x") in df.columns and (col+"_y") in df.columns:
      final_cols.append(col)
  return df,final_cols

def make_sure_title_issubset_of_body(df): 
  for id in df.index:
    for col in final_cols:
      if pd.isnull(df[col+"_x"][id]) and not pd.isnull(df[col+"_y"][id]):
        df.loc[id, col+"_x"] = df[col+"_y"][id]
  #df is the final data
  return df

def remove_unnecessary_columns(df):
  x=[]
  for i in df.columns:
    if '_x' not in i and '_y' not in i:
      x.append(i)
  df = df.drop(x, axis=1)
  # Now, our df is ready
  return df

#--------------------------------------------------------------------------------------------------------------------

# To Calculate Importance of one feature value


#To calculate no. of time a value occurs in a particular feature in df
def count_freq(df, col, value):
  c=0
  for i in df[col]:
    if value in str(i):
      c+=1
  return c
  #count_freq('colodf, r_x','blue')


def calc_importance(df, col, val):
  a = df[col+"_y"].tolist().count(val) # Numerator i.e occurence of the value in title 
  b = count_freq(df, col+"_x", val)    # Denominator i.e occurence of the value in the dataset
  if b==0:
    return 0
  return(a/b)

#print(calc_importance('color', 'green'))

def make_single_dict(df, cols):
  # Calculate and store importance of all individual features : Around 550 features
  dict={}
  for col in cols:
    for val in df[col+"_y"].unique():
      if not pd.isnull(val):
        dict[val]=round(calc_importance(df, col, val),4)
  return dict

# The final function
def score_single(dict, val):
  try:
    return dict[val]
  except:
    return 0

#print(score_single('crop'))


#-----------------------------------------------------------------------------------------

# TO Deal with cross features: Total around 6k feature crosses

def make_dicts_cross(df, final_cols): # Return frequency distribution of key: value pair for both data and parsed_titles
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
  return d1, d2
      

def in_dict2(i,j,d2): # Checks whether a key: value pair is in d2 dictionary
  if i not in d2.keys():
    return False
  if j not in d2[i].keys():
    return False
  return True

def make_cross_dict(d1, d2):
  cross_dict={}
  for i in d1.keys():
    cross_dict[i]={}
    for j in d1[i].keys():
      if not in_dict2(i,j, d2): 
        continue 
      cross_dict[i][j] = round(d2[i][j]/d1[i][j],4)
  return cross_dict
  
#print(cross_dict)
    
# Final function
def score_cross(i, cross_dict):
  val1 = i[0]
  val2 = i[1]
  try:
    return cross_dict[val1][val2]
    
  except:
    try:
      return cross_dict[val2][val1]
    except:
      return 0

def val_to_col(final_cols, df2):# To get Column name from a given value
  val_to_col_dict={}
  for col in final_cols:
    for j in df2[col].unique():
      if pd.isnull(j): continue
      val_to_col_dict[j] = col
  return val_to_col_dict
  
def make_json_file_with_data(dict, cross_dict, val_to_col_dict, output_file): 
  final_dict={'feature_importance': [dict, cross_dict, val_to_col_dict]}
  json_object = json.dumps(final_dict, indent = 4) 
  jsonFile = open(output_file, "w")
  jsonFile.write(json_object)
  jsonFile.close()

#---------------------------------------------------------------------------------------
# Precision Calculation

def calculate_precision(dict, df, final_cols):
  # Precision Calculation:  
  # Precision = TruePositives / (TruePositives + FalsePositives)
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
        imp = score_single(dict, val.strip())
        if imp > threshold:
          if val == val2:
            true_positives += 1
          else:
            false_positives += 1
  return true_positives/(true_positives + false_positives)


# Main Function
if __name__=="__main__":  
  df1,df2,cols = preprocess(details_dataset, titles_parsed_dataset)
  df, final_cols = merge_dataframes(df1,df2,cols)
  df = make_sure_title_issubset_of_body(df)
  df = remove_unnecessary_columns(df)
  dict = make_single_dict(df, cols)
  d1,d2 = make_dicts_cross(df, final_cols)
  cross_dict = make_cross_dict(d1,d2)
  val_to_col_dict = val_to_col(final_cols, df2)
  
  # Make final json file and calculate precision
  
  make_json_file_with_data(dict, cross_dict, val_to_col_dict, output_file) 
  precision = calculate_precision(dict, df, final_cols)
  print("Precision with threshold ",threshold, "is:  ", precision)
    
