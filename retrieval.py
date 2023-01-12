from feature_importance import *
import numpy as np
import sys

query = {'color': 'red', 'length': 'crop', 'type': 'top'}

# Opening JSON file
f = open('threshold.json')
threshold = json.load(f)['threshold']


def subsets(numbers):
    if numbers == []:
        return [[]]
    x = subsets(numbers[1:])
    return x + [[numbers[0]] + y for y in x]
 
# wrapper function
def subsets_of_size(numbers, n):
    return [x for x in subsets(numbers) if len(x)==n]

def extract_important_features_single(features): # returns only the important features from {query} Returns {K1_v1: 1,...}
  ans={}
  for col,val in features.items():
    if score_single(val) >= threshold:
      ans[col+"_"+val] = 1
    else:
      ans[col+"_"+val] = 0

  return ans


def extract_important_features_cross(features): # returns only the important cross features from {query} Returns {i1, i2...}
  ans=[]
  temp = subsets_of_size(list(features.values()), 2)
  #print(temp)

  for i in temp:
    if score_cross(i) >= threshold:
      ans.append(i[0])
      ans.append(i[1])

  ans=list(np.unique(ans))
  return ans


def make_dict(query): # query: ['col_red', 'type_crop']
  ans={}
  for i in query:
    temp=i.split('_')
    starting_string = ''
    for j in range(len(temp)-1):
      starting_string += temp[j]+'_'
    starting_string = starting_string[:-1]
    ans[starting_string]=temp[-1]
  return ans

def driver(query):
  ans=[]
  
  # take into account single importance
  temp = extract_important_features_single(query) 
  confirmed=[]
  not_confirmed=[]
  
  # take into account cross importance
  temp2 = extract_important_features_cross(query)
  for i in temp2:
    confirmed.append(getCol(i)+"_"+i)

  for col,val in temp.items():
    if col in confirmed: continue
    if val == 1:
      confirmed.append(col)
    else:
      not_confirmed.append(col)

  #print("confirmed", confirmed)
    
    
  # Now make all subsets of not_confirmed and add it with confirmed and fire the query
  
  for size in range(len(not_confirmed)+1):
    temp = subsets_of_size(not_confirmed, size)
    for subset in temp:
      working_set = subset + confirmed
      #print(working_set)
      ans_dict = make_dict(working_set)
      ans.append(ans_dict)
    
  return ans



def compute_mini_retrievers(query):
    return driver(query)


#print(compute_mini_retrievers(query), threshold)