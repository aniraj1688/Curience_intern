# import json


# # Opening JSON file
# f = open('importance.json')

# data = json.load(f)

# print(data['feature_importance'][0])


from feature_importance import *

print(score_cross(['crop', 'rede']))