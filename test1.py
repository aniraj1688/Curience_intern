import json


# Opening JSON file
f = open('importance.json')

data = json.load(f)

print(data['feature_importance'])
