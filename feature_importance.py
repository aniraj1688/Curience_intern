import json

MODEL = None
filename = 'importance.json'

def load_model():
    f = open(filename)
    MODEL = json.load(f)
    #print(MODEL['feature_importance'][1])
    return MODEL

def score_single(val):
    MODEL = load_model()
    dict = MODEL['feature_importance'][0]
    try:
        return dict[val]
    except:
        return 0

def score_cross(i):
    MODEL = load_model()
    cross_dict = MODEL['feature_importance'][1]
    val1 = i[0]
    val2 = i[1]
    try:
        return cross_dict[val1][val2]
        
    except:
        try:
            return cross_dict[val2][val1]
        except:
            return 0

