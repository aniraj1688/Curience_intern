from retrieval import *
from absl import app
from absl import flags
import sys
import json

FLAGS = flags.FLAGS
flags.DEFINE_string('threshold', '', '')
flags.FLAGS(sys.argv)
threshold = float(FLAGS.threshold)


def write_threshold_in_file():
	th = {'threshold': threshold}
	json_object = json.dumps(th, indent = 4) 
	jsonFile = open('threshold.json', "w")
	jsonFile.write(json_object)
	jsonFile.close()

 
if __name__=="__main__":
    query_parse = {'color': 'black', 'length': 'crop', 'type': 'top'} # You can set query
    write_threshold_in_file()
    print(compute_mini_retrievers(query_parse))