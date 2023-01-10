from retrieval import *
from absl import app
from absl import flags
import sys
import json
FLAGS = flags.FLAGS


flags.DEFINE_string('th', '', '')
flags.FLAGS(sys.argv)
threshold = float(FLAGS.th)

th = {'threshold': threshold}

json_object = json.dumps(th, indent = 4) 
#print(json_object)
jsonFile = open('threshold.json', "w")
jsonFile.write(json_object)
jsonFile.close()


query_parse = {'color': 'red', 'length': 'crop', 'type': 'top'}

def main():
	print(compute_mini_retrievers(query_parse))
	#perform_dummy_Search()
	#print(output)
 
if __name__=="__main__":
    main()