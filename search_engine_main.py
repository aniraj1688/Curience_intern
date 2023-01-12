from retrieval import *
from absl import app
from absl import flags
import sys
import json

FLAGS = flags.FLAGS
flags.DEFINE_float('threshold',0.0,0.0)
flags.FLAGS(sys.argv)
threshold = FLAGS.threshold

 
if __name__=="__main__":
    query_parse = {'color': 'black', 'length': 'crop', 'type': 'top'} # You can set query
    print(compute_mini_retrievers(query_parse, threshold))