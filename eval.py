#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
#from tensorflow.contrib import learn
from helper import InputHelper
# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_string("model", "/data4/abhijeet/runs/1501247621/checkpoints/model-400", "Load trained model checkpoint (Default: None)")
tf.flags.DEFINE_string("eval_filepath", "/home/halwai/gta_data/final/", "testing folder (default: /home/halwai/gta_data/final)")
tf.flags.DEFINE_integer("max_frames", 20, "Maximum Number of frame (default: 20)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

if FLAGS.eval_filepath==None or FLAGS.model==None :
    print("Eval or Vocab filepaths are empty.")
    exit()

# load data and map id-transform based on training time vocabulary
inpH = InputHelper()
x1_test,x2_test,y_test = inpH.getTestDataSet(FLAGS.eval_filepath, FLAGS.max_frames)

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = FLAGS.model
print(checkpoint_file)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_imgs = graph.get_operation_by_name("input_imgs").outputs[0]
        input_x1 = graph.get_operation_by_name("input_x1").outputs[0]
        input_x2 = graph.get_operation_by_name("input_x2").outputs[0]
        input_y = graph.get_operation_by_name("input_y").outputs[0]

        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        # Tensors we want to evaluate
        conv_output = graph.get_operation_by_name("conv/output").outputs[0]
        predictions = graph.get_operation_by_name("output/distance").outputs[0]

        # Generate batches for one epoch
        batches = inpH.batch_iter(x1_test,x2_test,y_test, 1, 1, [[104, 114, 124], (227, 227)] ,shuffle=False)
        # Collect the predictions here
        all_predictions = []
        all_d=[]
        for (x1_dev_b,x2_dev_b,y_dev_b) in batches:
            [x1] = sess.run([conv_output], {input_imgs: x1_dev_b})
            [x2] = sess.run([conv_output], {input_imgs: x2_dev_b})
            [batch_predictions] = sess.run([predictions], {input_x1: x1, input_x2: x2, input_y:y_dev_b, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])
            print(batch_predictions)
            d = np.copy(batch_predictions)
            d[d>=0.5]=1
            d[d<0.5]=0
            batch_acc = np.mean(y_dev_b==d)
            all_d = np.concatenate([all_d, d])
            print("DEV acc {}".format(batch_acc))
        for ex in all_predictions:
            print(ex) 
        correct_predictions = float(np.mean(all_d == y_test))
        print("Accuracy: {:g}".format(correct_predictions))
