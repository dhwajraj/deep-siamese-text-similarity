#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
from input_helpers import InputHelper
# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_string("eval_filepath", "/Users/dhwaj/model_dual_20400/ll", "Evaluate on this data (Default: None)")
tf.flags.DEFINE_string("vocab_filepath", "/Users/dhwaj/model_dual_20400/vocab", "Load training time vocabulary (Default: None)")
tf.flags.DEFINE_string("model", "/Users/dhwaj/model_dual_20400/model-142200", "Load trained model checkpoint (Default: None)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

if FLAGS.eval_filepath==None or FLAGS.vocab_filepath==None or FLAGS.model==None :
    print("Eval or Vocab filepaths are empty.")
    exit()

# load data and map id-transform based on training time vocabulary
inpH = InputHelper()
x_test,y_test = inpH.getTestDataSet(FLAGS.eval_filepath, FLAGS.vocab_filepath, 600, 5)

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = FLAGS.model
print checkpoint_file
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        input_y = graph.get_operation_by_name("input_y0").outputs[0]

        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output0/scores0").outputs[0]

        accuracy = graph.get_operation_by_name("accuracy0/accuracy0").outputs[0]
        #emb = graph.get_operation_by_name("embedding/W").outputs[0]
        #embedded_chars = tf.nn.embedding_lookup(emb,input_x)
        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(zip(x_test,y_test)), 2*FLAGS.batch_size, 1, shuffle=False)
        # Collect the predictions here
        all_predictions = []

        for db in batches:
            x_dev_b,y_dev_b = zip(*db)
            batch_predictions, batch_acc = sess.run([predictions,accuracy], {input_x: x_dev_b, input_y:y_dev_b, dropout_keep_prob: 1.0})
            #all_predictions = np.concatenate([all_predictions, batch_predictions])
            #print("DEV acc {}".format(batch_acc))
            print np.argmax(y_dev_b, 1), batch_predictions
            
             
        y_simple = np.argmax(y_test, 1)
        correct_predictions = float(np.sum(all_predictions == y_simple))
        print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
