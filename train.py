#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import re
import os
import time
import datetime
import gc
from input_helpers import InputHelper
from siamese_network import SiameseLSTM
from tensorflow.contrib import learn
import gzip
from random import random
# Parameters
# ==================================================

tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_string("training_files", "person_match.train2", "training file (default: None)")
tf.flags.DEFINE_integer("hidden_units", 50, "Number of hidden units in softmax regression layer (default:50)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 300, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

if FLAGS.training_files==None:
    print "Input Files List is empty. use --training_files argument."
    exit()
 
max_document_length=30
inpH = InputHelper()
train_set, dev_set, vocab_processor,sum_no_of_batches = inpH.getDataSets(FLAGS.training_files,max_document_length, 10, FLAGS.batch_size)

# Training
# ==================================================
print("starting graph def")
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    print("started session")
    with sess.as_default():
        siameseModel = SiameseLSTM(
            sequence_length=max_document_length,
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            hidden_units=FLAGS.hidden_units,
            l2_reg_lambda=FLAGS.l2_reg_lambda,
            batch_size=FLAGS.batch_size)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        print("initialized siameseModel object")
    
    grads_and_vars=optimizer.compute_gradients(siameseModel.loss)
    tr_op_set = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    print("defined training_ops")
    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.merge_summary(grad_summaries)
    print("defined gradient summaries")
    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=100)

    # Write vocabulary
    vocab_processor.save(os.path.join(checkpoint_dir, "vocab"))

    # Initialize all variables
    sess.run(tf.initialize_all_variables())
    
    print("init all variables")
    graph_def = tf.get_default_graph().as_graph_def()
    graphpb_txt = str(graph_def)
    with open(os.path.join(checkpoint_dir, "graphpb.txt"), 'w') as f:
        f.write(graphpb_txt)


    def train_step(x1_batch, x2_batch, y_batch):
        """
        A single training step
        """
        if random()>0.5:
            feed_dict = {
                             siameseModel.input_x1: x1_batch,
                             siameseModel.input_x2: x2_batch,
                             siameseModel.input_y: y_batch,
                             siameseModel.dropout_keep_prob: FLAGS.dropout_keep_prob,
            }
        else:
            feed_dict = {
                             siameseModel.input_x1: x2_batch,
                             siameseModel.input_x2: x1_batch,
                             siameseModel.input_y: y_batch,
                             siameseModel.dropout_keep_prob: FLAGS.dropout_keep_prob,
            }
        _, step, loss, accuracy, dist = sess.run([tr_op_set, global_step, siameseModel.loss, siameseModel.accuracy, siameseModel.distance],  feed_dict)
        time_str = datetime.datetime.now().isoformat()
        d = np.copy(dist)
        d[d>=0.5]=999.0
        d[d<0.5]=1
        d[d>1.0]=0
        accuracy = np.mean(y_batch==d)
        print("TRAIN {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        print y_batch, dist, d

    def dev_step(x1_batch, x2_batch, y_batch):
        """
        A single training step
        """ 
        if random()>0.5:
            feed_dict = {
                             siameseModel.input_x1: x1_batch,
                             siameseModel.input_x2: x2_batch,
                             siameseModel.input_y: y_batch,
                             siameseModel.dropout_keep_prob: FLAGS.dropout_keep_prob,
            }
        else:
            feed_dict = {
                             siameseModel.input_x1: x2_batch,
                             siameseModel.input_x2: x1_batch,
                             siameseModel.input_y: y_batch,
                             siameseModel.dropout_keep_prob: FLAGS.dropout_keep_prob,
            }
        step, loss, accuracy, dist = sess.run([global_step, siameseModel.loss, siameseModel.accuracy, siameseModel.distance],  feed_dict)
        time_str = datetime.datetime.now().isoformat()
        d = np.copy(dist)
        d[d>=0.5]=999.0
        d[d<0.5]=1
        d[d>1.0]=0
        accuracy = np.mean(y_batch==d)
        print("DEV {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        print y_batch, dist, d
        return accuracy

    # Generate batches
    batches=inpH.batch_iter(
                list(zip(train_set[0], train_set[1], train_set[2])), FLAGS.batch_size, FLAGS.num_epochs)

    ptr=0
    max_validation_acc=0.0
    for nn in xrange(sum_no_of_batches*FLAGS.num_epochs):
        batch = batches.next()
        if len(batch)<1:
            continue
        x1_batch,x2_batch, y_batch = zip(*batch)
        if len(y_batch)<1:
            continue
        train_step(x1_batch, x2_batch, y_batch)
        current_step = tf.train.global_step(sess, global_step)
        sum_acc=0.0
        if current_step % FLAGS.evaluate_every == 0:
            print("\nEvaluation:")
            dev_batches = inpH.batch_iter(list(zip(dev_set[0],dev_set[1],dev_set[2])), FLAGS.batch_size, 1)
            for db in dev_batches:
                if len(db)<1:
                    continue
                x1_dev_b,x2_dev_b,y_dev_b = zip(*db)
                if len(y_dev_b)<1:
                    continue
                acc = dev_step(x1_dev_b, x2_dev_b, y_dev_b)
                sum_acc = sum_acc + acc
        	print("")
        if current_step % FLAGS.checkpoint_every == 0:
            if sum_acc >= max_validation_acc:
                max_validation_acc = sum_acc
                saver.save(sess, checkpoint_prefix, global_step=current_step)
                tf.train.write_graph(sess.graph.as_graph_def(), checkpoint_prefix, "graph"+str(nn)+".pb", as_text=False)
                print("Saved model {} with sum_accuracy={} checkpoint to {}\n".format(nn, max_validation_acc, checkpoint_prefix))
