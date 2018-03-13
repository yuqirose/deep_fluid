""" 
Adapted from Seq2seq with Tensor-Train RNN
Rose Yu, March 2018 @ Caltech
"""

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy

# Import dataset data
from reader import MantaData
from model import Seq2Seq
from train_config import *


# Command line arguments 
flags = tf.flags
flags.DEFINE_string("data_path", "../data/holstm/",
          "Data input directory.")
flags.DEFINE_string("model", "LSTM", "Model used for learning.")
flags.DEFINE_string("save_path", "./log/holstm/",
          "Model output directory.")

flags.DEFINE_bool("use_error_prop", True,
                  "Feed previous output as input in RNN")
flags.DEFINE_integer("hidden_size", 16, "hidden layer size")
flags.DEFINE_float("learning_rate", 1e-2, "learning rate")
flags.DEFINE_integer("num_steps",20,"Training sequence length")

FLAGS = flags.FLAGS

# Dataset
dataset = MantaData()

# Model
model = Seq2Seq()

# Optimizer
optimizer = tf.train.RMSPropOptimizer(learning_rate=config.learning_rate)
train_op = optimizer.minimize(train_loss)

# Start training
with tf.Session() as sess:
     # Run the initializer
    sess.run(init)

    for step in range(1, training_steps+1):
        batch_x, batch_y = dataset.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, tr_loss = sess.run([merged, train_loss], feed_dict={X: batch_x,Y: batch_y})
            train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
            train_writer.add_summary(summary, step)
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(tr_loss) )
    print("Optimization Finished!")

    # Calculate accuracy for valid inps
    valid_data = dataset.validation.inps.reshape((-1, num_steps, num_input))
    valid_label = dataset.validation.outs
    va_loss = sess.run(test_loss, feed_dict={X_test: valid_data, Y_test: valid_label})
    print("Validation Loss:", va_loss)

    # Fetch test prediction
    fetches = {
        "true":Y_test,
        "pred":test_pred,
        "loss":test_loss
    }
    # Calculate accuracy for test inps
    test_data = dataset.test.inps.reshape((-1, num_test_steps, num_input))
    test_label = dataset.test.outs
    test_vals = sess.run(fetches, feed_dict={X_test: test_data, Y_test: test_label})
    print("Testing Loss:", test_vals["loss"])


