
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import tensorflow as tf
import csv
import sys
import random
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from sklearn.cross_validation import KFold


# create weight variables,
# see https://www.tensorflow.org/versions/r0.7/tutorials/mnist/pros/index.html for more information
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
def conv2d(x, W, padding):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
def max_pool_8x8(arg):
  return tf.nn.max_pool(arg, ksize=[1,8,8,1],strides=[1,8,8,1], padding='SAME')

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')
flags.DEFINE_integer('max_steps', 10000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')

def create_graph(variable_len, length_root):
    x_input = tf.placeholder(tf.float32, [None, variable_len], name='x-input')
    #x = x_input
    #c_filter = tf.Variable(tf.zeros([20,20]), name='filter')
    #strides = [1,1,1,1]

    #x = tf.nn.conv2d(x_input,c_filter, strides, 'SAME', False, name="relu_nn")
    #W1 = tf.Variable(tf.zeros([variable_len, variable_len]), name='transform')
    #
    x = x_input
    x_image = tf.reshape(x, [-1, length_root,length_root, 1])

    with tf.name_scope('feature'):
        W1 = weight_variable([10,10,1,32])
        b1 = bias_variable([32])
        h_c1 = tf.nn.relu(conv2d(x_image, W1, 'VALID') + b1)
        h_p1 = max_pool_8x8(h_c1)

        W2 = weight_variable([10,10,32,64])
        b2 = bias_variable([64])
        h_c2 = tf.nn.relu(conv2d(h_c1, W2, 'VALID') + b2)
        h_p2 = max_pool_8x8(h_c2)

        W3 = weight_variable([10,10,64,128])
        b3 = bias_variable([128])
        h_c3 = tf.nn.relu(conv2d(h_c2, W3, 'VALID') + b3)
        h_p = max_pool_8x8(h_c3)

    with tf.name_scope('cnn1'):
      W_conv1 = weight_variable([3, 3, 128, 128])
      b_conv1 = bias_variable([128])
      h_conv1 = tf.nn.relu(conv2d(h_p, W_conv1, 'SAME') + b_conv1)
      h_pool1 = max_pool_2x2(h_conv1)
    with tf.name_scope('cnn2'):
      W_conv2 = weight_variable([3, 3, 128, 128])
      b_conv2 = bias_variable([128])

      h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 'SAME') + b_conv2)
      h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('cnn3'):
      W_conv3 = weight_variable([3, 3, 128, 128])
      b_conv3 = bias_variable([128])

      h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, 'SAME') + b_conv3)
      h_pool3 = max_pool_2x2(h_conv3)


    #with tf.name_scope('cnn4'):
     # W_conv4 = weight_variable([3, 3, 128, 256])
     # b_conv4 = bias_variable([256])

      #h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
      #h_pool4 = max_pool_2x2(h_conv4)

    with tf.name_scope('dcl'):
      W_fc1 = weight_variable([1152, 1024])
      b_fc1 = bias_variable([1024])

      h_pool2_flat = tf.reshape(h_pool3, [-1, 1152])
      h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
      keep_prob = tf.placeholder(tf.float32)
      h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Use a name scope to organize nodes in the graph visualizer
    with tf.name_scope('Wx_b'):
      #x = tf.matmul(x, W1)
      W_fc2 = weight_variable([1024, 2])
      b_fc2 = bias_variable([2])

      y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
      #y = tf.nn.softmax(tf.matmul(h_fc1_drop, W) + b)
      #y1 = tf.nn.softmax(tf.matmul(x, W1) + b1)
      #y2 = tf.nn.softmax(tf.matmul(x, W2) + b2)
      #y3 = tf.nn.softmax(tf.matmul(x, W3) + b3)
      #y = tf.add_n([y0,y1,y2,y3])

    # Add summary ops to collect data
    #_ = tf.histogram_summary('transform', W1)
    #_ = tf.histogram_summary('weights', W_fc2)
    #_ = tf.histogram_summary('biases', b_fc2)
    #_ = tf.histogram_summary('y', y)
    #_ = tf.histogram_summary('weights1', W1)
    #_ = tf.histogram_summary('biases1', b1)
    #_ = tf.histogram_summary('y1', y1)

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 2], name='y-input')
    # More name scopes will clean up the graph representation
    with tf.name_scope('xent'):
      #cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
      cross_entropy = -tf.reduce_sum(y_ * tf.log(y))# - tf.reduce_sum(y_*tf.log(y1)) - tf.reduce_sum(y_*tf.log(y2)) - tf.reduce_sum(y_*tf.log(y3))  - tf.reduce_sum(y_*tf.log(y0))
      _ = tf.scalar_summary('cross entropy', cross_entropy)
    with tf.name_scope('train'):
      train_step = tf.train.AdamOptimizer(
          FLAGS.learning_rate).minimize(cross_entropy)
      tf.histogram_summary('W1', W1)

    with tf.name_scope('test'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      _ = tf.scalar_summary('accuracy', accuracy)

    # Merge all the summaries and write them out to /tmp/mnist_logs
    merged = tf.merge_all_summaries()

    return [x_input, y_, keep_prob, train_step, merged, accuracy, y]

def main(_):
    # Import data
    #mnist = input_data.read_data_sets('/tmp/data/', one_hot=True,
    #                                 fake_data=FLAGS.fake_data)
    length = 144
    length_root = 12
    f_class = open(sys.argv[1], 'rb') # opens the csv file
    f_data = open(sys.argv[2], 'rb')
    variable_len = 0;
    try:
        c_class = csv.reader(f_class)  # creates the reader object
        x=list(c_class)
        d_class =numpy.array(x).astype('float')
        #print(d_class)
        c_matrix = csv.reader(f_data)
        x=list(c_matrix)
        d_matrix = numpy.array(x).astype('float')
        d_matrix_fill = numpy.zeros([d_matrix.shape[0], length], numpy.float)
        d_matrix_fill[:d_matrix.shape[0], :d_matrix.shape[1]] = d_matrix
        d_matrix = d_matrix_fill
        variable_len = len(d_matrix[0])

        print(variable_len)
    finally:
        f_class.close()      # closing
        f_data.close()

    test_cnn(d_class, d_matrix, variable_len, length_root)

def test_cnn(d_class, d_matrix, variable_len, length_root):
    transformer = numpy.zeros([d_matrix.shape[1], variable_len], numpy.float)
    for i in range(d_matrix.shape[1]):
        transformer[i,i] = 1
    d_matrix = numpy.dot(d_matrix, transformer)
    print(d_matrix.shape[0])
    print(d_matrix.shape[1])

    # Create the model
    [
        x_input,
        y_, keep_prob,
        train_step,
        merged,
        accuracy,
        y
    ] = create_graph(variable_len, length_root)

    # Train the model, and feed in test data and record summaries every 10 steps
    cls = [];
    a = 0;
    b = 0;
    for row in d_class:
        cls.append([row == -1 and 1, row==1 and 1])
        if row == -1:
            a += 1
        b += 1
    #print(a/b)
    cls = numpy.array(cls).astype(float)

    ## the following code implemets a 10-fold cross-validation tensorflow
    kf = KFold(d_matrix.shape[0], n_folds=3)
    class_predict = numpy.zeros([d_matrix.shape[0]])
    for train_indc, test_indc in kf:
        class_predict[test_indc] = train_steps(
            d_matrix[train_indc],
            cls[train_indc],
            d_matrix[test_indc],
            cls[test_indc],
            x_input,
            y_,
            keep_prob,
            train_step,
            merged,
            accuracy,
            y)

    class_origin = numpy.zeros([cls.shape[0]])
    for i in range(cls.shape[0]):
        class_origin[i] = cls[i,0]
    print(class_origin)
    print(class_predict)
    fpr, tpr, _ = roc_curve(class_origin, class_predict)
    auc_val = auc(fpr, tpr)

    print('Overall auc: %0.4f' % (auc_val))

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % auc_val )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('./images/plot1.png', format='png')


def train_steps(
  train_data,
  train_class,
  test_data,
  test_class,
  x_input,
  y_,
  keep_prob,
  train_step,
  merged,
  accuracy,
  y):


  sess = tf.InteractiveSession()

  tf.initialize_all_variables().run()
  train_indc = range(train_data.shape[0])
  batch_size = int((train_data.shape[0])/9)


  writer = tf.train.SummaryWriter('./logs', sess.graph_def)
  #print(indc)
  for i in range(FLAGS.max_steps):

    if i % 9 == 0:  # Record summary data and the accuracy
      #if FLAGS.fake_data:
      #  batch_xs, batch_ys = mnist.train.next_batch(
      #      100, fake_data=FLAGS.fake_data)
      #  feed = {x: batch_xs, y_: batch_ys}
      #else:
      #print(indc)
      #data, cls = d_matrix[indc[i%10*21:(i%10+1)*21]], d_class[indc[i%10*21:(i%10+1)*21]]
      #for j in range(len(cls)):
      #  feed = {x: [data[j]], y_: [[cls[j]==-1 and 1,cls[j]==1 and 1]]}
      #  result = sess.run([merged, accuracy], feed_dict=feed)
      #  summary_str = result[0]
       # acc = result[1]
       # writer.add_summary(summary_str, i)
      #  print('Accuracy at step %s: %s' % (i, acc))
      #for j in range(len(cls)):
      feed = {x_input: test_data, y_:test_class, keep_prob: 0.7}
      result = sess.run([merged, accuracy, y], feed_dict=feed)
      summary_str = result[0]
      acc = result[1]
      #y_predict = result[2]
      writer.add_summary(summary_str, i)
      y_predict = numpy.zeros([result[2].shape[0]])
      y_origin = numpy.zeros([test_class.shape[0]])
      for n in range(result[2].shape[0]):
        y_predict[n] = result[2][n,0]
        y_origin[n] = test_class[n,0]

      fpr, tpr, _ = roc_curve(y_origin, y_predict)
      auc_val = auc(fpr, tpr)

      print('Accuracy at step %s: acc: %0.2f auc: %0.4f' % (i, acc, auc_val))

      if i == FLAGS.max_steps-1:

        return y_predict

      #train_indc = range(190)
      #indc = range(211)
      random.shuffle(train_indc)
      #print(indc[i%10*21:(i%10+1)*21])
    else:
      #batch_xs, batch_ys = mnist.train.next_batch(
      #    100, fake_data=FLAGS.fake_data)
      #data,cls = d_matrix[indc[i%10*21:(i%10+1)*21]],d_class[indc[i%10*21:(i%10+1)*21]]
      #for j in range(len(cls)):
      #  feed = {x: [data[j]], y_: [[cls[j]==-1 and 1,cls[j]==1 and 1]]}

      #print(i%10*batch_size)
      #print(train_data.shape)
      #print(train_indc)
      data = train_data[train_indc[i%9*batch_size:(i%9+1)*batch_size]]
      cls = train_class[train_indc[i%9*batch_size:(i%9+1)*batch_size]]
      #for j in range(len(cls)):
      feed = {x_input: data, y_:cls, keep_prob:1}
        #print(feed)
      sess.run(train_step, feed_dict=feed)

  sess.close()

if __name__ == '__main__':
    f_class = open(sys.argv[1], 'rb') # opens the csv file
    f_data = open(sys.argv[2], 'rb')
    try:
        c_class = csv.reader(f_class)  # creates the reader object
        x=list(c_class)
        d_class =numpy.array(x).astype('float')
        #print(d_class)
        c_matrix = csv.reader(f_data)
        x=list(c_matrix)
        d_matrix = numpy.array(x).astype('float')
        d_matrix = d_matrix+6;

        print(str(d_matrix.shape[0]) + "," + str(d_matrix.shape[1]))


    finally:
        f_class.close()      # closing
        f_data.close()

    n_hidden = 47089#4096#47089
    n_root = 217#64#217 #64*64=4096
    test_cnn(d_class, d_matrix, n_hidden, n_root)
