---
layout: post
comments: true
categories: DeepLearning
---

* content
{:toc}

## Tensorlfow教程3 MNIST feed-forward neuron network
<p>Feed-forward神经网络是一种非常简单的神经网络，也是最基本的神经网络，和第一篇不同的是,这里的神经网络有多层,不过实际上效果并没有比第一篇文章的好，如但是还是直接参考的代码的。这里理清楚一个概念：Feed-forward neuro network包括了Single-layer perceptron以及Multi-layer perceptron，第一篇文章是single-layer的实现，下面就是Multi-layer perceptron的实现。下面是摘自维基百科的概念：</p>
> The feedforward neural network was the first and simplest type of artificial neural network devised. In this network, the information moves in only one direction, forward, from the input nodes, through the hidden nodes (if any) and to the output nodes. There are no cycles or loops in the network.
[第一篇教程](http://shuaizhang.tech/2016/11/29/Tensorflow%E6%95%99%E7%A8%8B1-MNIST/)
### TensorBoard
<p>本来这篇文章的很重要的一部分就是介绍tensorboard，但是由于目前tensorflow的一些问题，我的电脑暂时不能使用，所以我目前也没有看到真实的效果
，下面是官网的解释：</p>
> Currently, TensorBoard external dependencies (JS scripts, css, images etc.) are not part of PIP package built by CMake. As a result, when you navigate to TensorBoard on Windows, it shows a blank screen. Added CMake scripts to download TensorBoard dependencies and make them part of PIP package.
[issue5844](https://github.com/tensorflow/tensorflow/pull/5844)
<p>tensorboard的目的是让神经网络的结果可视化，但前提是要自己利用API(summary)去采集log,按照规范才能出来结果。官网给的效果图如下：</p>
![tensorboard](https://www.tensorflow.org/versions/r0.12/images/mnist_tensorboard.png)

<p>使用tensorboard首先按照代码，打印出相应的Log，然后使用下面的命令打开tensorboard,然后按照提示进入浏览器，输入查看地址：如：http://localhost:6006/</p>
>python -m tensorflow.tensorboard --logdir=path/to/log-directory

### 实现代码
* 文件1，mnist.py
{% highlight python %}
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

def inference(images, hidden1_units,hidden2_units):
    '''
    :param images:  image placeholder
    :param hidden1_units: size of first hidden layer
    :param hidden2_units: size of 2nd hidden layer
    :return:
    '''

    # hidden 1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                                                  stddev=1.0/ math.sqrt(float(IMAGE_PIXELS))),
                              name="weights")
        biases = tf.Variable(tf.zeros([hidden1_units]),name="biases")
        hidden1 = tf.nn.relu(tf.matmul(images, weights) +  biases)

    #hidden 2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units],
                                                  stddev=1.0/ math.sqrt(float(hidden1_units))),
                              name="weights")
        biases = tf.Variable(tf.zeros([hidden2_units]),name="biases")
        hidden2 = tf.nn.relu(tf.matmul(hidden1_units, weights) +  biases)

    # linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(tf.truncated_normal([hidden2_units, NUM_CLASSES],
                                                  stddev=1.0/ math.sqrt(float(hidden2_units))),
                              name="weights")
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),name="biases")
        logits = tf.nn.relu(tf.matmul(hidden1_units, weights) +  biases)

    return logits

def loss(logits, labels):
    '''
    loss function
    :param logits: logits tensor [batch_size, NUM_CLASSES]
    :param labels: labels tensor, [batch_size]
    :return:
    '''
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name="xentropy"
    )
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss

def training(loss, learning_rate):
    tf.summary.scaler('loss',loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='gloabal_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step = global_step)
    return train_op

def evalution(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))
{% endhighlight %} 

* 代码2
{% highlight python %}
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

FLAGS = None

def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(tf.float32, shape=(
        batch_size, mnist.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))

    return images_placeholder, labels_placeholder

def fill_feed_dict(data_set, images_pl, labels_pl):
    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                                   FLAGS.fake_data)
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed
    }
    return feed_dict

def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
    true_count = 0
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set, images_placeholder,
                                   labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)

    precision = true_count / num_examples
    print('num examples：%d Num correct: %d Precision @ 1: %0.04f' %(num_examples, true_count, precision))

def run_training():
    data_sets = input_data.read_data_sets(FLAGS.input_data_dir,
                                          FLAGS.fake_data)

    with tf.Graph().as_default():
        images_placeholder, labels_placeholder = placeholder_inputs(
            FLAGS.batch_size
        )

        logits = mnist.inference(images_placeholder,FLAGS.hidden1,
                                 FLAGS.hidden2)
        loss = mnist.loss(logits, labels_placeholder)

        train_op = mnist.training(loss, FLAGS.learning_rate)

        eval_correct = mnist.evaluation(logits, labels_placeholder)

        summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        # create a session for running Ops on the Graph.
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

        sess = tf.Session()

        summary_writer = tf.train.SummaryWriter(FLAGS.log_dir, sess.graph)

        sess.run(init)

        for step in range(FLAGS.max_steps):
            start_time = time.time()

            feed_dict = fill_feed_dict(data_sets.train,
                                       images_placeholder,
                                       labels_placeholder)
            _,loss_value = sess.run([train_op,loss],
                                    feed_dict=feed_dict)
            duration = time.time() - start_time

            #
            if step % 100 == 0:
                print('Step %d: loss = %.2f(%.3f sec)' % (step, loss_value, duration))
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            # save a checkpoint and evaluate the model periodically
            if ( step +1)%1000 == 0 or (step +1) == FLAGS.max_steps:
                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)

                # evaluate against the training set
                print('Training data eval')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.train)
                print('Validation data eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.validation)
                print('Test data eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.test)

def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--max_steps',
        type= int,
        default=2000,
        help='Number of steps to run trainer.'
    )

    parser.add_argument(
        '--hidden1',
        type= int,
        default=128,
        help='Number of units in hidden layer 1.'
    )

    parser.add_argument(
        '--hidden2',
        type= int,
        default=32,
        help='Number of units in hidden layer 2.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size.  Must divide evenly into the dataset sizes.'
     )

    parser.add_argument(
        '--input_data_dir',
        type=str,
        default='dataset',
        help='Directory to put the input data.'
    )
    parser.add_argument(
      '--log_dir',
      type=str,
      default='logs/fully_connected_feed',
      help='Directory to put the log data.'
    )
    parser.add_argument(
      '--fake_data',
      default=False,
      help='If true, uses fake data for unit testing.',
      action='store_true'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]]+unparsed)
{% endhighlight %} 
### 输出结果
> Step 0: loss = 2.31(0.081 sec)
Step 100: loss = 2.12(0.025 sec)
Step 200: loss = 1.80(0.025 sec)
Step 300: loss = 1.53(0.024 sec)
Step 400: loss = 1.30(0.027 sec)
Step 500: loss = 0.82(0.028 sec)
Step 600: loss = 0.90(0.025 sec)
Step 700: loss = 0.58(0.025 sec)
Step 800: loss = 0.53(0.027 sec)
Step 900: loss = 0.43(0.025 sec)
Training data eval
num examples：55000 Num correct: 48013 Precision @ 1: 0.8730
Validation data eval:
num examples：5000 Num correct: 4385 Precision @ 1: 0.8770
Test data eval:
num examples：10000 Num correct: 8802 Precision @ 1: 0.8802
Step 1000: loss = 0.42(0.108 sec)
Step 1100: loss = 0.47(0.653 sec)
Step 1200: loss = 0.45(0.027 sec)
Step 1300: loss = 0.35(0.029 sec)
Step 1400: loss = 0.41(0.027 sec)
Step 1500: loss = 0.46(0.025 sec)
Step 1600: loss = 0.38(0.026 sec)
Step 1700: loss = 0.51(0.028 sec)
Step 1800: loss = 0.65(0.024 sec)
Step 1900: loss = 0.46(0.025 sec)
Training data eval
num examples：55000 Num correct: 49604 Precision @ 1: 0.9019
Validation data eval:
num examples：5000 Num correct: 4546 Precision @ 1: 0.9092
Test data eval:
num examples：10000 Num correct: 9052 Precision @ 1: 0.9052
<p>可以看出，这里的精确度才0.909，比第一次还差一下，同时这也说明了CNN在处理图片中的厉害之处！</p>

### Reference
1. https://www.tensorflow.org/versions/r0.12/tutorials/mnist/tf/index.html
