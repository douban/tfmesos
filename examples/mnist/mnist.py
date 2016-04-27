import sys
import tensorflow as tf
from input_data import read_data_sets
from tfmesos import cluster
from threading import Thread, RLock


jobs_def = [
    {
        "name": "ps",
        "num": 2
    },
    {
        "name": "worker",
        "num": 5
    },
]

_lock = RLock()
mnist = read_data_sets("MNIST_data/", one_hot=True)
master = sys.argv[1]
with cluster(jobs_def, master=master, quiet=False) as targets:
    graph = tf.Graph()
    with graph.as_default():
        with tf.device(tf.train.replica_device_setter(ps_tasks=2)):
            W = tf.Variable(tf.zeros([784, 10]))
            b = tf.Variable(tf.zeros([10]))
            global_step = tf.Variable(0)
            x = tf.placeholder(tf.float32, [None, 784])
            y = tf.nn.softmax(tf.matmul(x, W) + b)
            y_ = tf.placeholder(tf.float32, [None, 10])
            cross_entropy = -tf.reduce_sum(y_*tf.log(y))

            steps = []
            for i in xrange(5):
                with tf.device('/job:worker/task:%d' % i):
                    steps.append(tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy, global_step=global_step))

            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            init_op = tf.initialize_all_variables()

            coord = tf.train.Coordinator()

        def train(i):
            with graph.as_default():
                with tf.Session(targets['/job:worker/task:%d' % i]) as sess:
                    step = 0
                    while not coord.should_stop() and step < 10000:
                        with _lock:
                            batch_xs, batch_ys = mnist.train.next_batch(100)

                        _, step = sess.run([steps[i], global_step], feed_dict={x: batch_xs, y_: batch_ys})
                    coord.request_stop()

        with tf.Session(targets['/job:worker/task:0']) as sess:
            sess.run(init_op)
            threads = [Thread(target=train, args=(i,)) for i in xrange(5)]
            for t in threads:
                t.start()

            coord.join(threads)
            print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
