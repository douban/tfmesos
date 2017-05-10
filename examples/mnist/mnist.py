import argparse
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tfmesos import cluster
from threading import Thread, RLock

parser = argparse.ArgumentParser()
parser.add_argument('-w', '--nworker', type=int, default=1)
parser.add_argument('-s', '--nserver', type=int, default=1)
parser.add_argument('-Gw', '--worker-gpus', type=int, default=0)
parser.add_argument('-C', '--containerizer_type', choices=["MESOS", "DOCKER"], type=lambda s: s.upper(), nargs='?')
parser.add_argument('-P', '--protocol', type=str)
args, cmd = parser.parse_known_args()
master = cmd[0] if cmd else None
nworker = args.nworker
nserver = args.nserver

extra_kw = {}
if args.containerizer_type:
    extra_kw['containerizer_type'] = args.containerizer_type

if args.protocol:
    extra_kw['protocol'] = args.protocol


jobs_def = [
    {
        "name": "ps",
        "num": nserver
    },
    {
        "name": "worker",
        "num": nworker,
        "gpus": args.worker_gpus,
    },
]

_lock = RLock()
mnist = read_data_sets("MNIST_data/", one_hot=True)
with cluster(jobs_def, master=master, quiet=False, **extra_kw) as c:
    graph = tf.Graph()
    with graph.as_default():
        with tf.device(tf.train.replica_device_setter(ps_tasks=nserver)):
            W = tf.Variable(tf.zeros([784, 10]))
            b = tf.Variable(tf.zeros([10]))
            global_step = tf.Variable(0)
            x = tf.placeholder(tf.float32, [None, 784])
            y = tf.nn.softmax(tf.matmul(x, W) + b)
            y_ = tf.placeholder(tf.float32, [None, 10])
            cross_entropy = -tf.reduce_sum(y_*tf.log(y))

            steps = []
            for i in range(nworker):
                with tf.device('/job:worker/task:%d' % i):
                    steps.append(tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy, global_step=global_step))

            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            init_op = tf.global_variables_initializer()
            coord = tf.train.Coordinator()

        def train(i):
            with graph.as_default():
                with tf.Session(c.targets['/job:worker/task:%d' % i]) as sess:
                    step = 0
                    while not coord.should_stop() and step < 10000:
                        with _lock:
                            batch_xs, batch_ys = mnist.train.next_batch(100)

                        _, step = sess.run([steps[i], global_step], feed_dict={x: batch_xs, y_: batch_ys})
                    coord.request_stop()

        with tf.Session(c.targets['/job:worker/task:0']) as sess:
            sess.run(init_op)
            threads = [Thread(target=train, args=(i,)) for i in range(nworker)]
            for t in threads:
                t.start()

            coord.join(threads)
            print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
