# coding: utf-8
from __future__ import print_function

import sys
import logging
import tensorflow as tf
from tfmesos import cluster


def main(argv):
    jobs_def = [
        {
            "name": "ps",
            "num": 2
        },
        {
            "name": "worker",
            "num": 2
        },
    ]
    mesos_master = sys.argv[1]
    with cluster(jobs_def, master=mesos_master, quiet=False) as targets:
        with tf.device('/job:ps/task:0'):
            a = tf.constant(10)

        with tf.device('/job:ps/task:1'):
            b = tf.constant(32)

        with tf.device("/job:worker/task:1"):
            op = a + b

        with tf.Session(targets['/job:worker/task:0']) as sess:
            print(sess.run(op))


if __name__ == '__main__':
    logging.basicConfig()
    main(sys.argv)
