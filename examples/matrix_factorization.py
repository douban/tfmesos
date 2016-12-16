# coding: utf-8
from __future__ import print_function
import sys
import numpy as np
import tensorflow as tf
from tfmesos import cluster



INFINITY = 10e+12


class NMF(object):

    def __init__(self, session, np_matrix, rank,
                 learning_rate=0.1):
        matrix = tf.constant(np_matrix, dtype=tf.float32)
        scale = 2 * np.sqrt(np_matrix.mean() / rank)
        initializer = tf.random_uniform_initializer(maxval=scale)

        with tf.device('/job:ps/task:0'):
            self.matrix_W = tf.get_variable(
                "W", (np_matrix.shape[0], rank), initializer=initializer
            )
        with tf.device("/job:ps/task:1"):
            self.matrix_H = tf.get_variable(
                "H", (rank, np_matrix.shape[1]), initializer=initializer
            )

        matrix_WH = tf.matmul(self.matrix_W, self.matrix_H)
        f_norm = tf.reduce_sum(tf.pow(matrix - matrix_WH, 2))

        nn_w = tf.reduce_sum(tf.abs(self.matrix_W) - self.matrix_W)
        nn_h = tf.reduce_sum(tf.abs(self.matrix_H) - self.matrix_H)
        constraint = INFINITY * (nn_w + nn_h)
        self.loss = f_norm + constraint
        self.constraint = constraint

        self.session = session
        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate
        ).minimize(self.loss)

    def run(self):
        self.session.run(tf.initialize_all_variables())

        with tf.device("/job:worker/task:0"):
            current_loss, current_constraint, _ = self.session.run([
                self.loss, self.constraint, self.optimizer
            ])

        return self.matrix_W.eval(), self.matrix_H.eval(), self.loss.eval()


def main(argv):
    max_iter = 100
    matrix = np.random.random((1000, 1000))
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
    mesos_master = argv[1]
    with cluster(jobs_def, master=mesos_master) as targets:
        with tf.Session(targets['/job:worker/task:1']) as session:
            nmf = NMF(session, matrix, 200)
            for i in range(max_iter):
                mat_w, mat_h, loss = nmf.run()
                print("loss#%d: %s" % (i, loss))

    err = np.power(matrix - np.matmul(mat_w, mat_h), 2)
    print("err mean: %s" % err.mean())
    print("loss: %s" % loss)


if __name__ == '__main__':
    main(sys.argv)
