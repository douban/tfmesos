# coding: utf-8

import sys
import socket
import resource
import tensorflow as tf
from tfmesos.utils import send, recv


def main(argv):
    mesos_task_id, maddr = argv[1:]
    mesos_task_id = int(mesos_task_id)
    maddr = maddr.split(':', 2)
    maddr = (maddr[0], int(maddr[1]))
    lfd = socket.socket()
    lfd.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lfd.bind(('', 0))
    addr = "%s:%s" % (socket.gethostname(), lfd.getsockname()[1])
    job_name = None
    task_index = None
    mem = None
    cpus = None
    c = socket.socket()
    c.connect(maddr)
    send(c, (mesos_task_id, addr))
    response = recv(c)
    cluster_def = response["cluster_def"]
    job_name = response["job_name"]
    task_index = response["task_index"]
    cpus = response["cpus"]
    mem = response["mem"]

    server_def = tf.train.ServerDef(
        cluster=tf.train.ClusterSpec(cluster_def).as_cluster_def(),
        job_name=job_name,
        task_index=task_index,
        protocol="grpc",
    )

    server_def.default_session_config.device_count["CPU"] = int(cpus)
    server_def.default_session_config.device_count["GPU"] = 0
    (soft, hard) = resource.getrlimit(resource.RLIMIT_AS)
    soft = min(float(mem), soft, hard)
    resource.setrlimit(resource.RLIMIT_AS, (soft, hard))

    server = tf.train.Server(server_def)

    send(c, 'ok')
    c.close()
    try:
        server.join()
    except:
        return


if __name__ == '__main__':
    sys.exit(main(sys.argv))
