# coding: utf-8

import os
import sys
import socket
import subprocess
import tensorflow as tf
import logging
from tfmesos.utils import send, recv

logger = logging.getLogger(__name__)


def main(argv):
    mesos_task_id, maddr = argv[1:]
    maddr = maddr.split(':', 2)
    maddr = (maddr[0], int(maddr[1]))
    lfd = socket.socket()
    lfd.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lfd.bind(('', 0))
    addr = "%s:%s" % (socket.gethostname(), lfd.getsockname()[1])
    job_name = None
    task_index = None
    cpus = None
    c = socket.socket()
    c.connect(maddr)
    send(c, (mesos_task_id, addr))
    response = recv(c)
    cluster_def = response["cluster_def"]
    job_name = response["job_name"]
    task_index = response["task_index"]
    cpus = response["cpus"]
    gpus = response["gpus"]
    cmd = response["cmd"]
    cwd = response["cwd"]
    extra_config = response['extra_config']

    forward_addresses = response['forward_addresses']
    protocol = response['protocol']

    forward_fd = None
    grpc_addr = '/job:%s/task:%s' % (job_name, task_index)
    if forward_addresses and grpc_addr in forward_addresses:
        addr = forward_addresses[grpc_addr]
        forward_fd = socket.socket()
        forward_fd.connect(addr)

    send(c, 'ok')
    c.close()

    if cmd is None:
        server_def = tf.train.ServerDef(
            cluster=tf.train.ClusterSpec(cluster_def).as_cluster_def(),
            job_name=job_name,
            task_index=task_index,
            protocol=protocol,
        )

        server_def.default_session_config.device_count["CPU"] = int(cpus)
        server_def.default_session_config.device_count["GPU"] = int(gpus)
        server = tf.train.Server(server_def)

        try:
            server.join()
        except Exception:
            return 0
    else:
        initial_cmd = extra_config.get('initializer')
        if initial_cmd is not None:
            subprocess.check_call(initial_cmd, shell=True)

        server_name = 'ps'
        worker_name = 'worker'
        ps_hosts = ','.join(cluster_def[server_name])
        worker_hosts = ','.join(cluster_def[worker_name])
        env = os.environ.copy()
        env.update({
            'PYTHONUNBUFFERED': '1',
            'TFMESOS_PS_HOSTS': ps_hosts,
            'TFMESOS_WORKER_HOSTS': worker_hosts,
            'TFMESOS_JOB_NAME': job_name,
            'TFMESOS_TASK_INDEX': str(task_index),
            'TFMESOS_DISTRIBUTED': '1'
        })

        prefix = '[%s:%s] ' % (job_name, task_index)
        prefix = prefix.encode('ascii')

        cmd = cmd.format(
            ps_hosts=ps_hosts, worker_hosts=worker_hosts,
            job_name=job_name, task_index=task_index
        )
        out = os.fdopen(sys.stdout.fileno(), 'wb', 1)
        try:
            p = subprocess.Popen(
                cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE,
                bufsize=1, env=env
            )
            for l in iter(p.stdout.readline, b''):
                out.write(l)
                if forward_fd:
                    forward_fd.send(prefix + l)

            return p.wait()
        finally:
            final_cmd = extra_config.get('finalizer')
            if final_cmd is not None:
                logger.info('Running clean up command {}'.format(final_cmd))
                subprocess.check_call(final_cmd, shell=True)

            out.close()
            if forward_fd:
                forward_fd.close()


if __name__ == '__main__':
    sys.exit(main(sys.argv))
