import os
import re
import sys
import math
import select
import signal
import socket
import thread
import getpass
import logging
import urllib2
import textwrap
from mesos.interface import mesos_pb2, Scheduler
from pymesos import MesosSchedulerDriver
from tfmesos.utils import send, recv, setup_logger


FOREVER = 0xFFFFFFFF
logger = logging.getLogger(__name__)


class Job(object):

    def __init__(self, name, num, cpus=1.0, mem=1024.0, gpus=0):
        self.name = name
        self.num = num
        self.cpus = cpus
        self.gpus = gpus
        self.mem = mem


class Task(object):

    def __init__(self, mesos_task_id, job_name, task_index,
                 cpus=1.0, mem=1024.0, gpus=0, volumes={}):
        self.mesos_task_id = mesos_task_id
        self.job_name = job_name
        self.task_index = task_index

        self.cpus = cpus
        self.gpus = gpus
        self.mem = mem
        self.volumes = volumes
        self.offered = False

        self.addr = None
        self.connection = None
        self.initalized = False

    def __str__(self):
        return textwrap.dedent('''
        <Task
          mesos_task_id=%s
          addr=%s
        >''' % (self.mesos_task_id, self.addr))

    def to_task_info(self, offer, master_addr, gpu_uuids=[]):
        ti = mesos_pb2.TaskInfo()
        ti.task_id.value = str(self.mesos_task_id)
        ti.slave_id.value = offer.slave_id.value
        ti.name = '/job:%s/task:%s' % (self.job_name, self.task_index)

        cpus = ti.resources.add()
        cpus.name = 'cpus'
        cpus.type = mesos_pb2.Value.SCALAR
        cpus.scalar.value = self.cpus

        mem = ti.resources.add()
        mem.name = 'mem'
        mem.type = mesos_pb2.Value.SCALAR
        mem.scalar.value = self.mem

        image = os.environ.get('DOCKER_IMAGE')

        if image is not None:
            ti.container.type = mesos_pb2.ContainerInfo.DOCKER
            ti.container.docker.image = image

            for path in ['/etc/passwd', '/etc/group']:
                v = ti.container.volumes.add()
                v.host_path = v.container_path = path
                v.mode = mesos_pb2.Volume.RO

            for dst, src in self.volumes.iteritems():
                v = ti.container.volumes.add()
                v.container_path = dst
                v.host_path = src
                v.mode = mesos_pb2.Volume.RW

            if self.gpus and gpu_uuids:
                hostname = offer.hostname
                url = 'http://%s:3476/docker/cli?dev=%s' % (
                    hostname, urllib2.quote(
                        ' '.join(gpu_uuids)
                    )
                )

                try:
                    docker_args = urllib2.urlopen(url).read()
                    for arg in docker_args.split():
                        k, v  = arg.split('=')
                        assert k.startswith('--')
                        k = k[2:]
                        p = ti.container.docker.parameters.add()
                        p.key = k
                        p.value = v

                    gpus = ti.resources.add()
                    gpus.name = 'gpus'
                    gpus.type = mesos_pb2.Value.SET
                    gpus.set.item.extend(gpu_uuids)
                except Exception:
                    logger.exception(
                        'fail to determine remote device parameter,'
                        ' disable gpu resources'
                    )

        else:
            if self.gpus and gpu_uuids:
                gpus = ti.resources.add()
                gpus.name = 'gpus'
                gpus.type = mesos_pb2.Value.SET
                gpus.set.item.extend(gpu_uuids)

        ti.command.shell = True
        cmd = [
            sys.executable, "-m", "%s.server" % __package__,
            str(self.mesos_task_id), master_addr
        ]
        ti.command.value = ' '.join(cmd)
        env = ti.command.environment.variables.add()
        env.name = 'PYTHONPATH'
        env.value = ':'.join(sys.path)
        return ti

_exc_info = None


def _raise(e):
    global _exc_info
    try:
        raise e
    except:
        _exc_info = sys.exc_info()
        thread.interrupt_main()


def _handle_sigint(signum, frame):
    global _prev_handler, _exc_info
    assert signum == signal.SIGINT
    if _exc_info is not None:
        raise _exc_info[1], None, _exc_info[2]
    elif _prev_handler is not None:
        return _prev_handler(signum, frame)

    raise KeyboardInterrupt

_prev_handler = signal.signal(signal.SIGINT, _handle_sigint)


class TFMesosScheduler(Scheduler):

    def __init__(self, task_spec, master=None, name=None, quiet=False,
                 volumes={}):
        self.started = False
        self.master = master or os.environ['MESOS_MASTER']
        self.name = name or '[tensorflow] %s %s' % (
            os.path.abspath(sys.argv[0]), ' '.join(sys.argv[1:]))
        self.task_spec = task_spec
        self.tasks = []
        for job in task_spec:
            for task_index in xrange(job.num):
                mesos_task_id = len(self.tasks)
                self.tasks.append(
                    Task(
                        mesos_task_id,
                        job.name,
                        task_index,
                        cpus=job.cpus,
                        mem=job.mem,
                        gpus=job.gpus,
                        volumes=volumes,
                    )
                )
        if not quiet:
            global logger
            setup_logger(logger)

    def resourceOffers(self, driver, offers):
        '''
        Offer resources and launch tasks
        '''

        for offer in offers:
            if all(task.offered for task in self.tasks):
                driver.declineOffer(offer.id,
                                    mesos_pb2.Filters(refuse_seconds=FOREVER))
                continue

            offered_cpus = offered_mem = 0.0
            offered_gpus = []
            offered_tasks = []

            for resource in offer.resources:
                if resource.name == "cpus":
                    offered_cpus = resource.scalar.value
                elif resource.name == "mem":
                    offered_mem = resource.scalar.value
                elif resource.name == "gpus":
                    offered_gpus = resource.set.item

            for task in self.tasks:
                if task.offered:
                    continue

                if not (task.cpus <= offered_cpus and
                        task.mem <= offered_mem and
                        task.gpus <= len(offered_gpus)):

                    continue

                offered_cpus -= task.cpus
                offered_mem -= task.mem
                gpus = int(math.ceil(task.gpus))
                gpu_uuids = offered_gpus[:gpus]
                offered_gpus = offered_gpus[gpus:]
                task.offered = True
                offered_tasks.append(
                    task.to_task_info(
                        offer, self.addr,
                        gpu_uuids=gpu_uuids))

            driver.launchTasks(offer.id, offered_tasks, mesos_pb2.Filters())

    def _start_tf_cluster(self):
        cluster_def = {}

        targets = {}
        for task in self.tasks:
            target_name = '/job:%s/task:%s' % (task.job_name, task.task_index)
            grpc_addr = 'grpc://%s' % task.addr
            targets[target_name] = grpc_addr
            cluster_def.setdefault(task.job_name, []).append(task.addr)

        for task in self.tasks:
            response = {
                "job_name": task.job_name,
                "task_index": task.task_index,
                "cpus": task.cpus,
                "mem": task.mem,
                "gpus": task.gpus,
                "cluster_def": cluster_def,
            }
            send(task.connection, response)
            assert recv(task.connection) == "ok"
            logger.info(
                "Device /job:%s/task:%s activated @ grpc://%s ",
                task.job_name,
                task.task_index,
                task.addr

            )
            task.connection.close()
        return targets

    def start(self):

        def readable(fd):
            return bool(select.select([fd], [], [], 0.1)[0])

        lfd = socket.socket()
        try:
            lfd.bind(('', 0))
            self.addr = '%s:%s' % (socket.gethostname(), lfd.getsockname()[1])
            lfd.listen(10)
            framework = mesos_pb2.FrameworkInfo()
            framework.user = getpass.getuser()
            framework.name = self.name
            framework.hostname = socket.gethostname()
            self.driver = MesosSchedulerDriver(self, framework, self.master)
            self.driver.start()
            while any((not task.initalized for task in self.tasks)):
                if readable(lfd):
                    c, _ = lfd.accept()
                    if readable(c):
                        mesos_task_id, addr = recv(c)
                        assert isinstance(mesos_task_id, int)
                        task = self.tasks[mesos_task_id]
                        task.addr = addr
                        task.connection = c
                        task.initalized = True
                    else:
                        c.close()
            return self._start_tf_cluster()
        except Exception:
            self.stop()
            raise
        finally:
            lfd.close()

    def registered(self, driver, framework_id, master_info):
        logger.info(
            "Tensorflow cluster registered. "
            "( http://%s:%s/#/frameworks/%s )",
            master_info.hostname, master_info.port, framework_id.value
        )

    def statusUpdate(self, driver, update):
        mesos_task_id = int(update.task_id.value)
        if update.state != mesos_pb2.TASK_RUNNING:
            task = self.tasks[mesos_task_id]
            if self.started:
                logger.error("Task failed: %s, %s", task, update.message)
                _raise(RuntimeError(
                    'Task %s failed! %s' % (id, update.message)))
            else:
                logger.warn("Task failed: %s, %s", task, update.message)
                if task.connection:
                    task.connection.close()

                driver.reviveOffers()

    def slaveLost(self, driver, slaveId):
        if self.started:
            logger.error("Slave %s lost:", slaveId.value)
            _raise(RuntimeError('Slave %s lost' % slaveId))

    def executorLost(self, driver, executorId, slaveId, status):
        if self.started:
            logger.error("Executor %s lost:", executorId.value)
            _raise(RuntimeError('Executor %s@%s lost' % (executorId, slaveId)))

    def error(self, driver, message):
        logger.error("Mesos error: %s", message)
        _raise(RuntimeError('Error ' + message))

    def stop(self):
        logger.debug("exit")

        if hasattr(self, "tasks"):
            for task in getattr(self, "tasks", []):
                if task.connection:
                    task.connection.close()

            del self.tasks

        if hasattr(self, "driver"):
            self.driver.stop()
            del self.driver
