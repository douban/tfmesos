import os
import sys
import math
import select
import socket
import getpass
import logging
import textwrap
from six import iteritems
from six.moves import urllib
from pymesos import Scheduler, MesosSchedulerDriver
from tfmesos.utils import send, recv, setup_logger


FOREVER = 0xFFFFFFFF
logger = logging.getLogger(__name__)


class Job(object):

    def __init__(self, name, num, cpus=1.0, mem=1024.0,
                 gpus=0, cmd=None, start=0):

        self.name = name
        self.num = num
        self.cpus = cpus
        self.gpus = gpus
        self.mem = mem
        self.cmd = cmd
        self.start = start


class Task(object):

    def __init__(self, mesos_task_id, job_name, task_index,
                 cpus=1.0, mem=1024.0, gpus=0, cmd=None, volumes={}):
        self.mesos_task_id = mesos_task_id
        self.job_name = job_name
        self.task_index = task_index

        self.cpus = cpus
        self.gpus = gpus
        self.mem = mem
        self.cmd = cmd
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

    def to_task_info(self, offer, master_addr, gpu_uuids=[],
                     gpu_resource_type=None):
        resources = [
            dict(
                name='cpus',
                type='SCALAR',
                scalar=dict(value=self.cpus),
            ),
            dict(
                name='mem',
                type='SCALAR',
                scalar=dict(value=self.mem),
            )
        ]

        cmd = ' '.join([
            sys.executable, '-m', '%s.server' % __package__,
            str(self.mesos_task_id), master_addr
        ])

        ti = dict(
            task_id=dict(
                value=str(self.mesos_task_id)
            ),
            agent_id=offer['agent_id'],
            name='/job:%s/task:%s' % (self.job_name, self.task_index),
            resources=resources,
            command=dict(
                shell=True,
                value=cmd,
                environment=dict(
                    variables=[
                        dict(
                            name='PYTHONPATH',
                            value=':'.join(sys.path)
                        )
                    ]
                )
            ),
        )

        image = os.environ.get('DOCKER_IMAGE')

        if image is not None:
            volumes = []
            ti.update(dict(
                container=dict(
                    type='DOCKER',
                    docker=dict(image=image),
                    volumes=volumes
                ),
            ))

            for path in ['/etc/passwd', '/etc/group']:
                volumes.append(dict(
                    host_path=path,
                    container_path=path,
                    mode='RO'
                ))

            for src, dst in iteritems(self.volumes):
                volumes.append(dict(
                    host_path=src,
                    container_path=dst,
                    mode='RW'
                ))

            if self.gpus and gpu_uuids and gpu_resource_type is not None:
                if gpu_resource_type == 'SET':
                    hostname = offer['hostname']
                    url = 'http://%s:3476/docker/cli?dev=%s' % (
                        hostname, urllib.parse.quote(
                            ' '.join(gpu_uuids)
                        )
                    )

                    try:
                        docker_args = urllib.request.urlopen(url).read()
                        for arg in docker_args.split():
                            k, v = arg.split('=')
                            assert k.startswith('--')
                            k = k[2:]
                            p = ti.container.docker.parameters.add()
                            p.key = k
                            p.value = v

                        resources.append(dict(
                            name='gpus',
                            type='SET',
                            set=dict(item=[gpu_uuids])
                        ))

                    except Exception:
                        logger.exception(
                            'fail to determine remote device parameter,'
                            ' disable gpu resources'
                        )
                else:
                    resources.append(dict(
                        name='gpus',
                        type='SCALAR',
                        scalar=dict(value=len(gpu_uuids))
                    ))

        else:
            if self.gpus and gpu_uuids and gpu_resource_type is not None:
                if gpu_resource_type == 'SET':
                    resources.append(dict(
                        name='gpus',
                        type='SET',
                        set=dict(item=[gpu_uuids])
                    ))
                else:
                    resources.append(dict(
                        name='gpus',
                        type='SCALAR',
                        scalar=len(gpu_uuids)
                    ))

        return ti


class TFMesosScheduler(Scheduler):

    def __init__(self, task_spec, master=None, name=None, quiet=False,
                 volumes={}, local_task=None):
        self.started = False
        self.master = master or os.environ['MESOS_MASTER']
        self.name = name or '[tensorflow] %s %s' % (
            os.path.abspath(sys.argv[0]), ' '.join(sys.argv[1:]))
        self.local_task = local_task
        self.task_spec = task_spec
        self.tasks = []
        for job in task_spec:
            for task_index in range(job.start, job.num):
                mesos_task_id = len(self.tasks)
                self.tasks.append(
                    Task(
                        mesos_task_id,
                        job.name,
                        task_index,
                        cpus=job.cpus,
                        mem=job.mem,
                        gpus=job.gpus,
                        cmd=job.cmd,
                        volumes=volumes
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
                driver.declineOffer(offer['id'], dict(refuse_seconds=FOREVER))
                continue

            offered_cpus = offered_mem = 0.0
            offered_gpus = []
            offered_tasks = []
            gpu_resource_type = None

            for resource in offer['resources']:
                if resource['name'] == 'cpus':
                    offered_cpus = resource['scalar']['value']
                elif resource['name'] == 'mem':
                    offered_mem = resource['scalar']['value']
                elif resource['name'] == 'gpus':
                    if resource['type'] == 'SET':
                        offered_gpus = resource['set']['item']
                    else:
                        offered_gpus = list(range(
                            int(resource['scalar']['value'])
                        ))

                    gpu_resource_type = resource['type']

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
                        offer, self.addr, gpu_uuids=gpu_uuids,
                        gpu_resource_type=gpu_resource_type
                    )
                )

            driver.launchTasks(offer['id'], offered_tasks)

    def _start_tf_cluster(self):
        cluster_def = {}

        targets = {}
        for task in self.tasks:
            target_name = '/job:%s/task:%s' % (task.job_name, task.task_index)
            grpc_addr = 'grpc://%s' % task.addr
            targets[target_name] = grpc_addr
            cluster_def.setdefault(task.job_name, []).append(task.addr)

        if self.local_task:
            job_name, addr = self.local_task
            cluster_def.setdefault(job_name, []).insert(0, addr)

        for task in self.tasks:
            response = {
                'job_name': task.job_name,
                'task_index': task.task_index,
                'cpus': task.cpus,
                'mem': task.mem,
                'gpus': task.gpus,
                'cmd': task.cmd,
                'cwd': os.getcwd(),
                'cluster_def': cluster_def,
            }
            send(task.connection, response)
            assert recv(task.connection) == 'ok'
            logger.info(
                'Device /job:%s/task:%s activated @ grpc://%s ',
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
            framework = dict(
                user=getpass.getuser(),
                name=self.name,
                hostname=socket.gethostname()
            )

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

            self.started = True
            return self._start_tf_cluster()
        except Exception:
            self.stop()
            raise
        finally:
            lfd.close()

    def registered(self, driver, framework_id, master_info):
        logger.info(
            'Tensorflow cluster registered. '
            '( http://%s:%s/#/frameworks/%s )',
            master_info['hostname'], master_info['port'], framework_id['value']
        )

    def statusUpdate(self, driver, update):
        mesos_task_id = int(update['task_id']['value'])
        if update['state'] != 'TASK_RUNNING':
            task = self.tasks[mesos_task_id]
            if self.started:
                if update['state'] != 'TASK_FINISHED':
                    logger.error(
                        'Task failed: %s, %s', task, update['message'])
                    raise RuntimeError(
                        'Task %s failed! %s' % (id, update['message']))
            else:
                logger.warn('Task failed: %s, %s', task, update['message'])
                if task.connection:
                    task.connection.close()

                driver.reviveOffers()

    def slaveLost(self, driver, agent_id):
        if self.started:
            logger.error('Slave %s lost:', agent_id['value'])
            raise RuntimeError('Slave %s lost' % agent_id['value'])

    def executorLost(self, driver, executor_id, agent_id, status):
        if self.started:
            logger.error('Executor %s lost: %s', executor_id['value'], status)
            raise RuntimeError('Executor %s@%s lost' % (
                executor_id['value'], agent_id['value']
            ))

    def error(self, driver, message):
        logger.error('Mesos error: %s', message)
        raise RuntimeError('Error ' + message)

    def stop(self):
        logger.debug('exit')

        if hasattr(self, 'tasks'):
            for task in getattr(self, 'tasks', []):
                if task.connection:
                    task.connection.close()

            del self.tasks

        if hasattr(self, 'driver'):
            self.driver.stop()
            del self.driver
