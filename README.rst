TFMesos 
========

.. image:: https://badges.gitter.im/douban/tfmesos.svg
   :alt: Join the chat at https://gitter.im/douban/tfmesos
   :target: https://gitter.im/douban/tfmesos?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

.. image:: https://img.shields.io/travis/douban/tfmesos.svg
    :target: https://travis-ci.org/douban/tfmesos/
.. image:: https://img.shields.io/pypi/v/tfmesos.svg
    :target: https://pypi.python.org/pypi/tfmesos
.. image:: https://img.shields.io/docker/automated/tfmesos/tfmesos.svg
    :target: https://hub.docker.com/r/tfmesos/tfmesos/

``TFMesos`` is a lightweight framework to help running distributed `Tensorflow <https://www.tensorflow.org>`_ Machine Learning tasks on `Apache Mesos <http://mesos.apache.org>`_ within `Docker <https://www.docker.com>`_ and `Nvidia-Docker <https://github.com/NVIDIA/nvidia-docker/>`_ .

``TFMesos`` dynamically allocates resources from a ``Mesos`` cluster, builds a distributed training cluster for ``Tensorflow``, and makes different training tasks mangeed and isolated in the shared ``Mesos`` cluster with the help of ``Docker``.


Prerequisites
--------------

* For ``Mesos >= 1.0.0``:

1. ``Mesos`` Cluster (cf: `Mesos Getting Started <http://mesos.apache.org/documentation/latest/getting-started>`_). All nodes in the cluster should be reachable using their hostnames, and all nodes have identical ``/etc/passwd`` and ``/etc/group``.
  
2. Setup ``Mesos Agent`` to enable `Mesos Containerizer <http://mesos.apache.org/documentation/container-image/>`_ and `Mesos Nvidia GPU Support <https://issues.apache.org/jira/browse/MESOS-4626>`_ (optional). eg: ``mesos-agent --containerizers=mesos --image_providers=docker --isolation=filesystem/linux,docker/runtime,cgroups/devices,gpu/nvidia``
    
3. (optional) A Distributed Filesystem (eg: `MooseFS <https://moosefs.com>`_)
  
4. Ensure latest ``TFMesos`` docker image (`tfmesos/tfmesos <https://hub.docker.com/r/tfmesos/tfmesos/>`_) is pulled across the whole cluster

* For ``Mesos < 1.0.0``:

1. ``Mesos`` Cluster (cf: `Mesos Getting Started <http://mesos.apache.org/documentation/latest/getting-started>`_). All nodes in the cluster should be reachable using their hostnames, and all nodes have identical ``/etc/passwd`` and ``/etc/group``.

2. ``Docker`` (cf: `Docker Get Start Tutorial <https://docs.docker.com/engine/installation/linux/>`_)

3. ``Mesos Docker Containerizer Support`` (cf: `Mesos Docker Containerizer <http://mesos.apache.org/documentation/latest/docker-containerizer/>`_)

4. (optional) ``Nvidia-docker`` installation (cf: `Nvidia-docker installation <https://github.com/NVIDIA/nvidia-docker/wiki/Installation>`_) and make sure nvidia-plugin is accessible from remote host (with ``-l 0.0.0.0:3476``)

5. (optional) A Distributed Filesystem (eg: `MooseFS <https://moosefs.com>`_)

6. Ensure latest ``TFMesos`` docker image (`tfmesos/tfmesos <https://hub.docker.com/r/tfmesos/tfmesos/>`_) is pulled across the whole cluster

If you are using ``AWS G2`` instance, here is a `sample <https://github.com/douban/tfmesos/blob/master/misc/setup-aws-g2.sh>`_ script to setup most of there prerequisites.


Running simple Test
------------------------
After setting up the mesos and pulling the docker image on a single node (or a cluser), you should be able to use the following command to run a simple test.

.. code:: bash

    $ docker run -e MESOS_MASTER=mesos-master:5050 \
        -e DOCKER_IMAGE=tfmesos/tfmesos \
        --net=host \
        -v /path-to-your-tfmesos-code/tfmesos/examples/plus.py:/tmp/plus.py \
        --rm \
        -it \
        tfmesos/tfmesos \
        python /tmp/plus.py mesos-master:5050

Successfully running the test should result in an output of 42 on the console.


Running in replica mode
------------------------
This mode is called `Between-graph replication` in official `Distributed Tensorflow Howto <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/how_tos/distributed/index.md#replicated-training>`_

Most distributed training models that Google has open sourced (such as `mnist_replica <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dist_test/python/mnist_replica.py>`_ and `inception <https://github.com/tensorflow/models/blob/master/inception/inception/inception_distributed_train.py>`_) are using this mode. In this mode, two kind of Jobs are defined with the names `'ps'` and `'wocker'`. `'ps'` tasks act as `'Parameter Server'` and `'worker'` tasks run the actual training process.

Here we use our modified `'mnist_replica' <https://github.com/douban/tfmesos/blob/master/examples/mnist/mnist_replica.py>`_ as example:

1. Checkout the `mnist` example codes into a directory in shared filesystem, eg: `/nfs/mnist`
2. Assume Mesos master is `mesos-master:5050`
3. Now we can launch this script using following commands:

CPU:

.. code:: bash

    $ docker run --rm -it -e MESOS_MASTER=mesos-master:5050 \
                 --net=host \
                 -v /nfs/mnist:/nfs/mnist \
                 -v /etc/passwd:/etc/passwd:ro \
                 -v /etc/group:/etc/group:ro \
                 -u `id -u` \
                 -w /nfs/mnist \
                 tfmesos/tfmesos \
                 tfrun -w 1 -s 1  \
                 -V /nfs/mnist:/nfs/mnist \
                 -- python mnist_replica.py \
                 --ps_hosts {ps_hosts} --worker_hosts {worker_hosts} \
                 --job_name {job_name} --worker_index {task_index}

GPU (1 GPU per worker):

.. code:: bash

    $ nvidia-docker run --rm -it -e MESOS_MASTER=mesos-master:5050 \
                 --net=host \
                 -v /nfs/mnist:/nfs/mnist \
                 -v /etc/passwd:/etc/passwd:ro \
                 -v /etc/group:/etc/group:ro \
                 -u `id -u` \
                 -w /nfs/mnist \
                 tfmesos/tfmesos \
                 tfrun -w 1 -s 1 -Gw 1 -- python mnist_replica.py \
                 --ps_hosts {ps_hosts} --worker_hosts {worker_hosts} \
                 --job_name {job_name} --worker_index {task_index}


Note:

In this mode, `tfrun` is used to prepare the cluster and launch the training script on each node, and worker #0 (the chief worker) will be launched in the local container.
`tfrun` will substitute `{ps_hosts}`, `{worker_hosts}`, `{job_name}`, `{task_index}` with corresponding values of each task.


Running in fine-grained mode
-----------------------------

This mode is called `In-graph replication` in official `Distributed Tensorflow Howto <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/how_tos/distributed/index.md#replicated-training>`_

In this mode, we have more control over the cluster spec. All nodes in the cluster is remote and just running a `Grpc` server. Each worker is driven by a local thread to run the training task.

Here we use our modified `mnist <https://github.com/douban/tfmesos/blob/master/examples/mnist/mnist.py>`_ as example:

1. Checkout the `mnist` example codes into a directory, eg: `/tmp/mnist`
2. Assume Mesos master is `mesos-master:5050`
3. Now we can launch this script using following commands:

CPU:

.. code:: bash

    $ docker run --rm -it -e MESOS_MASTER=mesos-master:5050 \
                 --net=host \
                 -v /tmp/mnist:/tmp/mnist \
                 -v /etc/passwd:/etc/passwd:ro \
                 -v /etc/group:/etc/group:ro \
                 -u `id -u` \
                 -w /tmp/mnist \
                 tfmesos/tfmesos \
                 python mnist.py 

GPU (1 GPU per worker):

.. code:: bash

    $ nvidia-docker run --rm -it -e MESOS_MASTER=mesos-master:5050 \
                 --net=host \
                 -v /tmp/mnist:/tmp/mnist \
                 -v /etc/passwd:/etc/passwd:ro \
                 -v /etc/group:/etc/group:ro \
                 -u `id -u` \
                 -w /tmp/mnist \
                 tfmesos/tfmesos \
                 python mnist.py --worker-gpus 1
