TFMesos 
========

TFMesos is a lightweight framework to help running distributed `Tensorflow <https://www.tensorflow.org>`_ Machine Learning tasks on `Apache Mesos <http://mesos.apache.org>`_ within `Docker <https://www.docker.com>`_ and `Nvidia-Docker <https://github.com/NVIDIA/nvidia-docker/>`_ .

TFMesos dynamically allocates resources from a Mesos cluster, builds a distributed training cluster for Tensorflow, and makes different training tasks mangeed and isolated in the shared Mesos cluster with the help of Docker.


Prerequisites
--------------

1. Mesos Cluster (cf: `Mesos Getting Started <http://mesos.apache.org/documentation/latest/getting-started>`_)

2. Docker (cf: `Docker Get Start Tutorial <https://docs.docker.com/engine/installation/linux/>`_)

3. Mesos Docker Containerizer Support (cf: `Mesos Docker Containerizer <http://mesos.apache.org/documentation/latest/docker-containerizer/>`_)

4. (optional) Nvidia-docker installation (cf: `Nvidia-docker installation <https://github.com/NVIDIA/nvidia-docker/wiki/Installation>`_) and make sure nvidia-plugin is accessiable from remote host (with ``-l 0.0.0.0:3476``)

5. Ensure latest TFMesos docker image is pulled across the whole cluster

Running in replica mode
------------------------


Running in fine-grained mode
-----------------------------

Note
----
