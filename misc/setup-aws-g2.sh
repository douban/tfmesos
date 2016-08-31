#!/bin/bash
set -e -x

DEBIAN_FRONTEND=noninteractive
CODENAME=$(lsb_release -cs)
VERSION=$(lsb_release -rs)
MACHINE=$(uname -m)
ARCH=$(dpkg --print-architecture)
CUDA_VERSION=7.5-18
DOCKER_VERSION=1.10.3-0~${CODENAME}
MESOS_VERSION=0.27.2-2.0.15.ubuntu${VERSION/./}
ND_VERSION_1=1.0.0
ND_VERSION_2=rc
ND_VERSION_3=3
ND_VERSION_4=1

apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/GPGKEY
apt-key adv --keyserver keyserver.ubuntu.com --recv E56151BF
echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu${VERSION/./}/${MACHINE} /"|tee /etc/apt/sources.list.d/cuda.list
echo "deb http://repos.mesosphere.com/ubuntu ${CODENAME} main"|tee /etc/apt/sources.list.d/mesosphere.list

curl -sSL https://get.docker.com/ | sh

apt-get update
apt-get dist-upgrade -y --force-yes
apt-get install -y --force-yes linux-image-extra-virtual cuda=${CUDA_VERSION} docker-engine=${DOCKER_VERSION} nvidia-modprobe mesos=${MESOS_VERSION} python
apt-mark hold cuda
apt-mark hold docker-engine
apt-mark hold mesos

ND_PKG=nvidia-docker_${ND_VERSION_1}.${ND_VERSION_2}.${ND_VERSION_3}-${ND_VERSION_4}_${ARCH}.deb
wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v${ND_VERSION_1}-${ND_VERSION_2}.${ND_VERSION_3}/${ND_PKG}
dpkg -i /tmp/${ND_PKG}
echo 'NVIDIA_DOCKER_PLUGIN_OPTS="-s /var/lib/nvidia-docker -l 0.0.0.0:3476"' |tee /etc/default/nvidia-docker
ln -s /lib/init/upstart-job /etc/init.d/nvidia-docker
service nvidia-docker start
echo `hostname`| tee /etc/mesos-slave/hostname
echo "docker,mesos" | tee /etc/mesos-slave/containerizers
python - << '__EOF'
from __future__ import print_function
import os
import urllib2

args = urllib2.urlopen('http://localhost:3476/mesos/cli').read()
def set_conf(keys, value):
    fn = os.path.join(os.sep, 'etc', 'mesos-slave', *keys)
    path = os.path.dirname(fn)
    if os.path.exists(fn):
        print('File exists: %s' % fn)
        return

    try:
        os.makedirs(path)
    except OSError:
        pass

    with open(fn, 'w+') as f:
        f.write(value)

for arg in args.split():
    key, value = arg.split('=')
    if not key.startswith('--'):
        print('Unknown key %s' % key)
        continue

    key = key[2:]
    if key not in {'attributes', 'resources'}:
        set_conf([key], value)
    else:
        for sub_arg in value.split(';'):
            sub_key, sub_value = sub_arg.split(':')
            set_conf([key, sub_key], sub_value)
__EOF