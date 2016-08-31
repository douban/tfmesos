import os
import re
import glob
from subprocess import check_output
from setuptools import setup, find_packages


def find_version(*paths):
    fname = os.path.join(*paths)
    with open(fname) as fhandler:
        version_file = fhandler.read()
        version_match = re.search(r"^__VERSION__ = ['\"]([^'\"]*)['\"]",
                                  version_file, re.M)

    if not version_match:
        raise RuntimeError("Unable to find version string in %s" % (fname,))

    version = version_match.group(1)

    try:
        command = 'git describe --tags'
        with open(os.devnull, 'w') as fnull:
            tag = check_output(
                command.split(),
                stderr=fnull).decode('utf-8').strip()

        if tag.startswith('v'):
            assert tag == 'v' + version
    except Exception:
        pass

    return version


def find_readme(*paths):
    with open(os.path.join(*paths)) as f:
        return f.read()

setup(
    name='tfmesos',
    version=find_version('tfmesos', '__init__.py'),
    packages=find_packages(),
    license='BSD License',
    description="Tensorflow on Mesos",
    long_description=find_readme('README.rst'),
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: Implementation :: CPython",
        "Operating System :: POSIX :: Linux",
        "Topic :: Software Development :: Libraries"
    ],
    install_requires=[
        'tensorflow>=0.8.0',
        'pymesos',
    ],
    scripts=glob.glob(os.path.join('script', '*')),
)
