import os
import re
import glob
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
    author="Zhongbo Tian",
    author_email="tianzhongbo@douban.com",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: Implementation :: CPython",
        "Operating System :: POSIX :: Linux",
        "Topic :: Software Development :: Libraries"
    ],
    install_requires=[
        'six',
        'addict',
        'tensorflow>=0.8.0',
        'pymesos>=0.2.2',
    ],
    scripts=glob.glob(os.path.join('script', '*')),
)
