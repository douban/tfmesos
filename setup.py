from setuptools import setup, find_packages

version = '0.0.1'
setup(
    name='tfmesos',
    version=version,
    packages=find_packages(),
    install_requires=[
        'tensorflow>=0.8.0',
        'pymesos',
    ],
)
