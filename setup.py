from setuptools import setup, find_packages

dependency_links = []
with open("requirements.txt") as fhandler:
    for line in fhandler:
        line = line.strip()
        _E_ = "-e "
        if line.startswith(_E_):
            line = line[3:]
        dependency_links.append(line)

version = '0.0.1'
setup(
    name='tfmesos',
    packages=find_packages(),
    install_requires=[
        'pymesos',
    ],
    dependency_links=dependency_links,
)
