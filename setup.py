from setuptools import setup

setup(
    name='deeplab',
    version='1.0',
    description='A library for analyzing chip properties',
    packages=['deeplab', 'deeplab.model','deeplab.trainer','deeplab.utilities'],
    package_dir={'': 'src'},
)