from setuptools import setup, find_packages

setup(
    name='src',
    packages=find_packages(include=['src','src.*']),
    version='0.1.0',
)