#!/usr/bin/env python
# coding=utf-8
"""
Author: liufr
Github: https://github.com/Fengrui-Liu
LastEditTime: 2020-11-21 11:35:35
Copyright 2020 liufr
Description:
"""

from os import path
from setuptools import setup, find_packages
from io import open

this_directory = path.abspath(path.dirname(__file__))

with open(path.join(this_directory, "requirements.txt"), encoding="utf-8") as f:
    requirements = f.read().splitlines()

with open(path.join(this_directory, "StreamAD/version.py")) as f:
    exec(f.read())


setup(
    name="StreamAD",
    version=__version__,
    description=("Python toolbox for stream anomaly (outlier) detection."),
    long_description=open("./README.md").read(),
    author="liufr",
    author_email="liufengrui18z@ict.ac.cn",
    packages=find_packages(exclude=["test"]),
    platforms=["all"],
    install_requires=requirements,
    include_package_data=True,
    url="https://github.com/Fengrui-Liu/StreamAD",
    setup_requires=["setuptools>=38.6.0"],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)