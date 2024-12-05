#!/usr/bin/env python3
"""Tide"""

from setuptools import setup, find_packages

# Get the long description from the README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="tide",
    version="0.1.0",
    description="Measured data visualization and pipelines",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BuildingEnergySimulationTools/tide",
    author="Nobatek/INEF4",
    author_email="bdurandestebe@nobatek.inef4.com",
    license="License :: OSI Approved :: BSD License",
    # keywords=[
    # ],
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.22.4, <2.0",
        "pandas>=2.0.0, <3.0",
        "scipy>=1.9.1, <2.0",
        "bigtree>=0.21.3",
        "scikit-learn>=1.2.2, <2.0",
        "statsmodels>=0.14.4",
        "matplotlib>=3.5.1",
        "plotly>=5.3.1",
        "requests>=2.32.3",
    ],
    packages=find_packages(exclude=["tests*"]),
    include_package_data=True,
)
