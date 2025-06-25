#!/usr/bin/env python

import re
import pathlib
import setuptools

# base source directory
base_dir = pathlib.Path(__file__).parent.resolve()

# extract the current version
version = 0.2

# run setup
setuptools.setup(
    name='voxelmorph',
    version=version,
    license='Apache 2.0',
    description='Image Registration with Convolutional Networks',
    url='https://github.com/voxelmorph/voxelmorph',
    keywords=['deformation', 'registration', 'imaging', 'cnn', 'mri'],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'packaging',
        'scikit-image',
        'h5py',
        'numpy<2.0',
        'scipy',
        'nibabel',
        'neurite>=0.2',
        'pandas'
    ]
)
