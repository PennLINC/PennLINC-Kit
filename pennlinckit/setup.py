#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    author="Max Bertolero",
    author_email='mbertolero@me.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="all the functions we use in the PennLINC",
    install_requires=[required],
    license="MIT license",
    include_package_data=True,
    keywords='pennlinckit',
    name='pennlinckit',
    packages=find_packages(include=['pennlinckit','pennlinckit.*']),
    url='https://pennlinc.github.io/PennLINC-Kit/',
    version='0.1.0',
    zip_safe=False,
)
