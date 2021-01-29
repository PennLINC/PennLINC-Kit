---
layout: default
title: Project Template
parent: Documentation
has_children: false
has_toc: false
nav_order: 3
---

### PennLINC-Kit

This is the toolkit of common functions used across the PennLINC labratory.

### Brief Project Description

Wondering how you can write out a cifti files with a custom colormap? Want to do some machine learning? Network science? Want to do a spin-test on parcels? These common applications can be found here

### Project Lead(s) 

Max Bertolero, the entire informatics team, and anyone who uses the code

### Faculty Lead(s)

Theodore Satterthwaite

### Analytic Replicator

The entire lab uses and validates this code

### Collaborators

Informatics Team

### Project Start Date

Ealy 2021

### Current Project Status

We are in the early stages of building this toolkit

### Dataset

All the data

### Github repo

https://github.com/PennLINC/PennLINC-Kit

### Path to data on filesystem

This will be stored in a container on PMACS and CUBIC

### Slack Channel

#informatics

### Code documentation

Python module for loading data

Object oriented, using read-only matrix loading:

import pennlinc_data

for dataset in [‘pic’,’abcd’]:

  data = pennlinc_data.load(dataset)
  
  matrices = data.matrices

  ef = data.behavior['ef']
  
  motion = data.qc['mean_FD']
  
  #analysis here

Python module to analyze data
Graph theory metrics, functions for converting a matrix to a graph
Create a connectome workbench surface
Spin test on parcels
Cross-validated prediction models

Anaconda container and defaults 
Install useful libs that we should use 
Good defaults for font, colors

Notebooks
PMACS “quirks” / SGE / multiprocessing
How to make a plot look good
How to analyze citation balance in your .bib
