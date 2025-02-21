# PySOMFFN

## DESCRIPTION

Core file for running Self-Organising Map - Feed Forward Network (SOM-FFN) 
method based on the MATLAB implementation of Peter Landschuetzer and 
originally described in Landschuetzer et al. (2013) Biogeosciences.
                                                                         
This Python implementation is under development within the Past, Present and 
Future Marine Climate Change Group of the Flanders Marine Institute (VLIZ), 
Belgium.                       
                                                                         
This Python implementation is separated into 2 classes:
    - SelfOrganisingMap
    - FeedForwardNetwork
The functions contained in these classes are described in detail within the
code files. These class files are written with the intention to be used
together or separately. There are a range of optional functions but some
functions must be used sequentially and this is indicated.

Input and output data files are supported in MATLAB (.mat), for legacy 
support and comparison with existing SOM-FFN, and netCDF (.nc) formats.

Figure plotting is currently supported for PNG format only.

Please contact Creators regarding test data and input data requirements.

 

## DEPENDENCIES

PySOMFFN is written in and uses the Python environment and requires:
    - Python version 3.12.3

The classes and their functions are dependent on the following packages:
    - CartoPy 0.22.0
    - MatPlotLib 3.6.3
    - MiniSom 2.3.3 (https://github.com/JustGlowing/minisom)
    - NumPy 1.26.4
    - SciPy 1.11.4
    - SKlearn 1.11.4
    - TensorFlow 2.18.0 (https://www.tensorflow.org/)
    - Xarray 2024.2.0
