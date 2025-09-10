# SBD_Street_Furniture
This repository contains details of the implementation of the methodology introduced in the paper "A Stochastic Birth-and-Death Approach for Street Furniture Geolocation in Urban Environments", presented as part of the Irish Machine Vision and Image Processing (IMVIP) conference 2025. The conference website can be found [here](https://imvipconference.github.io/) and proceedings [here](https://pure.ulster.ac.uk/en/publications/proceedings-of-the-irish-machine-vision-and-image-processing-conf-4). This repository contains two python scripts and several example files which can be used to simulated the methodologies discussed in the paper.

### Generation of Synthetic Data
Implementation of the synthetic data generation is given in the script **synth_detect.py**, with main function *propose_detections*. This function takes as input two *.csv* file names: one containing geographic coordinates (lon,lat) of street cameras, and the other, geographic coordinates of assets (street furniture). The output of this function is a *.csv* file (default name *detections.csv*) containing coordinates of (noisy) detections, along with simulated depth estimates, confidence measures, and indiactors of contaminate (false positive) detections. Users can alter the functions default parameters to simulate varying levels of noise.  

### Stochastic Birth & Death Simulation
Implementation of the stochastic birth & death algorithm is given in the script **sbd.py**, with main function *simulation()*. This function requires the following inputs to run:
1) A pair of (lon,lat) coordinates defining the top-left and bottom-right corners of a rectangular map area of interest.
2) A static image of large infrastructure occupying the area of interest.
3) A *.csv* file containing details of pairwise intersections of camera-to-object rays. This file should have columns (in order) *d1,d2* (depth estimates), *lat,lon* (coordinates of intersection), *delta1, delta2* (camera-to-intersection distances), *CNN1, CNN2* (measures of confidence). For details of how to extract pairwise camera-to-object ray intersections, readers are referred to [Krylov et. al. (2018)](https://www.mdpi.com/2072-4292/10/5/661).

The parameters of the model can be altered by editing lines 347 - 357 of the script.
