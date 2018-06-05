Directional Sinogram Inpainting for Limited Angle Tomography
Contact: Robert Tovey (rt446@cam.ac.uk)
Other Authors: Martin Benning, Carola-Bibiane Schoenlieb, Sean M. Collins,
Rowan K. Leary, Paul A. Midgley, Christoph Brune, Marinus J. Lagerwerf.

============================================================

This repository contains the code required to generate reconstructions 
for the figures in the relevant paper (see paper.pdf). To run with alternative
phantoms/settings in Exp4/5 simply uncomment/alter parameters in the first 
section of the script file.

This repository comes with the AIR Tools package (Per Christian Hansen, 
https://www.sciencedirect.com/science/article/pii/S0377042711005188) for 
convenience which provides the SIRT algorithm implementation. 

The copyright for all work within this repository (minus the AIR Tools package)
is the Creative Commons Attribution 4.0 International (CC BY 4.0) license.

Requirements are MATLAB (tested on 2016b) with CVX addon (http://cvxr.com/cvx/)
and the astra toolbox (https://www.astra-toolbox.com/). We advise using the 
Mosek solver for CVX, free academic licenses are available 
(http://cvxr.com/cvx/doc/mosek.html).

Included scripts are:
Exp1: Generates Figure 7
Exp2: Generates both plots of Figure 4
Exp3: Generates Figure 3
Exp4: Generates Figures 10 and 11. Figure 5 is a subplot of Figure 11
Exp5: Generates Figures 13 and 14
