# compensated_variable_prephasing

This repository contains instructions and source code to reproduce the results presented in 

> Phantom-based gradient waveform measurements with compensated variable-prephasing: Description and application to EPI at 7T
> Scholten H, Wech T, Homolya I, Köstler H
> (currently under review)

Please cite this work if you use the content of this repository in your project.

The code is written in MATLAB (R2023b)

## Preparation
In order for all scripts to run faultless, the *Michigan image reconstruction toolbox (MIRT)* has to be downloaded from [https://web.eecs.umich.edu/~fessler/irt/fessler.tgz](https://web.eecs.umich.edu/~fessler/irt/fessler.tgz) or [http://web.eecs.umich.edu/~fessler/irt/irt](http://web.eecs.umich.edu/~fessler/irt/irt). Information about the toolbox can be found [here](https://web.eecs.umich.edu/~fessler/code/). The content of the toolbox (the folder named *irt*) needs to be placed inside the folder named *MIRT_toolbox* of this repository.
Additionally, some scripts from the *UCI image reconstruction* repository need to be downloaded from [https://github.com/jdoepfert/UCI_image_reconstruction/tree/master/NUFFT/%40NUFFT](https://github.com/jdoepfert/UCI_image_reconstruction/tree/master/NUFFT/%40NUFFT) and placed inside the folder *EPI_recon_NUFFT/@NUFFT* of this repository. It should then contain NUFFT.m, ctranspose.m, mtimes.m, times.m, and transpose.m.

Furthermore, the data used for this publication need to be downloaded from [zenodo](t.b.a.).

The repository should finally contain the following folders and scripts at the same hierarchy level as this file:
* EPI_data
* EPI_recon_NUFFT
* EPI_results
* gradient_data
* GSTF_data
* helper_functions
* MIRT_toolbox
* *compare_gradients.m*
* *delay_optimization.m*
* *EPI_reco.m*
* *EPI_reco_multiDelay.m*
* *plot_images.m*
* *plotSequenceDiagrams.m*

## Scripts to reproduce the figures from the paper

### plotSequenceDiagrams.m

This script reproduces Figure 1 of the paper, i.e. it plots the sequence diagrams of the proposed measurement techniques.

### compare_gradients.m

This script reproduces Figures 2, 3, and 4 of the paper. It calculates the gradient progressions from the measured raw data, and the predictions based on the gradient system transfer function (GSTF). The raw data of the gradient measurements are contained in the folder *gradient_data*. The previously measured GSTFs are contained in *GSTF_data*.

### plot_images.m

This script reproduces Figure 5 of the paper, i.e. it plots the phantom EPI images and the respective centrally sampled k-space positions. The raw data of the EPI measurement are contained in the folder *EPI_data*. The images were first reconstructed with *EPI_reco_multiDelay.m* with multiple delays for the GSTF-based and the CVP-based trajectory corrections. The results were stored in *EPI_results*, and *delay_optimization.m* was used to determine the optimum delays as described in the paper. Finally, the images for plotting were reconstructed with the optimum delays using *EPI_reco.m*, and the results were also stored in *EPI_results*.
