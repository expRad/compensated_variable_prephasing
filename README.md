# compensated_variable_prephasing

This repository contains instructions and source code to reproduce the results presented in 

> Phantom-based gradient waveform measurements with compensated variable-prephasing: Description and application to EPI at 7T
> Scholten H, Wech T, Homolya I, Köstler H
> (currently under review)

Please cite this work if you use the content of this repository in your project. A preprint of the article is available on [arXiv](https://arxiv.org/abs/2409.07203).

The code is written in MATLAB (R2023b).

## Preparation
In order for all scripts to run faultlessly, the *Michigan image reconstruction toolbox (MIRT)* has to be downloaded from [https://web.eecs.umich.edu/~fessler/irt/fessler.tgz](https://web.eecs.umich.edu/~fessler/irt/fessler.tgz) or [http://web.eecs.umich.edu/~fessler/irt/irt](http://web.eecs.umich.edu/~fessler/irt/irt). Information about the toolbox can be found [here](https://web.eecs.umich.edu/~fessler/code/). The content of the toolbox (the folder named *irt*) needs to be placed inside the folder named *MIRT_toolbox* of this repository.
Additionally, some scripts from the *UCI image reconstruction* repository need to be downloaded from [https://github.com/jdoepfert/UCI_image_reconstruction/tree/master/NUFFT/%40NUFFT](https://github.com/jdoepfert/UCI_image_reconstruction/tree/master/NUFFT/%40NUFFT) and placed inside the folder *EPI_recon_NUFFT/@NUFFT* of this repository. It should then contain NUFFT.m, ctranspose.m, mtimes.m, times.m, and transpose.m.

Furthermore, the data used for this publication need to be downloaded from [zenodo](https://zenodo.org/doi/10.5281/zenodo.13742003).

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

## Image reconstruction

### EPI_reco_multiDelay.m

This script reconstructs the acquired EPI images in three different ways: First with a navigator-based phase correction, second with a GSTF-based trajectory correction, and third with a measurement-based trajectory correction. One can choose between the VP, CVP, and FCVP measurement of the EPI readout gradient. For the GSTF- and measurement-based trajectory correction, a delay correction is applied to compensate for dwell time differences. To find the optimal delay later on, the reconstruction is performed with multiple different delays in this script and a measure for the ghost intensity is determined for each delay, as described in the paper. The results are stored in *EPI_results*. 

### delay_optimization.m

This script determines the optimal delays for the GSTF- and measurement-based reconstructions as described in the paper.

### EPI_reco.m

This script performs five reconstructions of the acquired EPI images: 1. with the navigator-based phase correction, 2. with the GSTF-based trajectory correction, 3. with a trajectory correction based on the VP measurement of the EPI readout gradient, 4. with a trajectory correction based on the CVP measurement of the EPI readout gradient, 5. with a trajectory correction based on the FCVP measurement of the EPI readout gradient. For the last four reconstructions, the previously described additional delay correction is applied with the delays set in the script. The results are stored in *EPI_results*.

## Scripts to reproduce the figures from the paper

### plotSequenceDiagrams.m

This script reproduces Figure 1 of the paper, i.e. it plots the sequence diagrams of the proposed measurement techniques.

### compare_gradients.m

This script reproduces Figures 2, 3, and 4 of the paper. It calculates the gradient progressions from the measured raw data, and the predictions based on the gradient system transfer function (GSTF). The raw data of the gradient measurements are contained in the folder *gradient_data*. The previously measured GSTFs are contained in *GSTF_data*.

### plot_images.m

This script reproduces Figure 5 of the paper, i.e. it plots the phantom EPI images and the respective centrally sampled k-space positions. The raw data of the EPI measurement are contained in the folder *EPI_data*. The images were first reconstructed with *EPI_reco_multiDelay.m* with multiple delays for the GSTF-based and a measurement-based (VP, CVP, or FCVP) trajectory corrections. The results were stored in *EPI_results*, and *delay_optimization.m* was used to determine the optimum delays as described in the paper. Finally, the images for plotting were reconstructed with the optimum delays using *EPI_reco.m*, and the results were also stored in *EPI_results*, as described above.
