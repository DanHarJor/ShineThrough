#!/bin/sh

#Run this to create a conda enviroment for gene_ml projects and export the enviorment to a yml file
conda create -n shine_ai -c conda-forge -c pytorch -c anaconda optuna chaospy ipykernel matplotlib pytorch pandas numpy scikit-learn tqdm nb_conda_kernels f90nml xgboost mat4py -y
conda activate shine_ai
pip install GPy==1.13.1
