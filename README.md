This repository contains all the necessary modules used to develop a multi-compartment plant uptake model and further link it with the Global Sensitivity Analysis and parameter estimation algorithms

The description of the provided scripts and datasets for the dynamic plant uptake (DPU) model is as follows:

inputs.csv is the dataset for the DPU model parameters, which are constant over time
time_varying_data.csv is similar to the previous; this is the dataset for the parameters that are defined as time-varying for the model.
K_WS_Calculator.py is used to calculate the soil water partition coefficient using the provided input data.
PartitionCoeff_1.py is used to calculate the plant water partition coefficients for all the plant compartments; the algorithm is also utilised to describe the partition within the cell organelles.
PY_Code_Multi.py is the main scripts which solves the differential equation for all 5 compartments, utilizing the values from the previous scripts and storing the data in csv file.
DPU_Run.py is the same script as "PY_Code_Multi.py" except that the whole is defined as a function, which was later utilised to call the model.
Gsat_DPU.py is used to carry out the sensitivity analysis by coupling the plant uptake model with the sensitivity analysis library (SALib). This uses the DPU_Run.py function to call the model
DPU_pest_couple.py was developed to couple the parameter estimation algorithm PEST with the DPU model. This code also modifies the PEST workflow based on our requirements.
Sobol_LR.py is the algorithm for hybrid parameter optimisation coupled with DPU
