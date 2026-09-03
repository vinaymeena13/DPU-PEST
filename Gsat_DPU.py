from DPU_run import DPU_run
from SALib import ProblemSpec
import numpy as np
import os
import pandas as pd

# Get the current working directory
current_dir = os.getcwd()

# Define the problem
sp = ProblemSpec({
    'names': ['k_{dF}', 'I_{mF1}', 'I_{mF2}', 'psi'], # Name of each parameter
    'bounds': [[0.23*0.7, 0.23*1.3],[0.72*0.7, 0.72*1.3],[1.35*0.7, 1.35*1.3], [0.05*0.7, 0.05*1.3]],
               # bounds of each parameter
    'outputs': ['Y'] # name of outputs in expected order
})


# Generate samples, evaluate the function, and analyze the results
(sp.sample_sobol(1024, calc_second_order=True))

# Save samples in a text file
np.savetxt("param_values.txt", sp.samples)

# Open the experimental data in read mode
Exp_BTC = np.loadtxt(os.path.join(current_dir,'Exp.out'))
l = len(Exp_BTC)

# Define the output matrix
Y = np.zeros([sp.samples.shape[0]])
conc_only = np.zeros([sp.samples.shape[0],l])

for i, X in enumerate(sp.samples[:]):
    output = DPU_run(X)
    np.savetxt(f"output_{i}.txt", output)
    print(f"{i}")
    
all_output = []
for i in range(sp.samples.shape[0]):
    data = np.loadtxt(f"output_{i}.txt")
    value = data[:, 1].astype(float)
    all_output.append(value)

# Convert the result list to a NumPy array
result = np.array(all_output)

# Transpose the result array to get the desired shape
result = result.T

#print(all_output)
np.savetxt(f"conc_only.txt", result)

# Create empty lists to store the values
total_Si_list = []
first_Si_list = []
second_Si_list = []

# Provide the results to the interface
for i in range(result.shape[0]):
    Y = result[i,:]
    # print(f"Y.shape: {Y.shape}")
    # print(f"sp.samples.shape: {sp.samples.shape}")
    sp.set_results(Y)
    sp.analyze_sobol()
    total_Si, first_Si, second_Si = sp.to_df()

    # Transpose the total_Si DataFrame
    transposed_total_Si = total_Si.T
    transposed_first_Si = first_Si.T

    # Select the desired column (e.g., the first column with index 0)
    selected_total_Si = pd.DataFrame(transposed_total_Si.drop(transposed_total_Si.index[1], axis=0))
    selected_first_Si = pd.DataFrame(transposed_first_Si.drop(transposed_first_Si.index[1], axis=0))

    # Append the values of the selected column to the total_Si_list list
    total_Si_list.append(selected_total_Si)
    first_Si_list.append(selected_first_Si)
    
    # Concatenate all the DataFrames in the list
Total_SI = pd.concat(total_Si_list, ignore_index=True)
time_series = pd.DataFrame(Exp_BTC[:,0])
Total_SI = pd.concat([time_series, Total_SI], axis=1)
# Save the DataFrame to an Excel file
Total_SI.to_excel('Total_SI.xlsx', index=False)

    # Concatenate all the DataFrames in the list
First_SI = pd.concat(first_Si_list, ignore_index=True)
First_SI = pd.concat([time_series, First_SI], axis=1)
# Save the DataFrame to an Excel file
First_SI.to_excel('First_SI.xlsx', index=False)

