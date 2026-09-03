import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Get the current working directory
current_dir = os.getcwd()

# Open the input file in read mode for selector.in
with open('inputs.dat', 'r') as inp:
    # Read the values from the file and convert them to floats
    values = [float(line.strip()) for line in inp if line.strip()]

# Open the Inputs_1.in in read mode
with open(os.path.join(current_dir, 'inputs_1.csv'), 'r') as f1, open(os.path.join(current_dir, 'time_varying_data_1.csv'), 'r') as f2:
    # Read the contents of the file
    Inputs_1 = f1.read()
    Inputs_2 = f2.read()

# Replace the variables in the Inputs.in with values from the input file
# Use {variable_name} as a placeholder in Inputs.in where you want to insert the values
##variable_name = ['Iem_S1', 'Iem_S2', 'Iem_R1', 'Iem_R2', 'Iem_St1', 'Iem_St2', 'Iem_L1', 'Iem_L2', 'Iem_F1', 'Iem_F2']
variable_name = ['k_{dF}', 'I_{mF1}', 'I_{mF2}', 'psi']
values_dict = dict(zip(variable_name, values))
inputs = Inputs_1.format(**values_dict)
inputs_time = Inputs_2.format(**values_dict)

# Open the Selector.in file in write mode
with open(os.path.join(current_dir, 'inputs.csv'), 'w') as f1, open(os.path.join(current_dir, 'time_varying_data.csv'), 'w') as f2:
    # Write the output to the file
    f1.write(inputs)
    f2.write(inputs_time)
    
# Pest files generation
# Creating input.tpl and input.par
with open(os.path.join(current_dir, 'input.tpl'), 'w') as f1, open(os.path.join(current_dir, 'input.par'), 'w') as f2:
    f1.write('ptf #\n')
    f2.write('single point\n')
    for i in variable_name:
        user_in1 = input(f'{i} Est(E)/Discrete(D):')
        if user_in1.lower() == 'e':
            f1.write(f'# {i} #\n')
            f2.write(f'{i} {values_dict[i]} 1 0\n')
        else:
            f1.write(f'{values_dict[i]}\n')
        #else:
        #   print('Invalid input')
    
os.system('cmd /k "tempchek input.tpl"')
os.system('cmd /k "tempchek input.tpl inputs.dat input.par"')

# Open the experimental data in read mode
Exp_BTC = np.loadtxt(os.path.join(current_dir,'Exp.out'))
l = len(Exp_BTC)

# Creating output.ins, output.dat and running check
with open(os.path.join(current_dir, 'output.ins'), 'w') as f:
    f.write('pif #\n')
    # Write number of data of zeroth temporal moment to the file
    for i in range(l):
        f.write(f'l1 (o{i+1})26:49\n')
with open(os.path.join(current_dir, 'measure.obf'), 'w') as f:
    # Write experimental zeroth temporal moment to the file
    for i in range(l):
        f.write(f'o{i+1} {Exp_BTC[i, 1]}\n')
os.system('cmd /k "inschek output.ins"')
os.system('cmd /k "inschek output.ins output.dat"')
os.system('cmd /k "pestgen DPU_pest input.par measure.obf"')

# Edit .pst File to input and output files
# Open hydrus_pest.pst file for reading
with open('DPU_pest.pst', 'r') as f:
    # Read the contents of the file into a list of lines
    lines = f.readlines()

# Modify model command line (5th from the bottom)
lines[-5] = lines[-5].replace('model', 'PY_Code_Multi_45.py')
lines[-3] = lines[-3].replace('model.tpl', 'input.tpl')
lines[-3] = lines[-3].replace('model.inp', 'inputs.dat')
lines[-2] = lines[-2].replace('model.ins', 'output.ins')
lines[-2] = lines[-2].replace('model.out', 'output.dat')
lines[8] = lines[8].replace('30', '100')
lines[22] = lines[22].replace('-1.000000E+10', '1.000000E-10')
lines[22] = lines[22].replace('1.000000E+10', '1')
lines[20] = lines[20].replace('-1.000000E+10', '1.000000E-10')
lines[20] = lines[20].replace('1.000000E+10', '50')
lines[21] = lines[21].replace('-1.000000E+10', '1.000000E-10')
lines[21] = lines[21].replace('1.000000E+10', '50')
lines[23] = lines[23].replace('-1.000000E+10', '1.000000E-10')
lines[23] = lines[23].replace('1.000000E+10', '50')
##lines[32] = lines[32].replace('-1.000000E+10', '0.1')
##lines[32] = lines[32].replace('1.000000E+10', '10')
##lines[33] = lines[33].replace('-1.000000E+10', '0.1')
##lines[33] = lines[33].replace('1.000000E+10', '10')
##lines[34] = lines[34].replace('-1.000000E+10', '0.1')
##lines[34] = lines[34].replace('1.000000E+10', '10')
##lines[35] = lines[35].replace('-1.000000E+10', '0.1')
##lines[35] = lines[35].replace('1.000000E+10', '10')
lines[27] = lines[27].replace('1.0', '3.0')
lines[32] = lines[32].replace('1.0', '3.0')
lines[34] = lines[34].replace('1.0', '1.0')
##lines[12] = lines[12].replace('9999', '10')
##lines[27] = lines[27].replace('obsgroup', 'regul')
##lines[29] = lines[29].replace('obsgroup', 'regul')
##lines[30] = lines[30].replace('obsgroup', 'regul')
##lines.insert(26, 'obsgroup2\n')
##lines.insert(26, 'obsgroup3\n')
##lines.insert(27, 'obsgroup3\n')
##lines[2] = lines[2].replace('estimation', 'regularisation')
##lines.insert(-1, '* prior information\n')
##lines.insert(-1, 'pi1 1 * iem_S1 + 1 * iem_R1 = 100  1.0 obsgroup2\n')
##lines.insert(-1, 'pi2 1 * iem_S2 + 1 * iem_R2 = 100  1.0 obsgroup2\n')
##lines.insert(-1,'* regularisation\n')
##lines.insert(-1, '1.0000000E-10  1.0500000E-10  0.1000000\n')
##lines.insert(-1, '10.0   1.0e-10    1.0e10\n')
##lines.insert(-1, ' 1.3   1.0e-2     1\n')
##del lines[-1]   # or lines.pop() or lines = lines[:-1]

###Advanced modifications in pest control file
##lines[3] = lines[3].rsplit(' ', 1)[0] + ' 2\n'
##lines[3] = lines[3].replace('0', '2')
##lines[11] = lines[11].replace('1', '2')

# Open the file for writing
with open('DPU_pest.pst', 'w') as f:
    # Write the modified lines to the file
    f.writelines(lines)

##os.system('cmd /k "addreg1 DPU_pest_1 DPU_pest"')

os.system('cmd /k "pestchek DPU_pest"')
os.system('cmd /k "pest DPU_pest"')

# Storing data in excel sheet
Exp_BTC = np.loadtxt(os.path.join(current_dir,'Exp.out'))
Sim_BTC = np.loadtxt(os.path.join(current_dir,'output.dat'))

# Extract time and concentration columns
exp_time, exp_concentration = Exp_BTC[:, 0], Exp_BTC[:, 1]
sim_time, sim_concentration = Sim_BTC[:, 0], Sim_BTC[:, 1]

# Create a new figure
plt.figure(figsize=(10, 6))

# Plot experimental data in blue
plt.plot(exp_time, exp_concentration, label='Experimental', marker='o', color='blue', linestyle='None')

# Plot modeled data in red
plt.plot(sim_time, sim_concentration, label='Modeled_L', linestyle='-', color='red')


# Set labels and title
plt.xlabel('Time (days)')
plt.ylabel('Concentration (mmol/kg)')
plt.title('Experimental vs. Modeled Concentration')

# Add legend
plt.legend()

# Save the plot as a PNG file with 600 DPI
#plt.savefig('concentration_plot.png', dpi=600)

# Show the plot (optional)
plt.show()

# Calculate error
#d = Exp_BTC[:, 1]-Sim_BTC[:, 1]
#mse = np.mean(d**2)
#mae = np.mean(abs(d))
#rmse = np.sqrt(mse)
#r2 = 1-(sum(d**2)/sum((Exp_BTC[:, 1]-np.mean(Exp_BTC[:, 1]))**2))

# Create a DataFrame from the NumPy arrays
#df = pd.DataFrame({'Time':Exp_BTC[:, 0],'Exp_BTC': Exp_BTC[:, 1], 'Sim_BTC': Sim_BTC[:, 1],
                  # 'MSE':mse,'MAE':mae,'RMSE':rmse,'R2':r2})

# Write the DataFrame to an Excel file
#df.to_excel('output.xlsx', index=False)
    

# Execute Hydrus batch file
#os.system('cmd /k "run_hydrus "C:\Program Files\PC-Progress\HYDRUS 5.01 64-bit\Bin\H2D_Dual64.exe" return.txt"')
