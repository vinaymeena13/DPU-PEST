import os
import numpy as np
import warnings
from typing import Dict, Union, Any
import pandas as pd
from K_WS_Calculator import SoilWaterPartitionCalculator
# NEW IMPORTS FOR TIME-VARYING TEMP
from K_WS_Calculator import load_time_varying_data, get_value

NumericType = Union[float, int, np.ndarray]

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
variable_name = ['k_{dF}', 'I_{mF1}', 'I_{mF2}', 'psi']
values_dict = dict(zip(variable_name, values))
inputs = Inputs_1.format(**values_dict)
inputs_time = Inputs_2.format(**values_dict)

# Open the Selector.in file in write mode
with open(os.path.join(current_dir, 'inputs.csv'), 'w') as f1, open(os.path.join(current_dir, 'time_varying_data.csv'), 'w') as f2:
    # Write the output to the file
    f1.write(inputs)
    f2.write(inputs_time)


def load_inputs(csv_file: str) -> Dict:
    """Load input parameters from CSV file into nested dictionary."""
    df = pd.read_csv(csv_file)
    inputs = {}
    for _, row in df.iterrows():
        keys = row['Parameter'].split('.')
        try:
            value = eval(row['Value'])
        except:
            value = row['Value']
        d = inputs
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        d[keys[-1]] = value
    return inputs


def save_results_to_excel(results: Dict[str, Any], output_file: str):
    """Save results to Excel in Parameter-Value format."""
    data = []
    compartments = ['root', 'stem', 'leaf', 'fruit']
    membranes = ['C', 'Vac', 'Xyl', 'Phl']

    # Helper to convert numeric values (including NumPy types) to Python built-in float
    # Leaves strings and non-convertible types unchanged
    def format_value(v):
        try:
            return float(v)
        except (TypeError, ValueError, OverflowError):
            return v

    # Scalar values
    scalar_keys = ['P_n', 'P_d', 'logK_ow_d', 'gamma_n', 'gamma_d', 'gamma_n_xyl',
                   'gamma_d_xyl', 'P_R_chem', 'lambda', 'F', 'compound_type']
    for key in scalar_keys:
        if key in results:
            data.append({'Parameter': key, 'Value': results[key]})

    # Electric potential terms (only for ionic)
    potential_keys = ['N_CW', 'N_VacC', 'N_XylC', 'N_PhC']
    for key in potential_keys:
        if key in results:
            data.append({'Parameter': key, 'Value': results[key]})

    # Compartment-specific values
    dict_keys = ['K_n', 'K_d', 'K_CW', 'K_VacC', 'K_XylC', 'K_PhC',
                 'K_XylW', 'K_PhW', 'K_VacW', 'K_PA', 'K_jW_calculated']
    for key in dict_keys:
        if key in results and isinstance(results[key], dict):
            for comp in compartments:
                if comp in results[key]:
                    data.append({'Parameter': f"{key}_{comp}", 'Value': results[key][comp]})

    # Transport ratios
    transport_keys = ['K_RXyl', 'K_StXyl', 'K_Lph', 'K_Stph', 'K_Rph']
    for key in transport_keys:
        if key in results:
            data.append({'Parameter': key, 'Value': results[key]})

    # Soil fractions
    if 'f_ns' in results:
        data.append({'Parameter': 'f_ns', 'Value': results['f_ns']})
    if 'f_ds' in results:
        data.append({'Parameter': 'f_ds', 'Value': results['f_ds']})

    # Intermediate values
    intermediate_keys = ['J_n_soil', 'J_d_soil', 'pH_soil', 'pKa_water', 'K_HSA_global_L_per_kg', 'I_xyl']
    for key in intermediate_keys:
        if key in results:
            data.append({'Parameter': key, 'Value': results[key]})

    # f_n, f_d, J_n, J_d only for ionic
    if 'f_n' in results:
        for comp in compartments:
            for mem in membranes:
                if mem in results['f_n'][comp]:
                    data.append({'Parameter': f"f_n_{comp}_{mem}", 'Value': results['f_n'][comp][mem]})
    if 'f_d' in results:
        for comp in compartments:
            for mem in membranes:
                if mem in results['f_d'][comp]:
                    data.append({'Parameter': f"f_d_{comp}_{mem}", 'Value': results['f_d'][comp][mem]})
    if 'J_n' in results:
        for comp in compartments:
            for mem in membranes:
                if mem in results['J_n'][comp]:
                    data.append({'Parameter': f"J_n_{comp}_{mem}", 'Value': results['J_n'][comp][mem]})
    if 'J_d' in results:
        for comp in compartments:
            for mem in membranes:
                if mem in results['J_d'][comp]:
                    data.append({'Parameter': f"J_d_{comp}_{mem}", 'Value': results['J_d'][comp][mem]})

    # === NEW: Apply formatting to all values just before creating the DataFrame ===
    for item in data:
        item['Value'] = format_value(item['Value'])

    df_new = pd.DataFrame(data)
    df_new.to_excel(output_file, index=False)
    print(f"\nResults saved to: {output_file}")


def compute_all(
    logK_ow: NumericType,
    pKa: NumericType,
    z: int,
    i_sign: int,
    W_dict: Dict[str, NumericType],
    L_dict: Dict[str, NumericType],
    K_AW: NumericType,
    pH_values: Dict[str, NumericType],
    pH_soil: NumericType,
    pKa_water: NumericType,
    K_HSA: NumericType,
    Pr_dict: Dict[str, NumericType],
    a: NumericType,
    b_Roots: NumericType,        # NEW: separate for roots
    b_Shoots: NumericType,       # NEW: separate for stem/leaf/fruit
    I: NumericType,
    E_dict: Dict[str, NumericType],
    P_wall: NumericType,
    P_R_water: NumericType,
    k_setchenov: NumericType,
    A: NumericType,
    csv_file: str,
    current_time: float,
    time_varying_file: str,
    return_intermediate: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Main computation function - supports both neutral and ionic compounds.
    K_jW is calculated internally: K_jW = W + L * a * K_ow^b  (now with compartment-specific b)
    """
    compartments = ['root', 'stem', 'leaf', 'fruit']
    membranes = ['C', 'Vac', 'Xyl', 'Phl']
    shoots = ['stem', 'leaf', 'fruit']

    # === STRICT: Temperature MUST be loaded from time-varying file (NO FALLBACK) ===
    if time_varying_file is None:
        raise ValueError("time_varying_file is REQUIRED: temperature must be loaded from the time-varying data file.")
    if current_time is None:
        raise ValueError("current_time is REQUIRED: it is needed to extract the correct temperature for this time step.")

    tv_data = load_time_varying_data(time_varying_file)
    Temp = get_value(tv_data, current_time, 'Temp')

    # === END STRICT SECTION ===

    # Detect compound type
    is_neutral = (pKa_water == 0)
##    print("\n" + "="*70)
##    if is_neutral:
##        print("NEUTRAL COMPOUND (pka = 0) → Simplified model")
##    else:
##        print(f"IONIC COMPOUND (z = {z}, i_sign = {i_sign}) → Full ionizable model")
##    print("="*70 + "\n")

    # Convert to numpy
    logK_ow = np.asarray(logK_ow)
    pKa = np.asarray(pKa)
    I = np.asarray(I)
    Temp = np.asarray(Temp)

    R, F = 8.314, 96484.56

    # NEW: compartment-specific b
    b_dict = {'root': b_Roots}
    for comp in shoots:
        b_dict[comp] = b_Shoots


    # Permeabilities (remain global, as in original code)
    P_n = 1.0 / (1.0 / (10**(logK_ow - 6.7)) + 1.0 / P_wall)

    if is_neutral:
        P_d = P_n
        logK_ow_d = logK_ow
        gamma_n = gamma_d = gamma_n_xyl = gamma_d_xyl = 1.0
        N_CW = N_VacC = N_XylC = N_PhC = 0.0
        f_ns, f_ds = 1.0, 0
        J_n_soil, J_d_soil = 1.0, 0
        P_R_chem = P_n
        # Calculate K_jW for neutral compounds
        K_jW_calc = {}
        for comp in compartments:
            val = W_dict[comp] + L_dict[comp] * a * (10 ** logK_ow) ** b_dict[comp]
            K_jW_calc[comp] = val
##            if verbose:
##                print(f"K_jW[{comp}] = {float(val):.6f} (W + L·a·K_ow^b) with b = {b_dict[comp]}")
    else:
        logK_ow_d = logK_ow
        P_d = 1.0 / (1.0 / (10**(logK_ow_d - 6.7)) + 1.0 / P_wall)

        # Activity coefficients
        sqrt_I = np.sqrt(I)
        log_gamma_d = -A * z**2 * ((sqrt_I / (1.0 + sqrt_I)) - 0.3 * I)
        gamma_d = 10**log_gamma_d
        gamma_n = 10**(k_setchenov * I)

        I_xyl = 0.01
        sqrt_I_xyl = np.sqrt(I_xyl)
        log_gamma_d_xyl = -A * z**2 * ((sqrt_I_xyl / (1.0 + sqrt_I_xyl)) - 0.3 * I_xyl)
        gamma_d_xyl = 10**log_gamma_d_xyl
        gamma_n_xyl = 10**(k_setchenov * I_xyl)

        # Electric potentials
        N_CW   = z * E_dict['C']   * F / (R * (273.15 + Temp))
        N_VacC = z * E_dict['Vac'] * F / (R * (273.15 + Temp))
        N_XylC = z * E_dict['Xyl'] * F / (R * (273.15 + Temp))
        N_PhC  = z * E_dict['Phl'] * F / (R * (273.15 + Temp))

        # Soil speciation
        inputs = load_inputs(csv_file)
        kws_calc = SoilWaterPartitionCalculator(inputs, time_varying_file)
        kws_res = kws_calc.calculate_K_WS(current_time)
        f_ns = kws_res.get('f_ns', 1.0)
        f_ds = kws_res.get('f_ds', 0.0)

        J_n_soil = 1.0 / (1.0 + 10**(i_sign * (pH_soil - pKa_water)))
        J_d_soil = 1.0 - J_n_soil
        P_R_chem = J_n_soil * P_n + J_d_soil * P_d

        # Import K_jW directly from inputs.csv for ionic compounds
        try:
            ionic_kjw_dict = inputs['growth_params']['K_jW_ionic']
            K_jW_calc = {}
            for comp in compartments:
                if comp not in ionic_kjw_dict:
                    raise KeyError(comp)
                val = ionic_kjw_dict[comp]
                K_jW_calc[comp] = float(val)
##                if verbose:
##                    print(f"K_jW[{comp}] = {float(val):.6f} (imported from inputs.csv for ionic compound)")
        except KeyError as ke:
            raise ValueError(
                f"For ionic compounds (pKa ≠ 0), you must provide all four parameters in inputs.csv: "
                f"growth_params.K_jW_ionic.root, .stem, .leaf, .fruit. Missing or undefined: {ke}"
            )

    # K_n and K_d (now using compartment-specific b)
    K_n = {}
    K_d = {}
    for comp in compartments:
        lipid_n = L_dict[comp] * a * ((10**logK_ow)**0.85)
        lipid_d = L_dict[comp] * a * ((10**logK_ow_d)**0.85) if not is_neutral else lipid_n
        protein = Pr_dict[comp] * K_HSA
        K_n[comp] = lipid_n + protein
        K_d[comp] = lipid_d + protein if not is_neutral else K_n[comp]

    # Fractions f_n, f_d (only needed for ionic)
    if is_neutral:
        f_n = {c: {m: 1.0 for m in membranes} for c in compartments}
        f_d = {c: {m: 0.0 for m in membranes} for c in compartments}
        J_n_dict = {c: {m: 1.0 for m in membranes} for c in compartments}
        J_d_dict = {c: {m: 0.0 for m in membranes} for c in compartments}
    else:
        J_n_dict, J_d_dict, f_n, f_d = {}, {}, {}, {}
        for comp in compartments:
            J_n_dict[comp] = {}
            J_d_dict[comp] = {}
            f_n[comp] = {}
            f_d[comp] = {}
            W_j = W_dict[comp]

            for mem in membranes:
                pH_m = pH_values[mem]
                gamma_n_mem = gamma_n_xyl if mem == 'Xyl' else gamma_n
                gamma_d_mem = gamma_d_xyl if mem == 'Xyl' else gamma_d

                if mem in ['Xyl', 'Phl']:
                    W_use = 1.0
                    Kn_use = Kd_use = 0.0
                elif mem in ['C']:
                    W_use = 0.87
                    L_use = 0.02
                    Kn_use, Kd_use = L_use * a * (10**logK_ow)**0.85 + protein, L_use * a * (10**logK_ow_d)**0.85 + protein
                else:
                    W_use = 0.87
                    L_use = 0.02
                    Kn_use, Kd_use = L_use * a * (10**logK_ow)**0.85 + protein, L_use * a * (10**logK_ow_d)**0.85 + protein

                exp_term = 10**(i_sign * (pH_m - pKa))
                Jn = 1.0 / (1.0 + exp_term)
                J_n_dict[comp][mem] = Jn
                J_d_dict[comp][mem] = 1.0 - Jn

                denom = (W_use / gamma_n_mem + Kn_use / gamma_n_mem +
                         (exp_term * W_use / gamma_d_mem) + (exp_term * Kd_use / gamma_d_mem))
                f_n[comp][mem] = 1.0 / denom
                f_d[comp][mem] = f_n[comp][mem] * exp_term

    # Partition coefficients
    def calculate_K_ab(comp, mem_a, mem_b, N, use_soil=False):
        N = np.asarray(N)
        if abs(N) < 1e-10:
            boltz = 1.0
        else:
            expN = np.exp(N)
            boltz = N / (expN - 1.0)
            boltz = np.where(abs(expN - 1) < 1e-12, 1.0, boltz)

        if use_soil and mem_b == 'W':
            fn_b, fd_b = f_ns, f_ds
        
        else:
            fn_b = f_n[comp].get(mem_b, 1.0)
            fd_b = f_d[comp].get(mem_b, 0.0)

        fn_a = f_n[comp].get(mem_a, 1.0)
        fd_a = f_d[comp].get(mem_a, 0.0)

        num = fn_b * P_n + fd_b * P_d * boltz
        den = fn_a * P_n + fd_a * P_d * np.exp(N) * boltz
        return np.where(den == 0, 1e-12, num / den)

    K_CW   = {c: calculate_K_ab(c, 'C',   'W', N_CW,   True)  for c in compartments}
    K_VacC = {c: calculate_K_ab(c, 'Vac', 'C', N_VacC)      for c in compartments}
    K_XylC = {c: calculate_K_ab(c, 'Xyl', 'C', N_XylC)      for c in compartments}
    K_PhC  = {c: calculate_K_ab(c, 'Phl', 'C', N_PhC)       for c in compartments}

    K_XylW = {c: K_XylC[c] * K_CW[c] for c in compartments}
    K_PhW  = {c: K_PhC[c]  * K_CW[c] for c in compartments}
    K_VacW = {c: K_VacC[c] * K_CW[c] for c in compartments}

    # Transport ratios and K_PA
    safe_div = lambda x, y: x/y if y != 0 else 1e-12
    K_RXyl = safe_div(K_jW_calc['root'], K_XylW['root'])
    K_StXyl = safe_div(K_jW_calc['stem'], K_XylW['stem'])
    K_Lph = safe_div(K_jW_calc['leaf'], K_PhW['leaf'])
    K_Stph = safe_div(K_jW_calc['stem'], K_PhW['stem'])
    K_Rph = safe_div(K_jW_calc['root'], K_PhW['root'])
    K_PA = {c: K_jW_calc[c] / K_AW for c in compartments}

    if is_neutral:
       lambda_val = 1
##       K_RXyl = K_jW_calc['root']  # Explicit for neutral
##       K_StXyl = K_jW_calc['stem']  # Explicit for neutral
       
    else:
        lambda_val = np.minimum(1.0, P_R_chem / P_R_water)

    # Final results
    results = {
        'P_n': float(P_n), 'P_d': float(P_d), 'logK_ow_d': float(logK_ow_d),
        'K_n': K_n, 'K_d': K_d,
        'K_CW': K_CW, 'K_VacC': K_VacC, 'K_XylC': K_XylC, 'K_PhC': K_PhC,
        'K_XylW': K_XylW, 'K_PhW': K_PhW, 'K_VacW': K_VacW,
        'K_RXyl': float(K_RXyl), 'K_StXyl': float(K_StXyl),
        'K_Lph': float(K_Lph), 'K_Stph': float(K_Stph), 'K_Rph': float(K_Rph),
        'K_PA': K_PA, 'K_jW_calculated': K_jW_calc,
        'P_R_chem': float(P_R_chem), 'lambda': float(lambda_val), 'F': F,
        'f_ns': float(f_ns), 'f_ds': float(f_ds),
        'compound_type': 'neutral' if is_neutral else 'ionic'
    }

    if return_intermediate:
        extra = {
            'J_n_soil': float(J_n_soil), 'J_d_soil': float(J_d_soil),
            'pH_soil': float(pH_soil), 'pKa_water': float(pKa_water),
            'K_HSA_global_L_per_kg': float(K_HSA),
            'N_CW': float(N_CW), 'N_VacC': float(N_VacC),
            'N_XylC': float(N_XylC), 'N_PhC': float(N_PhC),
        }
        if not is_neutral:
            extra.update({'f_n': f_n, 'f_d': f_d, 'J_n': J_n_dict, 'J_d': J_d_dict,
                          'gamma_n': float(gamma_n), 'gamma_d': float(gamma_d),
                          'I_xyl': float(I_xyl)})
        results.update(extra)

    return results


def validate_excel_inputs(inputs: Dict) -> None:
    required = ['chemical_properties', 'partition_params', 'soil_params']
    for cat in required:
        if cat not in inputs:
            raise ValueError(f"Missing category: {cat}")

    chem = inputs['chemical_properties']
    needed = ['logK_ow', 'pKa', 'z', 'i_sign']
    for p in needed:
        if p not in chem:
            raise ValueError(f"Missing chemical_properties.{p}")

    part = inputs['partition_params']
    needed = ['pH_values', 'K_HSA', 'Pr_dict', 'a', 'b_Roots', 'b_Shoots', 'I',
              'E_dict', 'P_wall', 'P_R_water', 'k_setchenov', 'A']
    for p in needed:
        if p not in part:
            raise ValueError(f"Missing partition_params.{p}")

    if 'pH_soil' not in inputs['soil_params']:
        raise ValueError("Missing soil_params.pH_soil")


def run_computation_from_excel(csv_file: str, current_time: float,
                               time_varying_file: str,
                               output_excel_file: str = None,
                               return_intermediate: bool = True, verbose: bool = False):
    """
    NOTE: time_varying_file and current_time are now REQUIRED (no defaults) to enforce strict temperature loading.
    """
    inputs = load_inputs(csv_file)
    validate_excel_inputs(inputs)

    results = compute_all(
        logK_ow=inputs['chemical_properties']['logK_ow'],
        pKa=inputs['chemical_properties']['pKa'],
        z=inputs['chemical_properties']['z'],
        i_sign=inputs['chemical_properties']['i_sign'],
        W_dict=inputs['growth_params']['W_dict'],
        L_dict=inputs['growth_params']['L_dict'],
        K_AW=inputs['chemical_properties']['K_AW'],
        pH_values=inputs['partition_params']['pH_values'],
        pH_soil=inputs['soil_params']['pH_soil'],
        pKa_water=inputs['chemical_properties']['pKa_water'],
        K_HSA=inputs['partition_params']['K_HSA'],
        Pr_dict=inputs['partition_params']['Pr_dict'],
        a=inputs['partition_params']['a'],
        b_Roots=inputs['partition_params']['b_Roots'],
        b_Shoots=inputs['partition_params']['b_Shoots'],
        I=inputs['partition_params']['I'],
        E_dict=inputs['partition_params']['E_dict'],
        P_wall=inputs['partition_params']['P_wall'],
        P_R_water=inputs['partition_params']['P_R_water'],
        k_setchenov=inputs['partition_params']['k_setchenov'],
        A=inputs['partition_params']['A'],
        csv_file=csv_file,
        current_time=current_time,
        time_varying_file=time_varying_file,
        return_intermediate=return_intermediate,
        verbose=verbose
    )

    if output_excel_file:
        save_results_to_excel(results, output_excel_file)

##    print("\n" + "="*60)
##    print("CALCULATION SUCCESSFUL!")
##    print(f"Compound type: {results['compound_type'].upper()}")
##    print(f"K_jW calculated for root: {results['K_jW_calculated']['root']:.6f}")
##    print(f"K_PA[leaf]: {results['K_PA']['leaf']:.6f}")
##    print(f"K_CW[root]: {results['K_CW']['root']:.6f}")
##    print("="*60)

    return results


# === OPTIONAL BUT RECOMMENDED: Time-series runner ===
def run_partition_time_series(
    csv_file: str,
    time_varying_file: str,
    times: list[float],
    output_excel_file: str = None,
    return_intermediate: bool = True,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Runs the partition calculation for multiple time points.
    Temperature is freshly loaded from time_varying_file for each time step.
    """
    all_results = []

    for t in times:
        if verbose:
            print(f"\n=== Computing for t = {t} days ===")
        
        results = run_computation_from_excel(
            csv_file=csv_file,
            current_time=t,
            time_varying_file=time_varying_file,
            output_excel_file=None,
            return_intermediate=return_intermediate,
            verbose=verbose
        )
        
        flat_row = {'time_days': t}
        
        # Scalars
        scalar_keys = ['P_n', 'P_d', 'logK_ow_d', 'P_R_chem', 'lambda', 'F',
                       'f_ns', 'f_ds', 'compound_type']
        for k in scalar_keys:
            if k in results:
                flat_row[k] = results[k]
        
        # Compartment-specific
        dict_keys = ['K_n', 'K_d', 'K_CW', 'K_VacC', 'K_XylC', 'K_PhC',
                     'K_XylW', 'K_PhW', 'K_VacW', 'K_PA', 'K_jW_calculated']
        compartments = ['root', 'stem', 'leaf', 'fruit']
        for key in dict_keys:
            if key in results and isinstance(results[key], dict):
                for comp in compartments:
                    if comp in results[key]:
                        flat_row[f"{key}_{comp}"] = results[key][comp]
        
        # Transport ratios
        transport_keys = ['K_RXyl', 'K_StXyl', 'K_Lph', 'K_Stph', 'K_Rph']
        for k in transport_keys:
            if k in results:
                flat_row[k] = results[k]
        
        # Ionic intermediates (example selection)
        if 'N_CW' in results:
            for nkey in ['N_CW', 'N_VacC', 'N_XylC', 'N_PhC']:
                flat_row[nkey] = results[nkey]
        
        all_results.append(flat_row)
    
    df_time_series = pd.DataFrame(all_results)
    
    if output_excel_file:
        df_time_series.to_excel(output_excel_file, index=False)
        print(f"\nTime-series results saved to: {output_excel_file}")
    
    return df_time_series


if __name__ == "__main__":
    csv_file = os.path.join(current_dir, 'inputs.csv')
    time_varying_file = os.path.join(current_dir, 'time_varying_data.csv')
    output_file = os.path.join(current_dir, 'partition_results.xlsx')

    # Example single run (will fail if files/current_time missing due to strict checks)
    results = run_computation_from_excel(
        csv_file=csv_file,
        current_time=30.0,
        time_varying_file=time_varying_file,
        output_excel_file=output_file,
        return_intermediate=True,
        verbose=True
    )
