import os
import numpy as np
from typing import Dict
import pandas as pd

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
        value_str = str(row['Value']).strip()

        # Convert to int or float if possible
        try:
            if '.' not in value_str and 'E' not in value_str.upper() and 'e' not in value_str:
                value = int(value_str)
            else:
                value = float(value_str)
        except ValueError:
            value = value_str  # keep as string

        # Build nested dict
        d = inputs
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        d[keys[-1]] = value

    return inputs


def load_time_varying_data(csv_file: str) -> pd.DataFrame:
    """Load time-varying data from CSV file."""
    return pd.read_csv(csv_file)


def get_value(time_data: pd.DataFrame, current_time: float, column: str) -> float:
    """
    Get exact CSV value using precision-safe matching.
    No interpolation is performed.
    """

    # Determine decimal precision from CSV time column
    sample_times = time_data['Time'].astype(str)

    max_decimals = max(
        sample_times.apply(
            lambda x: len(x.split('.')[-1]) if '.' in x else 0
        )
    )

    # Round solver time to CSV precision
    current_time_rounded = round(current_time, max_decimals)

    # Round CSV times similarly
    rounded_times = time_data['Time'].round(max_decimals)

    row = time_data[rounded_times == current_time_rounded]

    if row.empty:
        raise ValueError(
            f"No data for time {current_time_rounded} "
            f"in column '{column}'"
        )

    return float(row[column].iloc[0])


class SoilWaterPartitionCalculator:
    """
    Calculates soil-water partition coefficient (K_WS) for:
      - Neutral compounds (pKa = 0): simplified model
      - Ionic/ionizable compounds (pKa ≠ 0): full speciation model
    """

    def __init__(self, inputs: Dict, time_varying_file: str = None):
        self.inputs = inputs
        self.time_varying_file = time_varying_file

        if time_varying_file:
            self.time_varying_data = load_time_varying_data(time_varying_file)
        else:
            self.time_varying_data = None

        # Extract soil parameters
        try:
            soil = inputs['soil_params']
            self.soil_params = {
                'OC': soil['OC'],
                'rho_dry': soil['rho_dry'],
                'pH_soil': soil['pH_soil'],
                'I': soil['I'],
                'A': soil['A'],
                'k_setchenov': soil['k_setchenov']
            }
        except KeyError as e:
            raise ValueError(f"Missing soil parameter: {e}")

        # Extract chemical properties
        try:
            chem = inputs['chemical_properties']
            self.chemical_params = {
                'logK_ow': chem['logK_ow'],
                'K_AW': chem['K_AW'],
                'z': chem['z'],
                'pKa': chem['pKa'],
                'pKa_water': chem['pKa_water'],    # may be missing for neutral
                'i_sign': chem.get('i_sign', 0)  # default 0 if not provided
            }
        except KeyError as e:
            raise ValueError(f"Missing chemical property: {e}")

        # Determine compound type
        self.is_neutral = (self.chemical_params['pKa_water'] == 0)

##        print("Compound type detection:")
##        if self.is_neutral:
##            print("  → NEUTRAL compound (pKa = 0) → using simplified model")
##        else:
##            print(f"  → IONIC compound (pKa = {self.chemical_params['pKa']}) → using full ionizable model")

    # ------------------------------------------------------------------
    # K_OC calculations
    # ------------------------------------------------------------------
    def calculate_K_OC_neutral(self, logK_ow: float) -> float:
        """Neutral compound: logK_OC = 0.81 * logK_ow + 0.1"""
        logK_OC = 0.81 * logK_ow + 0.1
        return 10 ** logK_OC

    def calculate_K_OC_ionic(self, logK_ow: float) -> Dict[str, float]:
        """Ionic compound: separate K_OC for neutral and dissociated forms"""
        if self.chemical_params['z'] == 1:
            logK_OC_n = 0.42 * logK_ow + 1.34
            logK_OC_d = 0.47 * logK_ow + 1.95
        
        else:
            logK_OC_n = 0.54 * logK_ow + 1.11
            logK_OC_d = 0.11 * logK_ow + 1.54
        return {'K_OC_n': 10**logK_OC_n, 'K_OC_d': 10**logK_OC_d}

    # ------------------------------------------------------------------
    # Activity coefficients (only used for ionic case)
    # ------------------------------------------------------------------
    def calculate_activity_coefficients(self, z: int, I: float, A: float, k_setchenov: float):
        sqrt_I = np.sqrt(I)
        log_gamma_d = -A * z**2 * (sqrt_I / (1 + sqrt_I) - 0.3 * I)
        gamma_d = 10 ** log_gamma_d
        gamma_n = 10 ** (k_setchenov * I)
        return {'gamma_n': gamma_n, 'gamma_d': gamma_d}

    # ------------------------------------------------------------------
    # Main K_WS calculation
    # ------------------------------------------------------------------
    def calculate_K_WS(self, current_time: float) -> Dict:
        if self.time_varying_data is None:
            raise ValueError("Time-varying data file must be provided.")
        if current_time is None:
            raise ValueError("current_time must be specified.")

        # Common inputs
        logK_ow = self.chemical_params['logK_ow']
        K_AW = self.chemical_params['K_AW']
        OC = self.soil_params['OC']
        rho_dry = self.soil_params['rho_dry']
        P_W = get_value(self.time_varying_data, current_time, 'P_W')
        rho_wet = get_value(self.time_varying_data, current_time, 'rho_wet')
        P_A = get_value(self.time_varying_data, current_time, 'P_A')

        result = {
            'Time': current_time,
            'logK_ow': logK_ow,
            'P_W_used': P_W,
            'rho_wet_used': rho_wet,
            'P_A_used': P_A,
            'compound_type': 'neutral' if self.is_neutral else 'ionic'
        }

        if self.is_neutral:
            # ==================== NEUTRAL COMPOUND ====================
            K_OC = self.calculate_K_OC_neutral(logK_ow)
            K_WS = rho_wet / (K_OC * OC * rho_dry + P_W + P_A * K_AW )

            result.update({
                'K_WS': K_WS,
                'K_OC': K_OC,
                'note': 'Neutral model: logK_OC = 0.81*logK_ow + 0.1'
            })

##            print(f"\n{'='*50}")
##            print(f"Time = {current_time:.3f} days | NEUTRAL COMPOUND")
##            print(f"logK_ow = {logK_ow} → K_OC = {K_OC:.3f} → K_WS = {K_WS:.3f}")
##            print(f"P_W = {P_W:.3e}, rho_wet = {rho_wet:.3f}")

        else:
            # ==================== IONIC COMPOUND ====================
            z = self.chemical_params['z']
            i_sign = self.chemical_params['i_sign']
            pKa_water = self.chemical_params['pKa_water']
            pH_soil = self.soil_params['pH_soil']
            I = self.soil_params['I']
            A = self.soil_params['A']
            k_setchenov = self.soil_params['k_setchenov']

            if pKa_water is None:
                raise ValueError("pKa_water is required for ionic compounds.")

            # K_OC for neutral and dissociated species
            K_OC_dict = self.calculate_K_OC_ionic(logK_ow)
            K_OC_n = K_OC_dict['K_OC_n']
            K_OC_d = K_OC_dict['K_OC_d']

            # Activity coefficients
            gamma = self.calculate_activity_coefficients(z, I, A, k_setchenov)
            gamma_n, gamma_d = gamma['gamma_n'], gamma['gamma_d']

            # Speciation
            exponent = i_sign * (pH_soil - pKa_water)
            J_n = 1.0 / (1.0 + 10**exponent)
            J_d = 1.0 - J_n

            # Total denominator for f_ns
            denom = (1 / gamma_n + (10**exponent) / gamma_d)

            f_ns = 1 / denom
            f_ds = 1 - f_ns

            K_OC_t = f_ns * K_OC_n + f_ds * K_OC_d

            # Final K_WS
            K_WS = rho_wet / (K_OC_t * OC * rho_dry + P_W)

            result.update({
                'K_WS': K_WS,
                'K_OC_n': K_OC_n,
                'K_OC_d': K_OC_d,
                'K_OC_t': K_OC_t,
                'f_ns': f_ns,
                'f_ds': f_ds,
                'gamma_n': gamma_n,
                'gamma_d': gamma_d,
                'pH_soil': pH_soil,
                'pKa_water': pKa_water,
                'z': z,
                'i_sign': i_sign
            })

##            print(f"\n{'='*60}")
##            print(f"Time = {current_time:.3f} days | IONIC COMPOUND")
##            print(f"pH = {pH_soil}, pKa = {pKa_water}, z = {z}, i_sign = {i_sign}")
##            print(f"K_OC_n = {K_OC_n:.1f}, K_OC_d = {K_OC_d:.1f}, K_OC_t = {K_OC_t:.1f}")
##            print(f"f_ns = {f_ns:.6f}, f_ds = {f_ds:.6f}")
##            print(f"gamma_n = {gamma_n:.3f}, gamma_d = {gamma_d:.3f}")
##            print(f"K_WS = {K_WS:.3f}")

        return result


# Convenience function (unchanged)
def calculate_K_WS_for_plant_model(csv_file: str, current_time: float, time_varying_file: str = None) -> float:
    inputs = load_inputs(csv_file)
    calc = SoilWaterPartitionCalculator(inputs, time_varying_file)
    return calc.calculate_K_WS(current_time)['K_WS']


# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================
if __name__ == "__main__":
    # UPDATE THESE PATHS TO YOUR ACTUAL FILES
    csv_file = os.path.join(current_dir, 'inputs.csv')
    time_varying_file = os.path.join(current_dir, 'time_varying_data.csv')

    try:
        inputs = load_inputs(csv_file)
        calculator = SoilWaterPartitionCalculator(inputs, time_varying_file)

        time_data = load_time_varying_data(time_varying_file)
        time_points = sorted(time_data['Time'].unique())

        results_list = []
        for t in time_points:
            res = calculator.calculate_K_WS(t)
            results_list.append(res)

        # Create DataFrame
        df = pd.DataFrame(results_list)

        # Select and order columns nicely
        base_cols = ['Time', 'compound_type', 'K_WS', 'P_W_used', 'rho_wet_used', 'logK_ow']
        if calculator.is_neutral:
            final_cols = base_cols + ['K_OC']
        else:
            final_cols = base_cols + ['K_OC_n', 'K_OC_d', 'K_OC_t', 'f_ns', 'f_ds',
                                      'pH_soil', 'pKa_water', 'z', 'i_sign']

        df = df[final_cols]

        # Save to Excel
        output_file = "K_WS_results.xlsx"
        df.to_excel(output_file, index=False)
        print("\n" + "="*60)
        print(f"ALL CALCULATIONS FINISHED! Results saved to: {output_file}")
        print(df[['Time', 'compound_type', 'K_WS']].to_string(index=False))

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Error: {e}")
