# PY_Code_Lag.py
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from PartitionCoeff_1 import compute_all
from K_WS_Calculator import calculate_K_WS_for_plant_model, load_time_varying_data, get_value

# Get the current working directory
current_dir = os.getcwd()

# Open the input file in read mode for selector.in
with open('inputs.dat', 'r') as inp:
    values = [float(line.strip()) for line in inp if line.strip()]

# Open the Inputs_1.in in read mode
with open(os.path.join(current_dir, 'inputs_1.csv'), 'r') as f1, \
     open(os.path.join(current_dir, 'time_varying_data_1.csv'), 'r') as f2:
    Inputs_1 = f1.read()
    Inputs_2 = f2.read()

# Replace the variables in the Inputs.in with values from the input file
variable_name = ['k_{dF}', 'I_{mF1}', 'I_{mF2}', 'psi']
values_dict = dict(zip(variable_name, values))
inputs = Inputs_1.format(**values_dict)
inputs_time = Inputs_2.format(**values_dict)

# Write updated files
with open(os.path.join(current_dir, 'inputs.csv'), 'w') as f1, \
     open(os.path.join(current_dir, 'time_varying_data.csv'), 'w') as f2:
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


class PlantConcentrationSolver:
    """Solves plant concentration transport using backward Euler method."""

    def __init__(self, csv_file: str = 'inputs.csv', chemical_properties: Dict = None, time_varying_file: str = None):
        self.inputs = load_inputs(csv_file)
        self.csv_file = csv_file
        self.time_varying_file = time_varying_file
        self.chemical_properties = chemical_properties or self.inputs['chemical_properties']

        if time_varying_file:
            self.time_varying_data = load_time_varying_data(time_varying_file)
            print(f"Time-varying data loaded successfully from {time_varying_file}")
            print(f"Time range: {self.time_varying_data['Time'].min()} to {self.time_varying_data['Time'].max()} days")

            # Add initial sources (C_in and Iem_S at t=0) to initial soil concentration
            initial_time = 0.0
            initial_C_in = self.get_current_C_in(initial_time)
            initial_Iem_S = self.get_current_Iem_S(initial_time)
            initial_MS = self.get_current_MS(initial_time)
            if initial_MS != 0:
                initial_add = (initial_C_in + initial_Iem_S) / initial_MS
                original_C_S = self.inputs['initial_conditions'].get('C_S', 0.0)
                self.inputs['initial_conditions']['C_S'] += initial_add
                print(f"Added initial soil concentration from C_in + Iem_S at t=0: {initial_add:.10f} mg/kg")
                print(f"New initial C_S: {self.inputs['initial_conditions']['C_S']:.10f} mg/kg")

            # Optional: Prevent double-counting by setting t=0 sources to zero after addition
            mask = self.time_varying_data['Time'] == initial_time
            self.time_varying_data.loc[mask, 'C_in'] = 0.0
            self.time_varying_data.loc[mask, 'Iem_S'] = 0.0

        else:
            self.time_varying_data = None
            self.C_in_static = self.inputs['environmental'].get('C_in', 0.0)

        self.setup_default_parameters()
        self.calculate_partition_coefficients()

    def setup_default_parameters(self):
        physical = self.inputs['physical']
        self.DO2 = physical['DO2']
        self.aqueous_layer_thickness = physical['aqueous_layer_thickness']
        self.rho_water = physical['rho_water']
        self.phi = physical['phi']

        soil = self.inputs['soil']
        self.AS = soil['AS']

        self.growth_params = self.inputs['growth_params']
        self.transport_params = self.inputs['transport_params']
        self.chemical_properties = self.inputs['chemical_properties']

        environmental = self.inputs['environmental']
        self.C_A = environmental['C_A']

    def get_current_C_in(self, current_time: float) -> float:
        if self.time_varying_data is not None:
            return get_value(self.time_varying_data, current_time, 'C_in')
        else:
            return self.C_in_static

    def get_current_stress(self, current_time: float) -> float:
        return get_value(self.time_varying_data, current_time, 'stress_f')

    def get_current_MS(self, current_time: float) -> float:
        return get_value(self.time_varying_data, current_time, 'MS')

    def get_current_rho_wet(self, current_time: float) -> float:
        return get_value(self.time_varying_data, current_time, 'rho_wet')

    def get_current_P_W(self, current_time: float) -> float:
        return get_value(self.time_varying_data, current_time, 'P_W')

    def get_current_P_A(self, current_time: float) -> float:
        return get_value(self.time_varying_data, current_time, 'P_A')

    def get_current_phi(self, current_time: float) -> float:
        return get_value(self.time_varying_data, current_time, 'phi')

    def get_current_Temp(self, current_time: float) -> float:
        return get_value(self.time_varying_data, current_time, 'Temp')

    def get_current_Iem_S(self, current_time: float) -> float:
        return get_value(self.time_varying_data, current_time, 'Iem_S')

    def get_current_Iem_R(self, current_time: float) -> float:
        return get_value(self.time_varying_data, current_time, 'Iem_R')

    def get_current_Iem_St(self, current_time: float) -> float:
        return get_value(self.time_varying_data, current_time, 'Iem_St')

    def get_current_Iem_L(self, current_time: float) -> float:
        return get_value(self.time_varying_data, current_time, 'Iem_L')

    def get_current_Iem_F(self, current_time: float) -> float:
        return get_value(self.time_varying_data, current_time, 'Iem_F')

    def is_ionic_compound(self) -> bool:
        pKa = self.chemical_properties.get('pKa_water', 0)
        return pKa != 0

    def calculate_partition_coefficients(self):
        print("Calculating partition coefficients...")
        coeffs = compute_all(**self.chemical_properties,
                             **self.inputs.get('partition_params', {}),
                             W_dict=self.inputs['growth_params']['W_dict'],
                             L_dict=self.inputs['growth_params']['L_dict'],
                             pH_soil=self.inputs['soil_params']['pH_soil'],
##                             pKa_water=self.inputs['chemical_properties']['pKa_water'],
                             csv_file=self.csv_file,
                             time_varying_file=self.time_varying_file,
                             current_time=0.0,
                             verbose=True)

        print(f"\n{'='*50}")
        print("PARTITION COEFFICIENTS:")
        print(f"K_Lph = {coeffs['K_Lph']:.6e}")
        print(f"K_Stph = {coeffs['K_Stph']:.6e}")
        print(f"K_Rph = {coeffs['K_Rph']:.6e}")
        print(f"K_StPA = {coeffs['K_PA']['stem']:.6e}")
        print(f"K_LPA = {coeffs['K_PA']['leaf']:.6e}")
        print(f"K_FPA = {coeffs['K_PA']['fruit']:.6e}")
        print(f"K_RPA = {coeffs['K_PA']['root']:.6e}")
        print(f"{'='*50}\n")

        self.transport_params['K_Lph'] = float(coeffs['K_Lph'])
        self.transport_params['K_Stph'] = float(coeffs['K_Stph'])
        self.transport_params['K_Rph'] = float(coeffs['K_Rph'])
        self.transport_params['K_StPA'] = float(coeffs['K_PA']['stem'])
        self.transport_params['K_LPA'] = float(coeffs['K_PA']['leaf'])
        self.transport_params['K_FPA'] = float(coeffs['K_PA']['fruit'])
        self.transport_params['K_RPA'] = float(coeffs['K_PA']['root'])
        self.transport_params['K_XylW'] = float(coeffs['K_XylW']['root'])
        self.transport_params['K_RXyl'] = float(coeffs['K_RXyl'])
        self.transport_params['K_StXyl'] = float(coeffs['K_StXyl'])
        self.lambda_val = float(coeffs['lambda'])
        self.partition_coefficients = coeffs

        self.Kow_i = 10 ** self.chemical_properties['logK_ow']
        self.KAW_i = self.chemical_properties['K_AW']
        self.mi = self.inputs['calculation']['mi']

        print(f"K_RXyl = {self.transport_params['K_RXyl']:.6e}")
        print(f"K_StXyl = {self.transport_params['K_StXyl']:.6e}")

    def calculate_permeabilities(self) -> Tuple[float, float, float]:
        P_C = 10 ** (0.704 * np.log10(self.Kow_i) - 11.2) * 86400
        P_air = self.KAW_i * np.sqrt(300) / (200 * np.sqrt(self.mi)) * 86400
        P_aqua = self.DO2 * np.sqrt(32 / self.mi) / self.aqueous_layer_thickness
        P_C_tot = 1 / (1 / P_C + 1 / P_air + 1 / P_aqua)
        return P_C, P_air, P_C_tot

    # ============================================================
    # CORRECT MATLAB-CONSISTENT BIOMASS + FRUIT GROWTH SECTION
    # ============================================================

    def _logistic_mass(self, t, M0, Mmax, k):
        return Mmax / (1 + (Mmax / M0) * np.exp(-k * t))


    def _logistic_dMdt(self, M, Mmax, k):
        return k * M * (1 - M / Mmax)


    def _logistic_d2Mdt2(self, M, Mmax, k):
        p1 = 2 * k**2 / Mmax**2
        p2 = -3 * k**2 / Mmax
        p3 = k**2
        return p1 * M**3 + p2 * M**2 + p3 * M


    # ------------------------------------------------------------ 
    # MASS (now correctly handles DYNAMIC)
    # ------------------------------------------------------------ 
    def calculate_mass(self, t: float, compartment: str) -> float:
        params = self.growth_params[compartment]
        M_max = float(params['M_max'])
        M_0   = float(params['M_0'])
        k     = float(params['k'])
        DYNAMIC = self.growth_params.get('DYNAMIC', 1)

        if DYNAMIC == 0:
            # Constant model as requested
            return M_max if compartment != 'F' else 1.21

        # Dynamic model (exactly as in PY_Code_Lag.py)
        if compartment != 'F':
            return self._logistic_mass(t, M_0, M_max, k)

        # Fruit with lag (full sum)
        nF = int(params['nF'])
        lF = float(params['lF'])
        M0_per   = M_0 / nF
        Mmax_per = M_max / nF
        total_mass = 0.0
        for i in range(nF):
            lag = i * lF
            M_i = self._logistic_mass(t - lag, M0_per, Mmax_per, k) if t >= lag else M0_per
            total_mass += M_i
        return total_mass


    # ------------------------------------------------------------
    # FIRST DERIVATIVE dM/dt (MATLAB CONSISTENT)
    # ------------------------------------------------------------
    def calculate_dMdt(self, t: float, compartment: str) -> float:
        DYNAMIC = self.growth_params.get('DYNAMIC', 1)
        if DYNAMIC == 0:
            return 0.0

        params = self.growth_params[compartment]
        M_max = float(params['M_max'])
        M_0   = float(params['M_0'])
        k     = float(params['k'])

        if compartment != 'F':
            M = self.calculate_mass(t, compartment)
            return self._logistic_dMdt(M, M_max, k)

        # Fruit
        nF = int(params['nF'])
        lF = float(params['lF'])
        M0_per   = M_0 / nF
        Mmax_per = M_max / nF
        total_dMdt = 0.0
        for i in range(nF):
            lag = i * lF
            if t >= lag:
                M_i = self._logistic_mass(t - lag, M0_per, Mmax_per, k)
                total_dMdt += self._logistic_dMdt(M_i, Mmax_per, k)
        return total_dMdt


    # ------------------------------------------------------------
    # SECOND DERIVATIVE d²M/dt² (CRITICAL FOR TRANSPIRATION)
    # ------------------------------------------------------------
    def calculate_d2Mdt2(self, t: float, compartment: str) -> float:
        DYNAMIC = self.growth_params.get('DYNAMIC', 1)
        if DYNAMIC == 0:
            return 0.0

        params = self.growth_params[compartment]
        M_max = float(params['M_max'])
        M_0   = float(params['M_0'])
        k     = float(params['k'])

        if compartment != 'F':
            M = self.calculate_mass(t, compartment)
            return self._logistic_d2Mdt2(M, M_max, k)

        nF = int(params['nF'])
        lF = float(params['lF'])
        M0_per   = M_0 / nF
        Mmax_per = M_max / nF
        total_d2 = 0.0
        for i in range(nF):
            lag = i * lF
            if t >= lag:
                M_i = self._logistic_mass(t - lag, M0_per, Mmax_per, k)
                total_d2 += self._logistic_d2Mdt2(M_i, Mmax_per, k)
        return total_d2


    # ------------------------------------------------------------
    # TRANSPIRATION Q_R (NOW EXACTLY MATLAB)
    # ------------------------------------------------------------
    def calculate_growth_based_Q_R(self, t: float) -> float:
        DYNAMIC = self.growth_params.get('DYNAMIC', 1)
        if DYNAMIC == 0:
            return 1.68

        # Dynamic (exactly as in Lag.py)
        dMRdt  = self.calculate_dMdt(t, 'R')
        dMStdt = self.calculate_dMdt(t, 'St')
        dMLdt  = self.calculate_dMdt(t, 'L')
        dMFdt  = self.calculate_dMdt(t, 'F')
        dMtotaldt = dMRdt + dMStdt + dMLdt + dMFdt
        Tc = self.transport_params['T']
        
        return Tc * dMtotaldt


    def calculate_area(self, t: float, compartment: str) -> float:
        M_t = self.calculate_mass(t, compartment)
        SA = self.growth_params[compartment]['SA']
        return SA * M_t

    def calculate_stomatal_permeability(self, t: float, compartment: str) -> float:
        current_Q_R = self.calculate_growth_based_Q_R(t)
        current_Temp = self.get_current_Temp(t)
        current_phi = self.get_current_phi(t)

        A_L = self.calculate_area(t, 'L')
        A_F = self.calculate_area(t, 'F')

        if (A_L + A_F) > 0:
            L_F = A_L / (A_L + A_F)
            F_F = A_F / (A_L + A_F)
        else:
            L_F = F_F = 0.0

        if compartment == 'St':
            Q_in = current_Q_R
        elif compartment == 'L':
            Q_in = current_Q_R * L_F
        elif compartment == 'F':
            Q_in = current_Q_R * F_F
        elif compartment == 'R':
            Q_in = current_Q_R
        else:
            Q_in = 0.0

        A_t = self.calculate_area(t, compartment)
        if A_t <= 0:
            return 0.0

        p_H2O_sat = 610.7 * 10 ** ((7.5 * current_Temp) / (237 + current_Temp))
        C_H2O_sat = p_H2O_sat / (461.9 * (current_Temp + 273.15))
        return (Q_in * self.rho_water / (A_t * (C_H2O_sat - current_phi * C_H2O_sat))
                * np.sqrt(18 / self.mi) * self.KAW_i)

    def calculate_total_permeability(self, t: float, compartment: str) -> float:
        _, _, P_C_tot = self.calculate_permeabilities()
        P_S = self.calculate_stomatal_permeability(t, compartment)
        return P_S + P_C_tot

    def update_time_varying_parameters(self, t: float) -> Dict:
        params = {}
        for comp in ['R', 'St', 'L', 'F']:
            params[f'M_{comp}'] = self.calculate_mass(t, comp)
            params[f'A_{comp}'] = self.calculate_area(t, comp)
            params[f'P_{comp}'] = self.calculate_total_permeability(t, comp)
        return params

    def solve_all_compartments(self, C_prev: np.ndarray, dt: float, t: float) -> np.ndarray:
        current_time = t + dt
        params = self.update_time_varying_parameters(current_time)
        K_WS = calculate_K_WS_for_plant_model(self.csv_file, current_time, self.time_varying_file)
        current_C_in = self.get_current_C_in(current_time)
        current_MS = self.get_current_MS(current_time)
        current_rho_wet = self.get_current_rho_wet(current_time)
        current_P_W = self.get_current_P_W(current_time)
        current_P_A = self.get_current_P_A(current_time)
        current_Iem_S = self.get_current_Iem_S(current_time)
        current_Iem_R = self.get_current_Iem_R(current_time)
        current_Iem_St = self.get_current_Iem_St(current_time)
        current_Iem_L = self.get_current_Iem_L(current_time)
        current_Iem_F = self.get_current_Iem_F(current_time)
        current_Temp = self.get_current_Temp(current_time)

        # Debug prints
        if abs(current_time % 10) < dt or current_time < 2*dt:
            print(f"\n=== ALL COMPARTMENTS at t={current_time:.1f} days ===")
            print(f"K_WS = {K_WS:.6e}")
            print(f"MS = {current_MS:.6e}, rho_wet = {current_rho_wet:.6e}")
            print(f"Previous: C_S={C_prev[0]:.6e}, C_R={C_prev[1]:.6e}, C_St={C_prev[2]:.6e}, C_L={C_prev[3]:.6e}, C_F={C_prev[4]:.6e}")

        psi = self.transport_params['psi']
        v_dep = self.transport_params['v_dep']
        fp = self.transport_params['fp']
        CA = self.C_A
        AS = self.AS
        F = self.lambda_val * self.transport_params['K_XylW']
        Q_Leach = self.transport_params['Q_Leach']
        k_deg = self.transport_params['k_deg']
        alpha_n = self.transport_params['alpha_n']
        g_R = self.transport_params['g_R']
        g_St = self.transport_params['g_St']
        g_L = self.transport_params['g_L']
        g_F = self.transport_params['g_F']
        mdeg_R = self.transport_params['mdeg_R']
        mdeg_St = self.transport_params['mdeg_St']
        mdeg_L = self.transport_params['mdeg_L']
        mdeg_F = self.transport_params['mdeg_F']
        T_ph = self.transport_params['T_ph']
        k_Arr = self.transport_params['k_Arr']
        alpha_n_st = self.transport_params['alpha_n_st']
        alpha_n_L = self.transport_params['alpha_n_L']
        alpha_n_F = self.transport_params['alpha_n_F']
        alpha_n_R = self.transport_params['alpha_n_R']
        rho_St = self.transport_params['rho_St']
        rho_L = self.transport_params['rho_L']
        rho_F = self.transport_params['rho_F']
        rho_R = self.transport_params['rho_R']
        dxW = self.transport_params['dxW']
        dxA = self.transport_params['dxA']
        DYNAMIC = self.growth_params['DYNAMIC']

        # Fruit growth related (phloem)
        WpF = self.growth_params['W_dict']['fruit']
        dMFdt = self.calculate_dMdt(current_time, 'F')
        phloem_to_fruit = dMFdt * (1 - WpF) * self.transport_params['T_ph']

        A_L = params['A_L']
        A_F = params['A_F']
        if A_L + A_F > 0:
            L_F = A_L / (A_L + A_F)
            F_F = A_F / (A_L + A_F)
        else:
            L_F = F_F = 0.0

        # === CONSTANT vs DYNAMIC Q VALUES ===
        if DYNAMIC == 0:
            current_Q_R = 1.68
            Q_L = 1.13
            Q_F = 0.55
        else:
            current_Q_R = self.calculate_growth_based_Q_R(current_time) * self.get_current_stress(current_time)
            Q_L = current_Q_R * L_F - phloem_to_fruit
        if current_time < 46.0:
            Q_F = 0.0
            phloem_to_fruit_effective = 0.0
        else:
            Q_F = current_Q_R * F_F + phloem_to_fruit
            phloem_to_fruit_effective = phloem_to_fruit
            
        A = np.zeros((5, 5))
        b = np.zeros(5)

        #Soil(0)
        f_attach = 0.01
        Dw = self.DO2 * np.sqrt(32 / self.mi)
        Dg = self.DO2 * np.sqrt(18 / self.mi)
        T_w = current_P_W**(10/3) / ((current_P_W + current_P_A)**2)
        T_a = current_P_A**(10/3) / ((current_P_W + current_P_A)**2)
        Dw_eff = Dw * current_P_W * K_WS * T_w
        Dg_eff = Dw * current_P_W * K_WS * self.KAW_i * T_a
        PS = 1 / (dxW / (Dw_eff + Dg_eff) + dxA * current_rho_wet / (Dg * K_WS * self.KAW_i)) * 2

        coeff_S_loss = (Q_Leach * K_WS / current_MS) + (F * current_Q_R * K_WS / current_MS) + \
                       (alpha_n * AS * PS * current_rho_wet / current_MS) + k_deg * k_Arr ** (current_Temp - 20)
        coeff_from_R = psi * current_Q_R / (current_MS * self.transport_params['K_Rph'])

        A[0, 0] = 1 + dt * coeff_S_loss
        A[0, 1] = -dt * coeff_from_R
        b[0] = C_prev[0] + dt * (current_C_in / current_MS + AS * v_dep * fp / current_MS * CA) + (current_Iem_S / current_MS)

        # Root (1)
        A_R = params['A_R']
        P_R = params['P_R']
        coeff_from_S = F * current_Q_R * K_WS / params['M_R']
        coeff_from_St = psi * current_Q_R / (params['M_R'] * self.transport_params['K_Stph'])
        coeff_R_loss = psi * current_Q_R / (params['M_R'] * self.transport_params['K_Rph']) + \
                       current_Q_R / (params['M_R'] * self.transport_params['K_RXyl']) + \
                       alpha_n_R * A_R * P_R * rho_R / (params['M_R'] * self.transport_params['K_RPA']) + \
                       mdeg_R * k_Arr ** (current_Temp - 20) + g_R
        dep_R = (A_R * v_dep * fp / params['M_R'] + A_R * P_R * (1 - fp) / params['M_R']) * CA
        external_R = current_Iem_R / params['M_R']

        A[1, 0] = -dt * (f_attach + coeff_from_S)
        A[1, 1] = 1 + dt * coeff_R_loss
        A[1, 2] = -dt * coeff_from_St
        b[1] = C_prev[1] + dt * dep_R + external_R

        # Stem (2)
        coeff_from_R = current_Q_R / (params['M_St'] * self.transport_params['K_RXyl'])
        phloem_flow = phloem_to_fruit + psi * current_Q_R
        coeff_from_L = phloem_flow / (params['M_St'] * self.transport_params['K_Lph'])
        coeff_St_loss = alpha_n_st * params['A_St'] * params['P_St'] * rho_St / \
                        (params['M_St'] * self.transport_params['K_StPA']) + \
                        phloem_flow / (params['M_St'] * self.transport_params['K_Stph']) + \
                        current_Q_R / (params['M_St'] * self.transport_params['K_StXyl']) + \
                        mdeg_St * k_Arr ** (current_Temp - 20) + g_St
        dep_St = (params['A_St'] * v_dep * fp / params['M_St'] + params['A_St'] * params['P_St'] * (1 - fp) / params['M_St']) * CA
        external_St = current_Iem_St / params['M_St']   # ← fixed: was * M_St

        A[2, 0] = -dt * f_attach
        A[2, 1] = -dt * coeff_from_R
        A[2, 2] = 1 + dt * coeff_St_loss
        A[2, 3] = -dt * coeff_from_L
        b[2] = C_prev[2] + dt * dep_St + external_St

        # Leaves (3)
        coeff_from_St = Q_L / (params['M_L'] * self.transport_params['K_StXyl'])
        phloem_loss_leaf = phloem_to_fruit + psi * current_Q_R
        coeff_L_loss = alpha_n_L * A_L * params['P_L'] * rho_L / \
                       (params['M_L'] * self.transport_params['K_LPA']) + \
                       phloem_loss_leaf / (params['M_L'] * self.transport_params['K_Lph']) + \
                       mdeg_L * k_Arr ** (current_Temp - 20) + g_L
        dep_L = (A_L * v_dep * fp / params['M_L'] + A_L * params['P_L'] * (1 - fp) / params['M_L']) * CA
        external_L = current_Iem_L / params['M_L']

        A[3, 0] = -dt * f_attach
        A[3, 2] = -dt * coeff_from_St
        A[3, 3] = 1 + dt * coeff_L_loss
        b[3] = C_prev[3] + dt * dep_L + external_L

        # Fruits (4)
        current_time = t + dt   # or just use t — both are approximate here
        if current_time < 46:
            coeff_from_St = 0.0
            phloem_to_fruit_effective = 0.0   # also block phloem contribution
        else:
            coeff_from_St_xyl_ph = Q_F / (params['M_F'] * self.transport_params['K_StXyl'])
            coeff_from_St_ph     = phloem_to_fruit / (params['M_F'] * self.transport_params['K_Stph'])
            coeff_from_St        = coeff_from_St_xyl_ph + coeff_from_St_ph
            phloem_to_fruit_effective = phloem_to_fruit
        coeff_F_loss = alpha_n_F * A_F * params['P_F'] * rho_F / \
                       (params['M_F'] * self.transport_params['K_FPA']) + \
                       (mdeg_F * (k_Arr ** (current_Temp - 20))) + g_F
        dep_F = (A_F * v_dep * fp / params['M_F'] + A_F * params['P_F'] * (1 - fp) / params['M_F']) * CA
        external_F = current_Iem_F / params['M_F']

        A[4, 0] = -dt * 0
        A[4, 2] = -dt * coeff_from_St
        A[4, 4] = 1 + dt * coeff_F_loss
        b[4] = C_prev[4] + dt * dep_F + external_F

        if abs(current_time % 10) < dt or current_time < 2*dt:
            print(f"Matrix A:\n{A}")
            print(f"Vector b: {b}")

        try:
            cond_num = np.linalg.cond(A)
            if cond_num > 1e10:
                print(f"Warning: High condition number {cond_num:.2e}, using pseudoinverse")
                solution = np.linalg.pinv(A) @ b
            else:
                solution = np.linalg.solve(A, b)
            solution = np.maximum(solution, 0.0)

            if abs(current_time % 10) < dt or current_time < 2*dt:
                print(f"Solved: C_S={solution[0]:.6e}, C_R={solution[1]:.6e}, C_St={solution[2]:.6e}, C_L={solution[3]:.6e}, C_F={solution[4]:.6e}")

        except np.linalg.LinAlgError as e:
            print(f"Linear algebra error at t={current_time}: {e}")
            solution = np.maximum(C_prev, 0.0)
        except Exception as e:
            print(f"Unexpected error at t={current_time}: {e}")
            solution = np.maximum(C_prev, 0.0)

        return solution

    def solve_system(self, initial_conditions: Dict, dt: float, total_time: float) -> Tuple[np.ndarray, np.ndarray]:
        t_array = np.arange(0, total_time + dt, dt)
        n_steps = len(t_array)

        concentrations = np.zeros((n_steps, 5))  # S, R, St, L, F

        concentrations[0] = [
            initial_conditions['C_S'],
            initial_conditions['C_R'],
            initial_conditions['C_St'],
            initial_conditions['C_L'],
            initial_conditions['C_F']
        ]

        print(f"\n{'='*60}")
        print("INITIAL CONDITIONS:")
        print(f"C_S(0) = {concentrations[0, 0]:.10f} mg/kg")
        print(f"C_R(0) = {concentrations[0, 1]:.10f} mg/kg")
        print(f"C_St(0) = {concentrations[0, 2]:.10f} mg/kg")
        print(f"C_L(0) = {concentrations[0, 3]:.10f} mg/kg")
        print(f"C_F(0) = {concentrations[0, 4]:.10f} mg/kg")
        print(f"{'='*60}\n")

        for i in range(1, n_steps):
            t_prev = t_array[i - 1]
            C_prev = concentrations[i - 1]
            C_new = self.solve_all_compartments(C_prev, dt, t_prev)
            concentrations[i] = C_new

        return t_array, concentrations

    def plot_results(self, t_array: np.ndarray, concentrations: np.ndarray, save_path: str = None):
        fig, axes = plt.subplots(5, 1, figsize=(12, 14))
        compartments = ['Soil', 'Root', 'Stem', 'Leaves', 'Fruits']
        colors = ['brown', 'orange', 'green', 'darkgreen', 'red']

        for i, (ax, comp, color) in enumerate(zip(axes, compartments, colors)):
            ax.plot(t_array, concentrations[:, i], color=color, linewidth=2)
            ax.scatter(0, concentrations[0, i], color='red', s=60, zorder=5,
                       label='t=0 (initial)' if i == 0 else "")

            ax.set_yscale('log')
            positive_vals = concentrations[:, i][concentrations[:, i] > 0]
            if len(positive_vals) > 0:
                y_min = max(1e-12, positive_vals.min() * 0.5)
            else:
                y_min = 1e-12
            ax.set_ylim(bottom=y_min)

            ax.set_ylabel(f'{comp}\nConcentration\n(mg/kg)', fontsize=11, fontweight='bold')
            ax.grid(True, which='both', ls='--', alpha=0.4)
            ax.set_xlim([t_array[0], t_array[-1]])

            if i == 0:
                ax.set_title('Plant Compartment Concentrations (Log Scale)', fontsize=14, fontweight='bold')
                ax.legend()
            if i == 4:
                ax.set_xlabel('Time (days)', fontsize=12, fontweight='bold')

            max_conc = np.max(concentrations[:, i])
            ax.text(0.98, 0.95, f'Max: {max_conc:.3e}', transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

    def plot_mass_results(self, t_array: np.ndarray, save_path: str = None):
        plt.figure(figsize=(12, 8))
        compartments = ['R', 'St', 'L', 'F']
        names = ['Root', 'Stem', 'Leaves', 'Fruits']
        colors = ['orange', 'green', 'darkgreen', 'red']
        for comp, name, color in zip(compartments, names, colors):
            mass = np.array([self.calculate_mass(t, comp) for t in t_array])
            plt.plot(t_array, mass, label=f'{name} Mass', color=color, linewidth=2)
        plt.xlabel('Time (days)')
        plt.ylabel('Mass (kg)')
        plt.title('Plant Compartment Mass Growth Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

    def save_to_csv(self, t_array: np.ndarray, concentrations: np.ndarray, filename: str = 'plant_concentrations_time_varying.csv'):
        df = pd.DataFrame({
            'Time': t_array,
            'C_Soil': concentrations[:, 0],
            'C_Root': concentrations[:, 1],
            'C_Stem': concentrations[:, 2],
            'C_Leaves': concentrations[:, 3],
            'C_Fruits': concentrations[:, 4]
        })
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
        print(f"→ C_Soil at t=0 = {concentrations[0, 0]:.10f} (directly from inputs.csv)")

    def save_mass_to_csv(self, t_array: np.ndarray, filename: str = 'plant_mass_time_varying.csv'):
        compartments = ['R', 'St', 'L', 'F']
        data = {'Time': t_array}
        for comp in compartments:
            data[f'M_{comp}'] = [self.calculate_mass(t, comp) for t in t_array]
        data['M_total'] = [sum(self.calculate_mass(t, comp) for comp in compartments) for t in t_array]
        data['Q_R'] = [self.calculate_growth_based_Q_R(t) for t in t_array]
        pd.DataFrame(data).to_csv(filename, index=False)
        print(f"Mass data (including total biomass M_total and Q_R) saved to {filename}")

    def run_simulation(self, initial_conditions: Dict = None, dt: float = None,
                       total_time: float = None, save_csv: bool = True,
                       csv_filename: str = 'plant_concentrations_time_varying.csv',
                       plot_filename: str = None, plot_mass: bool = True,
                       mass_plot_filename: str = None, save_mass_csv: bool = True,
                       mass_csv_filename: str = 'plant_mass_time_varying.csv'):

        if initial_conditions is None:
            initial_conditions = self.inputs['initial_conditions']
        if dt is None:
            dt = self.inputs['simulation']['dt']
        if total_time is None:
            total_time = self.inputs['simulation']['total_time']

        print("=== PLANT UPTAKE SIMULATION STARTED ===")
        print(f"Initial soil concentration (t=0): {initial_conditions['C_S']:.10f} mg/kg ← FROM inputs.csv")

        t_array, concentrations = self.solve_system(initial_conditions, dt, total_time)

        self.plot_results(t_array, concentrations, plot_filename)
        if plot_mass:
            self.plot_mass_results(t_array, mass_plot_filename)
        if save_csv:
            self.save_to_csv(t_array, concentrations, csv_filename)
        if save_mass_csv:
            self.save_mass_to_csv(t_array, mass_csv_filename)

        print("Simulation completed. All concentration plots use logarithmic Y-scale.")
        return t_array, concentrations

if __name__ == "__main__":
    print("PLANT CONCENTRATION TRANSPORT MODEL")
    time_varying_file = 'time_varying_data.csv'
    inputs_file = 'inputs.csv'

    solver = PlantConcentrationSolver(csv_file=inputs_file, time_varying_file=time_varying_file)
    t_array, results = solver.run_simulation(
        save_csv=True,
        plot_filename='concentrations_log_scale.png',
        plot_mass=True,
        mass_plot_filename='mass.png'
    )

    exp_file = 'Exp.out'
    if os.path.exists(exp_file):
        exp_data = np.loadtxt(exp_file)
        modelled_time = t_array
        modelled_fruits = results[:, 4]
        Modelled_F = np.column_stack((modelled_time, modelled_fruits))

        l = len(exp_data)
        BTC_F = np.zeros((l, 2))
        for i in range(l):
            idx = np.argmin(np.abs(Modelled_F[:, 0] - exp_data[i, 0]))
            BTC_F[i, :] = Modelled_F[idx, :]

        np.savetxt('output.dat', BTC_F, delimiter=' ', fmt='%.18e')
        print("output.dat successfully created.")
    else:
        print(f"Warning: Experimental file '{exp_file}' not found.")
