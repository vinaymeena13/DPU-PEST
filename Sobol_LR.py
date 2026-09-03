# ======================================================================
# Sobol_optimization_weighted.py
# ======================================================================
"""
Sobol Sequence based parameter optimization for the
Dynamic Plant Uptake (DPU) model.

Optimizes:
    kd_F, Ie_F1, Ie_F2

Objective:
    Weighted Sum of Squared Errors (absolute)

    Phi = sum( w_i * (C_obs_i - C_sim_i)^2 )

Uses:
    - DPU_run(parameters)
    - Obs_W.xlsx  (Time, Concentration, Weightage)

Optimization workflow:

    Sobol global search
            ↓
    Best Sobol solution
            ↓
    Nelder-Mead local refinement
            ↓
    Final refined solution

The script stores BOTH:
    1. Best solution obtained directly from Sobol search
    2. Solution obtained after local refinement

Both results are stored in the same Excel sheet.
"""

# ======================================================================
# 1. IMPORTS
# ======================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.stats import qmc

from DPU_run import DPU_run


# ======================================================================
# 2. USER SETTINGS
# ======================================================================

# ----------------------------------------------------------------------
# Parameters to optimize
# ----------------------------------------------------------------------

PARAMETER_NAMES = [
    'kd_F',
    'Ie_F1',
    'Ie_F2'
]


# ----------------------------------------------------------------------
# Parameter bounds
# ----------------------------------------------------------------------

PARAMETER_BOUNDS = [
    (1.0e-10, 50.0),   # kd_F
    (1.0e-10, 50.0),   # Ie_F1
    (1.0e-10, 50.0),   # Ie_F2
]


# ----------------------------------------------------------------------
# Observation file
# ----------------------------------------------------------------------

OBSERVATION_FILE = "Obs_W.xlsx"

OBSERVATION_SHEET = 0

TIME_COLUMN_NAME = "Time"

OBSERVATION_COLUMN_NAME = "Concentration"

WEIGHT_COLUMN_NAME = "Weightage"


# ======================================================================
# 3. SOBOL SETTINGS
# ======================================================================

# Number of Sobol samples.
#
# Sobol sequences work most naturally with powers of two.
#
# 512 = 2^9
#
# Therefore, the script generates exactly 512 Sobol points.

SOBOL_POWER = 9

SOBOL_SAMPLES = 2 ** SOBOL_POWER


# ----------------------------------------------------------------------
# Sobol scrambling
# ----------------------------------------------------------------------
#
# Scrambling improves the distribution and makes the sequence
# less sensitive to the exact structure of the parameter space.
#
# True is recommended.

SOBOL_SCRAMBLE = True


# ----------------------------------------------------------------------
# Sobol random seed
# ----------------------------------------------------------------------

SOBOL_SEED = 20260817


# ======================================================================
# 4. LOCAL REFINEMENT SETTINGS
# ======================================================================

USE_LOCAL_REFINEMENT = True

LOCAL_METHOD = "Nelder-Mead"

LOCAL_MAXITER = 500


# ======================================================================
# 5. OUTPUT SETTINGS
# ======================================================================

OUTPUT_DIR = "Sobol_results"


# ======================================================================
# 6. LOAD OBSERVATIONS
# ======================================================================

def load_observations():

    df = pd.read_excel(
        OBSERVATION_FILE,
        sheet_name=OBSERVATION_SHEET
    )

    required = [
        TIME_COLUMN_NAME,
        OBSERVATION_COLUMN_NAME,
        WEIGHT_COLUMN_NAME
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns in {OBSERVATION_FILE}: {missing}"
        )

    obs_time = pd.to_numeric(
        df[TIME_COLUMN_NAME],
        errors="coerce"
    ).to_numpy(dtype=float)

    obs_value = pd.to_numeric(
        df[OBSERVATION_COLUMN_NAME],
        errors="coerce"
    ).to_numpy(dtype=float)

    weights = pd.to_numeric(
        df[WEIGHT_COLUMN_NAME],
        errors="coerce"
    ).to_numpy(dtype=float)

    valid = (
        np.isfinite(obs_time)
        &
        np.isfinite(obs_value)
        &
        np.isfinite(weights)
    )

    obs_time = obs_time[valid]

    obs_value = obs_value[valid]

    weights = weights[valid]

    if len(obs_time) == 0:

        raise ValueError(
            "No valid observations found."
        )

    order = np.argsort(obs_time)

    return (
        obs_time[order],
        obs_value[order],
        weights[order]
    )


# ======================================================================
# 7. EXTRACT MODEL OUTPUT
# ======================================================================

def extract_model_output(model_result):

    arr = np.asarray(
        model_result,
        dtype=float
    )

    if arr.ndim != 2 or arr.shape[1] < 2:

        raise ValueError(
            "DPU_run did not return a usable "
            "(time, concentration) array."
        )

    return arr[:, 0], arr[:, 1]


# ======================================================================
# 8. GET SIMULATED VALUES AT OBSERVATION TIMES
# ======================================================================

def get_simulated_at_observation_times(
    model_result,
    obs_time
):

    model_time, model_conc = extract_model_output(
        model_result
    )

    valid = (
        np.isfinite(model_time)
        &
        np.isfinite(model_conc)
    )

    model_time = model_time[valid]

    model_conc = model_conc[valid]

    if len(model_time) == 0:

        raise ValueError(
            "Model returned no finite results."
        )

    simulated = np.empty(
        len(obs_time),
        dtype=float
    )

    for i, t in enumerate(obs_time):

        idx = np.argmin(
            np.abs(model_time - t)
        )

        simulated[i] = model_conc[idx]

    return simulated


# ======================================================================
# 9. OBJECTIVE FUNCTION  (Weighted Least-Squares - ABSOLUTE)
# ======================================================================

def weighted_objective(
    parameters,
    obs_time,
    obs_value,
    weights
):

    """
    Objective to minimize (Weighted Least-Squares - absolute):

        Phi = sum( w_i * (C_obs_i - C_sim_i)^2 )
    """

    try:

        parameters = np.asarray(
            parameters,
            dtype=float
        )

        # --------------------------------------------------------------
        # Check for invalid parameter values
        # --------------------------------------------------------------

        if not np.all(
            np.isfinite(parameters)
        ):

            return 1e30


        # --------------------------------------------------------------
        # Check parameter bounds
        # --------------------------------------------------------------

        for value, bounds in zip(
            parameters,
            PARAMETER_BOUNDS
        ):

            lower, upper = bounds

            if (
                value < lower
                or
                value > upper
            ):

                return 1e30


        # --------------------------------------------------------------
        # Run DPU model
        # --------------------------------------------------------------

        model_result = DPU_run(
            parameters
        )


        # --------------------------------------------------------------
        # Obtain simulated values at observation times
        # --------------------------------------------------------------

        simulated = (
            get_simulated_at_observation_times(
                model_result,
                obs_time
            )
        )


        # --------------------------------------------------------------
        # Validate simulation
        # --------------------------------------------------------------

        if (
            len(simulated) != len(obs_value)
            or
            not np.all(
                np.isfinite(simulated)
            )
        ):

            return 1e30


        # --------------------------------------------------------------
        # Calculate residuals (absolute)
        # --------------------------------------------------------------

        residual = (
            obs_value
            -
            simulated
        )


        # --------------------------------------------------------------
        # Calculate Weighted SSE
        # --------------------------------------------------------------

        phi = np.sum(
            weights * residual**2
        )


        # --------------------------------------------------------------
        # Check objective value
        # --------------------------------------------------------------

        if not np.isfinite(phi):

            return 1e30


        return float(phi)


    except Exception as e:

        print(
            f"  Evaluation failed: {e}"
        )

        return 1e30


# ======================================================================
# 10. SOBOL GLOBAL OPTIMIZATION
# ======================================================================

def sobol_optimization(
    objective_function,
    bounds,
    args=(),
    power=9,
    scramble=True,
    seed=20260817
):

    """
    Perform global search using a Sobol low-discrepancy sequence.

    Parameters
    ----------
    objective_function :
        Objective function to minimize.

    bounds :
        List of (lower, upper) parameter bounds.

    args :
        Additional arguments passed to objective_function.

    power :
        Generates 2**power Sobol points.

    scramble :
        Whether to scramble the Sobol sequence.

    seed :
        Random seed for reproducibility.

    Returns
    -------
    best_position :
        Best parameter set found by Sobol search.

    best_value :
        Best Weighted SSE found by Sobol search.

    history :
        Convergence history.

    all_results :
        All Sobol parameter combinations and their objective values.

    n_evaluations :
        Number of DPU model evaluations.
    """


    # ==================================================================
    # Number of parameters
    # ==================================================================

    n_parameters = len(bounds)


    # ==================================================================
    # Lower and upper bounds
    # ==================================================================

    lower_bounds = np.array(
        [
            b[0]
            for b in bounds
        ],
        dtype=float
    )

    upper_bounds = np.array(
        [
            b[1]
            for b in bounds
        ],
        dtype=float
    )


    # ==================================================================
    # Create Sobol sampler
    # ==================================================================

    sampler = qmc.Sobol(
        d=n_parameters,
        scramble=scramble,
        seed=seed
    )


    # ==================================================================
    # Generate Sobol points
    # ==================================================================

    # random_base2(m) generates exactly 2**m points.

    sobol_unit_points = sampler.random_base2(
        m=power
    )


    # ==================================================================
    # Scale Sobol points to actual parameter bounds
    # ==================================================================

    sobol_parameter_points = (
        qmc.scale(
            sobol_unit_points,
            lower_bounds,
            upper_bounds
        )
    )


    # ==================================================================
    # Evaluate every Sobol point
    # ==================================================================

    n_evaluations = 0

    best_position = None

    best_value = np.inf

    history = []

    all_results = []


    print()

    print(
        f"Total Sobol samples = "
        f"{len(sobol_parameter_points)}"
    )

    print()


    for i, parameters in enumerate(
        sobol_parameter_points
    ):

        # --------------------------------------------------------------
        # Calculate objective
        # --------------------------------------------------------------

        objective_value = objective_function(
            parameters,
            *args
        )

        n_evaluations += 1


        # --------------------------------------------------------------
        # Store all results (using PARAMETER_NAMES dynamically)
        # --------------------------------------------------------------

        row = {"Evaluation": n_evaluations}
        for name, val in zip(PARAMETER_NAMES, parameters):
            row[name] = val
        row["Weighted_SSE"] = objective_value
        all_results.append(row)


        # --------------------------------------------------------------
        # Update best result
        # --------------------------------------------------------------

        if objective_value < best_value:

            best_value = float(
                objective_value
            )

            best_position = (
                parameters.copy()
            )


        # --------------------------------------------------------------
        # Store convergence history
        # --------------------------------------------------------------

        history.append({

            "Evaluation":
                n_evaluations,

            "Best_Weighted_SSE":
                best_value
        })


        # --------------------------------------------------------------
        # Print progress
        # --------------------------------------------------------------

        print(

            f"Sobol evaluation "
            f"{i + 1:4d}/"
            f"{len(sobol_parameter_points)} | "
            f"Current SSE = "
            f"{objective_value:.10e} | "
            f"Best SSE = "
            f"{best_value:.10e}"
        )


    # ==================================================================
    # Convert all results to DataFrame
    # ==================================================================

    all_results = pd.DataFrame(
        all_results
    )


    # ==================================================================
    # Return
    # ==================================================================

    return (
        best_position,
        best_value,
        history,
        all_results,
        n_evaluations
    )


# ======================================================================
# 11. MAIN OPTIMIZATION
# ======================================================================

def main():

    warnings.filterwarnings(
        "ignore"
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    # ==================================================================
    # Load observations
    # ==================================================================

    (
        obs_time,
        obs_value,
        weights
    ) = load_observations()


    # ==================================================================
    # Print optimization information
    # ==================================================================

    print(
        "=" * 70
    )

    print(
        "SOBOL SEQUENCE PARAMETER OPTIMIZATION"
    )

    print(
        "DPU MODEL"
    )

    print(
        "=" * 70
    )

    print(
        f"Parameters       : "
        f"{PARAMETER_NAMES}"
    )

    print(
        f"Bounds           : "
        f"{PARAMETER_BOUNDS}"
    )

    print(
        f"Observations     : "
        f"{len(obs_time)}"
    )

    print(
        f"Sobol samples    : "
        f"{SOBOL_SAMPLES}"
    )

    print(
        f"Sobol scramble   : "
        f"{SOBOL_SCRAMBLE}"
    )

    print()


    # ==================================================================
    # SOBOL GLOBAL SEARCH
    # ==================================================================

    print(
        "=" * 70
    )

    print(
        "STARTING SOBOL GLOBAL OPTIMIZATION"
    )

    print(
        "=" * 70
    )


    (
        best_sobol_params,
        best_sobol_obj,
        sobol_history,
        all_sobol_results,
        sobol_nfev
    ) = sobol_optimization(

        objective_function=
            weighted_objective,

        bounds=
            PARAMETER_BOUNDS,

        args=(
            obs_time,
            obs_value,
            weights
        ),

        power=
            SOBOL_POWER,

        scramble=
            SOBOL_SCRAMBLE,

        seed=
            SOBOL_SEED
    )


    # ==================================================================
    # IMPORTANT:
    # Preserve the Sobol result BEFORE local refinement
    # ==================================================================

    best_sobol_params = (
        np.asarray(
            best_sobol_params,
            dtype=float
        ).copy()
    )

    best_sobol_obj = float(
        best_sobol_obj
    )


    # ==================================================================
    # PRINT SOBOL RESULT
    # ==================================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "SOBOL OPTIMIZATION RESULT"
    )

    print(
        "=" * 70
    )

    print(
        f"Best Sobol Weighted SSE = "
        f"{best_sobol_obj:.10e}"
    )


    for name, value in zip(
        PARAMETER_NAMES,
        best_sobol_params
    ):

        print(
            f"  {name:12s} = "
            f"{value:.10e}"
        )


    print(
        f"Number of evaluations = "
        f"{sobol_nfev}"
    )


    # ==================================================================
    # INITIALIZE FINAL RESULT
    # ==================================================================

    final_params = (
        best_sobol_params.copy()
    )

    final_obj = (
        best_sobol_obj
    )

    stage = (
        "Sobol Optimization"
    )


    # ==================================================================
    # INITIALIZE LOCAL REFINEMENT RESULT
    # ==================================================================

    # These variables preserve the result after local refinement.

    local_params = (
        best_sobol_params.copy()
    )

    local_obj_value = (
        best_sobol_obj
    )


    # ==================================================================
    # LOCAL REFINEMENT
    # ==================================================================

    if USE_LOCAL_REFINEMENT:

        print(
            "\n" + "=" * 70
        )

        print(
            "LOCAL REFINEMENT (Nelder-Mead)"
        )

        print(
            "=" * 70
        )


        # --------------------------------------------------------------
        # Local objective function
        # --------------------------------------------------------------

        def local_obj(x):

            x = np.asarray(
                x,
                dtype=float
            )


            # ----------------------------------------------------------
            # Keep local optimization inside parameter bounds
            # ----------------------------------------------------------

            for value, bounds in zip(
                x,
                PARAMETER_BOUNDS
            ):

                lower, upper = bounds

                if (
                    value < lower
                    or
                    value > upper
                ):

                    return 1e30


            return weighted_objective(
                x,
                obs_time,
                obs_value,
                weights
            )


        # --------------------------------------------------------------
        # Run Nelder-Mead
        # --------------------------------------------------------------

        res_local = minimize(

            local_obj,

            best_sobol_params,

            method=LOCAL_METHOD,

            options={
                "maxiter":
                    LOCAL_MAXITER,

                "xatol":
                    1e-12,

                "fatol":
                    1e-14,

                "disp":
                    True
            }
        )


        # --------------------------------------------------------------
        # Store actual local refinement result
        # --------------------------------------------------------------

        local_params = (
            res_local.x.copy()
        )

        local_obj_value = float(
            res_local.fun
        )


        # --------------------------------------------------------------
        # Determine final result
        # --------------------------------------------------------------

        if (
            local_obj_value
            <
            best_sobol_obj
        ):

            final_params = (
                local_params.copy()
            )

            final_obj = (
                local_obj_value
            )

            stage = (
                "Sobol + Local refinement"
            )


            print(
                "Local refinement improved "
                "the Sobol solution."
            )


        else:

            final_params = (
                best_sobol_params.copy()
            )

            final_obj = (
                best_sobol_obj
            )

            stage = (
                "Sobol Optimization"
            )


            print(
                "Local refinement did not "
                "improve the Sobol solution."
            )


    # ==================================================================
    # FINAL MODEL RUN
    # ==================================================================

    model_result = DPU_run(
        final_params
    )


    # ==================================================================
    # FINAL SIMULATED VALUES
    # ==================================================================

    simulated = (
        get_simulated_at_observation_times(
            model_result,
            obs_time
        )
    )


    # ==================================================================
    # FINAL RESIDUALS
    # ==================================================================

    residual = (
        obs_value
        -
        simulated
    )


    # ==================================================================
    # FINAL WEIGHTED SQUARED ERRORS
    # ==================================================================

    weighted_sq_err = (
        weights
        *
        residual**2
    )


    # ==================================================================
    # DIAGNOSTICS DATAFRAME
    # ==================================================================

    diagnostics = pd.DataFrame({

        "Time":
            obs_time,

        "Observed":
            obs_value,

        "Simulated":
            simulated,

        "Residual":
            residual,

        "Weight":
            weights,

        "Weighted_Squared_Error":
            weighted_sq_err
    })


    # ==================================================================
    # SAVE ALL SOBOL EVALUATIONS
    # ==================================================================

    all_sobol_results.to_csv(

        os.path.join(
            OUTPUT_DIR,
            "all_Sobol_evaluations.csv"
        ),

        index=False
    )


    # ==================================================================
    # SAVE SOBOL CONVERGENCE HISTORY
    # ==================================================================

    sobol_history_df = pd.DataFrame(
        sobol_history
    )


    sobol_history_df.to_csv(

        os.path.join(
            OUTPUT_DIR,
            "Sobol_convergence_history.csv"
        ),

        index=False
    )


    # ==================================================================
    # SAVE RESULTS SUMMARY
    # ======================================================================
    #
    # IMPORTANT:
    #
    # This table preserves BOTH stages:
    #
    #     Sobol Result
    #     Local Refinement Result
    #
    # The Sobol values are NOT replaced by the refined values.
    #
    # ======================================================================

    summary = pd.DataFrame({

        "Parameter":
            PARAMETER_NAMES
            +
            ["Weighted_SSE"],

        "Sobol Result":
            list(best_sobol_params)
            +
            [best_sobol_obj],

        "Local Refinement Result":
            list(local_params)
            +
            [local_obj_value]
    })


    # ==================================================================
    # SAVE SUMMARY EXCEL FILE
    # ==================================================================

    summary.to_excel(

        os.path.join(
            OUTPUT_DIR,
            "Sobol_and_Local_Refinement_results.xlsx"
        ),

        index=False
    )


    # ==================================================================
    # SAVE OBSERVED VS SIMULATED DATA
    # ==================================================================

    diagnostics.to_csv(

        os.path.join(
            OUTPUT_DIR,
            "observed_vs_simulated.csv"
        ),

        index=False
    )


    # ==================================================================
    # SOBOL CONVERGENCE PLOT
    # ==================================================================

    plt.figure(
        figsize=(9, 5)
    )


    plt.plot(

        sobol_history_df[
            "Evaluation"
        ],

        sobol_history_df[
            "Best_Weighted_SSE"
        ],

        linewidth=2
    )


    plt.xlabel(
        "Sobol Evaluation"
    )

    plt.ylabel(
        "Best Weighted SSE"
    )

    plt.title(
        "Sobol Optimization Convergence"
    )

    plt.grid(
        True,
        alpha=0.4
    )

    plt.tight_layout()


    plt.savefig(

        os.path.join(
            OUTPUT_DIR,
            "Sobol_convergence.png"
        ),

        dpi=300
    )

    plt.close()


    # ==================================================================
    # OBSERVED VS SIMULATED PLOT
    # ==================================================================

    plt.figure(
        figsize=(9, 5)
    )


    plt.scatter(

        obs_time,

        obs_value,

        s=50,

        label="Observed",

        zorder=5
    )


    plt.plot(

        obs_time,

        simulated,

        "r-",

        linewidth=2,

        label="Sobol optimized model"
    )


    plt.xlabel(
        "Time"
    )

    plt.ylabel(
        "Concentration"
    )

    plt.title(
        "Observed vs Sobol-Optimized Model"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.4
    )

    plt.tight_layout()


    plt.savefig(

        os.path.join(
            OUTPUT_DIR,
            "observed_vs_simulated.png"
        ),

        dpi=300
    )

    plt.close()


    # ==================================================================
    # FINAL REPORT
    # ==================================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL OPTIMIZATION RESULT"
    )

    print(
        "=" * 70
    )


    print(
        f"Optimization stage : "
        f"{stage}"
    )


    print(
        f"Final Weighted SSE : "
        f"{final_obj:.12e}"
    )


    # ==================================================================
    # FINAL PARAMETERS
    # ==================================================================

    print(
        "\nBest final parameters:"
    )


    for name, value in zip(
        PARAMETER_NAMES,
        final_params
    ):

        print(
            f"  {name:12s} = "
            f"{value:.12e}"
        )


    # ==================================================================
    # SOBOL PARAMETERS
    # ==================================================================

    print(
        "\nParameters BEFORE local refinement "
        "(Sobol result):"
    )


    for name, value in zip(
        PARAMETER_NAMES,
        best_sobol_params
    ):

        print(
            f"  {name:12s} = "
            f"{value:.12e}"
        )


    print(
        f"  {'Weighted_SSE':12s} = "
        f"{best_sobol_obj:.12e}"
    )


    # ==================================================================
    # LOCAL REFINEMENT PARAMETERS
    # ==================================================================

    print(
        "\nParameters AFTER local refinement:"
    )


    for name, value in zip(
        PARAMETER_NAMES,
        local_params
    ):

        print(
            f"  {name:12s} = "
            f"{value:.12e}"
        )


    print(
        f"  {'Weighted_SSE':12s} = "
        f"{local_obj_value:.12e}"
    )


    # ==================================================================
    # OBSERVATION-WISE RESULTS
    # ==================================================================

    print(
        "\nObservation-wise results:"
    )


    print(
        diagnostics.to_string(
            index=False
        )
    )


    # ==================================================================
    # TOTAL WEIGHTED SSE
    # ==================================================================

    print(
        f"\nTotal Weighted SSE = "
        f"{diagnostics['Weighted_Squared_Error'].sum():.12e}"
    )


    # ==================================================================
    # NUMBER OF EVALUATIONS
    # ==================================================================

    print(
        f"\nTotal Sobol model evaluations = "
        f"{sobol_nfev}"
    )


    # ==================================================================
    # OUTPUT LOCATION
    # ==================================================================

    print(
        f"\nResults saved in folder: "
        f"{os.path.abspath(OUTPUT_DIR)}"
    )


    print(
        "\nFiles generated:"
    )

    print(
        "  1. Sobol_and_Local_Refinement_results.xlsx"
    )

    print(
        "  2. all_Sobol_evaluations.csv"
    )

    print(
        "  3. Sobol_convergence_history.csv"
    )

    print(
        "  4. observed_vs_simulated.csv"
    )

    print(
        "  5. Sobol_convergence.png"
    )

    print(
        "  6. observed_vs_simulated.png"
    )


    print(
        "\nOptimization finished successfully."
    )


# ======================================================================
# 12. RUN MAIN PROGRAM
# ======================================================================

if __name__ == "__main__":

    main()
