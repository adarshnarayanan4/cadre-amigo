"""Readable terminal reports for CADRE preparation and optimization results."""


def print_heading(title):
    """Start a section on its own lines, separated from preceding solver output."""
    rule = "=" * 72

    print(f"\n{rule}\n{title}\n{rule}\n", flush=True)


def print_values(rows):
    """Align labels and already-formatted values without rounding saved data."""
    rows = list(rows)
    width = max(len(label) for label, _ in rows)

    for label, value in rows:
        print(f"  {label:<{width}} : {value}", flush=True)

    print(flush=True)


def print_status(title, rows):
    """Keep solver and setup diagnostics separate from the iteration table."""
    print_heading(title)

    print_values(rows)


def print_kkt_matrix(shape, stored_entries, maximum, factorization, system):
    """Identify the system being factored, including initialization solves."""
    stage = "MULTIPLIER ESTIMATE" if system == "normal_equations" else "NEWTON STEP"

    print_status(
        f"KKT MATRIX: {stage}",
        [
            ("Factorization", f"{factorization:,}"),
            ("Dimensions", f"{shape[0]:,} x {shape[1]:,}"),
            ("Stored entries (nnz)", f"{stored_entries:,}"),
            ("Largest absolute entry", f"{maximum:.6e}"),
        ],
    )


def print_validation(title, metrics):
    """Show numerical validation measurements one per line."""
    rows = []

    for key, value in metrics.items():
        if isinstance(value, bool):
            formatted = "Yes" if value else "No"
        elif isinstance(value, int):
            formatted = f"{value:,}"
        else:
            formatted = f"{value:.3e}"

        rows.append((key.replace("_", " "), formatted))

    print_status(title, rows)


def print_ranges(rows):
    """Keep each minimum, maximum, and allowed range on the same row."""
    print(f"  {'Quantity':<25} {'Minimum':>11} {'Maximum':>11}  Limits")
    print(f"  {'-' * 25} {'-' * 11} {'-' * 11}  {'-' * 12}")

    for label, minimum, maximum, limits in rows:
        print(f"  {label:<25} {minimum:>11.6f} {maximum:>11.6f}  {limits}")

    print(flush=True)


def print_feasibility(title, metrics):
    """Show battery ranges and small residuals without dumping a dictionary."""
    print_heading(title)

    print_ranges(
        [
            ("Battery SOC", metrics["SOC_min"], metrics["SOC_max"], "0.2 to 1.0"),
            (
                "Battery current [A]",
                metrics["current_min_A"],
                metrics["current_max_A"],
                "-10 to 5",
            ),
        ]
    )

    print_values(
        [
            ("SOC periodicity error", f"{metrics['periodicity_error']:.3e}"),
            ("Inequality violation", f"{metrics['inequality_violation']:.3e}"),
        ]
    )


def print_repair_header(interior):
    title = "INTERIOR-START REPAIR" if interior else "BASELINE REPAIR"
    print_heading(title)

    print(
        f"  {'Step':>4}  {'SOC min':>8} {'SOC max':>8}  "
        f"{'I min [A]':>10} {'I max [A]':>10}  "
        f"{'SOC error':>10} {'Violation':>9}",
        flush=True,
    )


def print_repair_iteration(iteration, metrics):
    print(
        f"  {iteration:>4}  "
        f"{metrics['SOC_min']:>8.5f} {metrics['SOC_max']:>8.5f}  "
        f"{metrics['current_min_A']:>10.5f} {metrics['current_max_A']:>10.5f}  "
        f"{metrics['periodicity_error']:>10.2e} "
        f"{metrics['inequality_violation']:>9.2e}",
        flush=True,
    )


def print_convergence(history):
    print_heading("OPTIMIZATION FINISHED")

    status = "Not converged"

    if history.get("converged", False):
        status = (
            "Acceptable convergence"
            if history.get("acceptable", False)
            else "Converged"
        )

    rows = [("Solver status", status)]
    iterations = history.get("iterations", [])

    if len(iterations) > 0:
        last = iterations[-1]
        rows.extend(
            [
                ("Final iteration", last.get("iteration", -1)),
                ("Final NLP residual", f"{last.get('nlp_error', float('nan')):.3e}"),
            ]
        )

    print_values(rows)


def print_replay_errors(errors):
    """Keep every independent error visible, with readable labels and units."""
    labels = {
        "solar_current_A": ("Solar current", "A"),
        "solar_voltage_V": ("Solar voltage", "V"),
        "solar_power_W": ("Solar power", "W"),
        "current_A": ("Battery current", "A"),
        "initial_SOC": ("Initial SOC", "SOC"),
        "P_comm_W": ("Communication power", "W"),
        "temperature_K": ("Temperature", "K"),
        "P_bat_W": ("Battery power", "W"),
        "SOC": ("Battery SOC", "SOC"),
        "Dr": ("Download rate", "CADRE units/s"),
        "Data": ("Downloaded data", "CADRE units"),
        "roll_angle_rad": ("Roll angle", "rad"),
        "antenna_gain": ("Antenna gain", ""),
        "angular_velocity": ("Angular velocity", "1/s"),
        "attitude_torque": ("Attitude torque", "N m"),
        "reaction_wheel_power_W": ("Reaction-wheel power", "W"),
    }

    print_heading("INDEPENDENT REPLAY VALIDATION")

    print(f"  {'Quantity':<25} {'Maximum error':>13}  Units")
    print(f"  {'-' * 25} {'-' * 13}  {'-' * 13}")

    for key, error in errors.items():
        label, unit = labels.get(key, (key.replace("_", " "), ""))
        print(f"  {label:<25} {error:>13.3e}  {unit}")

    print(flush=True)


def print_results(summary, diagnostics, output):
    """Present the final comparison, feasibility checks, and saved location."""
    print_heading("P_COMM OPTIMIZATION RESULTS")

    print_values(
        [
            ("Independent validation", "PASSED" if summary["validated"] else "FAILED"),
            (
                "Solver time",
                f"{summary['solver_time_s']:.2f} s ({summary['solver_time_scope']})",
            ),
        ]
    )

    print("  Downloaded data [CADRE data units]\n")

    print_values(
        [
            ("Feasible baseline", f"{summary['baseline_data']:,.6f}"),
            ("Initial guess", f"{diagnostics['initial_guess_data']:,.6f}"),
            ("Optimized", f"{diagnostics['optimized_data']:,.6f}"),
            ("Gain over baseline", f"{diagnostics['data_gain']:,.6f}"),
            ("Improvement", f"{diagnostics['improvement_percent']:.4f}%"),
            ("Original CADRE (unrepaired)", f"{diagnostics['original_data']:,.6f}"),
        ]
    )

    print_heading("COMMUNICATION POWER")

    print_ranges(
        [
            ("Control points [W]", *diagnostics["control_range"], "0 to 25"),
            ("Trajectory [W]", *diagnostics["power_range"], "0 to 25"),
        ]
    )

    print_values(
        [
            ("Largest control-point change", f"{diagnostics['control_change']:.6f} W"),
        ]
    )

    print_feasibility(
        "BATTERY FEASIBILITY (INDEPENDENT REPLAY)", diagnostics["battery_metrics"]
    )

    print_values(
        [
            ("Optimized initial SOC", f"{diagnostics['endpoint_soc'][0]:.9f}"),
            ("Optimized final SOC", f"{diagnostics['endpoint_soc'][1]:.9f}"),
            ("Overall bound violation", f"{summary['bound_violation']:.3e}"),
        ]
    )

    print_replay_errors(summary["reference_max_errors"])

    print_heading("SAVED RESULTS")
    print(f"  {output}\n")
    print("  summary.json          Full-precision results and validation")
    print("  solver_history.json   Iteration history")
    print("  trajectories.npz      Saved trajectories\n", flush=True)
