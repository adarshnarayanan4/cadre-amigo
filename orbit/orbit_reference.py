import numpy as np
import matplotlib.pyplot as plt

# Earth constants
MU = 398600.44      # Earth's gravitational parameter in km^3/s^2
RE = 6378.137       # Earth's radius in km

J2 = 1.08264e-3
J3 = -2.51e-6
J4 = -1.60e-6

C1 = -MU
C2 = -1.5 * MU * J2 * RE**2
C3 = -2.5 * MU * J3 * RE**3
C4 = 1.875 * MU * J4 * RE**4

# Helper functions
def skew(v):
    """Returns the skew-symmetric matrix of a vector v."""

    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def rotation_matrix(axis, angle):
    """Returns the rotation matrix for a specific axis and angle (in radians)."""

    I = np.eye(3)
    return (I + skew(axis) * np.sin(angle) + (1.0 - np.cos(angle)) * (np.outer(axis, axis) - I))

# Orbital elements to Cartesian state
def orbital_elements_to_state(alt_perigee, alt_apogee, raan, inclination, arg_perigee, true_anomaly):
    """Convert CADRE's orbital parameters into the initial Earth-centered inertial position and velocity. 
    Altitudes are in km. Angles are in degrees."""

    deg_to_rad = np.pi / 180.0

    # Convert altitude above Earth to orbital radius
    r_perigee = RE + alt_perigee
    r_apogee = RE + alt_apogee

    # Orbital eccentricity 
    e = ((r_apogee - r_perigee) / (r_apogee + r_perigee))

    # Semi-major axis
    a = (r_perigee + r_apogee) / 2.0

    # Semi-latus rectum
    p = a * (1 - e**2)

    # Convert angles from degrees to radians
    raan = raan * deg_to_rad
    inclination = inclination * deg_to_rad
    arg_perigee = arg_perigee * deg_to_rad
    true_anomaly = true_anomaly * deg_to_rad

    # Position in perifocal coordinate system
    r_mag = p / (1 + e * np.cos(true_anomaly))
    r_perifocal = np.array([r_mag * np.cos(true_anomaly),
                            r_mag * np.sin(true_anomaly),
                            0.0])

    # Velocity in perifocal coordinate system
    v_perifocal = np.array([-np.sqrt(MU / p) * np.sin(true_anomaly),
                            np.sqrt(MU / p) * (e + np.cos(true_anomaly)),
                            0.0])

    # Rotate from perifocal to Earth-centered inertial (ECI)
    z_axis = np.array([0.0, 0.0, 1.0])
    x_axis = np.array([1.0, 0.0, 0.0])

    R_raan = rotation_matrix(z_axis, raan)
    R_inc = rotation_matrix(x_axis, inclination)
    R_arg = rotation_matrix(z_axis, arg_perigee)
    rotation = (R_raan @ R_inc @ R_arg)
    r_eci = rotation @ r_perifocal
    v_eci = rotation @ v_perifocal

    return r_eci, v_eci

def orbit_dynamics(state):
    """Compute the time derivative of the spacecraft state
    state = [x, y, z, vx, vy, vz]
    returns = [vx, vy, vz, ax, ay, az]."""

    x = state[0]
    y = state[1]
    z = state[2]

    vx = state[3]
    vy = state[4]
    vz = state[5]

    # Distance from Earth's center
    r2 = x*x + y*y + z*z
    r = np.sqrt(r2)

    z2 = z*z
    z3 = z2 * z
    z4 = z3 * z

    r3 = r2 * r
    r4 = r3 * r
    r5 = r4 * r
    r7 = r5 * r2

    # J2, J3, J4 correction terms
    T2 = (1.0 - 5.0 * z2 / r2)
    T3 = (3.0 * z - 7.0 * z3 / r2)
    T4 = (1.0 - 14.0 * z2 / r2 + 21.0 * z4 / r4)

    common = C1/r3 + C2/r5*T2 + C3/r7*T3 + C4/r7*T4

    ax = common * x
    ay = common * y
    az = common * z

    # Additional z-direction effects
    az += 2.0*C2*z/r5
    az += C3/r7*(3.0*z2 - 0.6*r2)
    az += C4/r7*(4.0*z - (28.0/3.0)*z3/r2)

    return np.array([vx, vy, vz, ax, ay, az])

def rk4_step(state, dt):
    """Advance the spacecraft state forward by one time step using fourth-order Runge-Kutta integration.
    state = [x, y, z, vx, vy, vz]
    dt = time step in seconds"""
    k1 = orbit_dynamics(state)
    k2 = orbit_dynamics(state + 0.5 * dt * k1)
    k3 = orbit_dynamics(state + 0.5 * dt * k2)
    k4 = orbit_dynamics(state + dt * k3)

    next_state = state + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

    return next_state

def propagate_orbit(initial_state, dt, duration):
    """Propagate the spacecraft orbit over time.
    initial_state = initial [x, y, z, vx, vy, vz]
    dt = nominal time step [seconds]
    duration = total propagation time [seconds]"""

    num_full_steps = int(duration / dt)

    times = [0.0]
    states = [initial_state.copy()]

    # Propagate using full time steps
    for _ in range(num_full_steps):
        next_state = rk4_step(states[-1], dt)

        states.append(next_state)
        times.append(times[-1] + dt)

    # Take a smaller final step if necessary so that the trajectory ends exactly at "duration"
    remaining_time = duration - times[-1]

    if remaining_time > 1e-12:
        next_state = rk4_step(states[-1], remaining_time)

        states.append(next_state)
        times.append(duration)

    return np.array(times), np.array(states)

def calculate_orbital_period(alt_perigee, alt_apogee):
    """Calculate orbital period from perigee and apogee altitudes."""

    r_perigee = RE + alt_perigee
    r_apogee = RE + alt_apogee

    a = (r_perigee + r_apogee) / 2.0

    period = 2.0 * np.pi * np.sqrt(a**3 / MU)

    return period

# Test using CADRE's default orbit
if __name__ == "__main__":
    # CADRE default orbital parameters
    alt_perigee = 500.0
    alt_apogee = 500.0
    raan = 66.279
    inclination = 82.072
    arg_perigee = 0.0
    true_anomaly = 337.987

    # Initial position and velocity
    r0, v0 = orbital_elements_to_state(
        alt_perigee=alt_perigee,
        alt_apogee=alt_apogee,
        raan=raan,
        inclination=inclination,
        arg_perigee=arg_perigee,
        true_anomaly=true_anomaly
    )

    print()
    print("Initial position [km]:")
    print(r0)

    print()
    print("Initial velocity [km/s]:")
    print(v0)

    print()
    print("Position magnitude [km]:")
    print(np.linalg.norm(r0))

    print()
    print("Velocity magnitude [km/s]:")
    print(np.linalg.norm(v0))

    # Initial state and dynamics
    state0 = np.concatenate((r0, v0))
    state_dot = orbit_dynamics(state0)

    print()
    print("Initial state:")
    print(state0)

    print()
    print("Initial state derivative:")
    print(state_dot)

    # Orbit propagation
    dt = 10.0

    orbital_period = calculate_orbital_period(alt_perigee=alt_perigee, alt_apogee=alt_apogee)

    times, states = propagate_orbit(initial_state=state0, dt=dt, duration=orbital_period)

    print()
    print("Orbital period [s]:")
    print(orbital_period)

    print()
    print("Orbital period [min]:")
    print(orbital_period / 60.0)

    print()
    print("Number of time points:")
    print(len(times))

    print()
    print("Final propagation time [s]:")
    print(times[-1])

    # Propagation results
    print()
    print("Initial propagated state:")
    print(states[0])

    print()
    print("Final propagated state:")
    print(states[-1])

    # Initial vs. final position
    initial_position = states[0, 0:3]
    final_position = states[-1, 0:3]

    position_difference = np.linalg.norm(final_position - initial_position)

    print()
    print("Distance from starting position after one orbit [km]:")
    print(position_difference)

    # Orbit sanity checks
    position_magnitudes = np.linalg.norm(states[:, 0:3], axis=1)

    altitudes = position_magnitudes - RE

    print()
    print("Minimum altitude [km]:")
    print(np.min(altitudes))

    print()
    print("Maximum altitude [km]:")
    print(np.max(altitudes))

    # Plot trajectory
    x = states[:, 0]
    y = states[:, 1]
    z = states[:, 2]

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot orbit
    ax.plot(x, y, z, linewidth=2, label="Orbit")

    # Mark initial position
    ax.scatter(x[0], y[0], z[0], s=60, label="Initial Position")

    # Equal axis scaling
    max_range = np.max(np.abs(np.concatenate((x, y, z))))

    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    ax.set_box_aspect((1, 1, 1))

    # Labels and formatting
    ax.set_xlabel("X [km]", labelpad=10)
    ax.set_ylabel("Y [km]", labelpad=10)
    ax.set_zlabel("Z [km]", labelpad=10)

    ax.set_title("CADRE Orbit Reference Trajectory",pad=20)

    ax.legend()

    plt.tight_layout()
    plt.show()

    # Plot Earth
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)

    earth_x = RE * np.outer(np.cos(u), np.sin(v))

    earth_y = RE * np.outer(np.sin(u), np.sin(v))

    earth_z = RE * np.outer(np.ones_like(u), np.cos(v))

    ax.plot_surface(earth_x, earth_y, earth_z, alpha=0.25)