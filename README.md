# CADRE-AMIGO

This repo contains my work reproducing parts of the OpenMDAO CADRE CubeSat model in AMIGO.

The main goal is to take the original CADRE subsystems, understand what each one is doing, recreate them in Python/AMIGO, and check that the results match the original implementation before combining everything together.

## Current Progress

So far I have worked through:

- Orbit
- Attitude
- Sun position and eclipse
- Solar exposed area
- Thermal
- Power

## Folder Layout

```text
cadre-amigo/
├── orbit/
├── attitude/
├── sun/
├── solar/
├── thermal/
├── power/
├── CMakeLists.txt
└── README.md
```

Each folder contains the reference version of the subsystem and, where completed, the AMIGO version and validation scripts.

## Orbit

The orbit model calculates spacecraft position and velocity over time.

I kept the same CADRE orbital dynamics, including the J2, J3, and J4 perturbation terms.

The main difference is in how the trajectory is solved:

- CADRE reference: RK4 time propagation
- AMIGO version: trapezoidal collocation

I also ran a convergence study. When the timestep was cut in half, the position error decreased by about a factor of four, which is consistent with second-order convergence.

## Attitude

The attitude model uses the orbit position and velocity to determine spacecraft orientation.

It calculates things like:

- rotation matrices
- angular velocity
- angular acceleration
- torque
- body-frame velocity

When I give the AMIGO/reference attitude calculations the same orbit data, the results match to essentially machine precision.

## Sun

The Sun model determines where the Sun is relative to the spacecraft and whether the spacecraft is in sunlight or eclipse.

It calculates:

- Sun direction in inertial coordinates
- Sun direction in the spacecraft body frame
- azimuth and elevation
- line-of-sight/eclipse value

The line-of-sight value is:

```text
1 = sunlight
0 = eclipse
between 0 and 1 = eclipse transition
```

The reproduced Sun calculations match the original CADRE results.

## Solar

The solar model uses the Sun angles and panel fin angle to determine how much of each solar cell is exposed.

CADRE uses precomputed geometry data for 12 panels with 7 cells per panel.

One issue here is that the original CADRE code uses an old package called MBI for multidimensional spline interpolation. My first version used SciPy linear interpolation, so this part is not yet an exact replacement for the original CADRE solar interpolation.

## Thermal

The thermal model tracks five temperatures:

- four fin temperatures
- one spacecraft body temperature

The temperatures change based on solar heating, radiative cooling, and communication power.

The original CADRE version uses RK4 integration. The AMIGO version uses trapezoidal constraints.

The AMIGO result matches the trapezoidal reference exactly, and stays within about 0.3 K of the original CADRE RK4 result.

## Power

The power model uses:

- temperature
- illuminated solar-cell area
- current setpoint

to calculate solar-cell voltage.

The original CADRE code uses the old MBI interpolation package for this lookup. I recreated the needed MBI behavior in modern Python.

The first linear interpolation version had a maximum panel-voltage error of about 3.33 V.

After recreating the MBI spline behavior, the maximum voltage difference dropped to about:

```text
0.00054 V
```

The solar-power and total-power equations were then reproduced in AMIGO.

Current power results:

```text
AMIGO vs Python solar power difference = 0.0 W
AMIGO vs Python battery power difference = 0.0 W

AMIGO vs original CADRE solar power difference ≈ 0.00012 W
AMIGO vs original CADRE battery power difference ≈ 0.00012 W
```

The small remaining difference comes from the recreated MBI interpolation.

## Validation Approach

For each subsystem, I have generally been following this process:

```text
Original CADRE code
        ↓
Python reference
        ↓
Check against CADRE
        ↓
AMIGO version
        ↓
Check AMIGO against reference
```

Doing the subsystems separately makes it easier to find where differences are coming from before everything is coupled together.

## Next Steps

The next things I plan to work on are:

- battery/state-of-charge model
- improve the MBI recreation where needed
- move more of the lookup/interpolation behavior directly into AMIGO
- couple the individual subsystems into one full model

The long-term goal is to reproduce the full CADRE multidisciplinary model in AMIGO and then use it for coupled analysis and optimization.

