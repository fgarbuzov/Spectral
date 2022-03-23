# Spectral

Computational tool for 3+1D simulation of deformations of nonlinear elastic (viscoelastic) body using multidomain pseudospectral method.

The details of the method will be published soon in AIP proceedings of ICNAAM 2021.

Some simulations obtained with this code are available [here](https://www.researchgate.net/publication/356152589_Formation_of_long_strain_waves_in_viscoelastic_bar_subjected_to_a_longitudinal_pulse_load).

**Requirements**: Python 3.6 or higher

**Usage**: run `python simulation.py` with 7 arguments: material file (elastic moduli), body file (geometry and mesh), impact file (amplitude of impact and time-width), output filename, simulation time (in microseconds) and time step.

For example, run the following two commands to simulate strains from the impact on the end of the rectangular polystyrene bar 225 mm long and 10 mm thick:
- `python simulation.py params/ps_retarded_nonlin params/rect_bar_225_10_10_medium params/impact_medium test_retarded 90 1`
- `python simulation.py params/ps_increased_young params/rect_bar_225_10_10_medium params/impact_medium test_lin 90 1`

In the first simulation the bar is viscoelastic ([generalized Maxwell model](https://en.wikipedia.org/wiki/Generalized_Maxwell_model)) and nonlinearly elastic ([Murnaghan elastic energy](https://en.wikipedia.org/wiki/Acoustoelastic_effect#Non-linear_elastic_theory_for_hyperelastic_materials)), while in the second simulation the bar is linear and absolutely elastic. The simulation results are saved in `simulation_results` directory.

To see the computed strains enter the name of the simulation results files in the ResultsVisualization notebook and execute all but the last two cells.
