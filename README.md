# Spectral

Computational tool for 3+1D simulation of deformations of nonlinear elastic (viscoelastic) body using multidomain pseudospectral method.

The details of the method will be published soon in AIP proceedings of ICNAAM 2021.

Some simulations obtained with this code are available [here](https://www.researchgate.net/publication/356152589_Formation_of_long_strain_waves_in_viscoelastic_bar_subjected_to_a_longitudinal_pulse_load).

**Requirements**: Python 3.6 or higher

**Usage**: run `python simulation.py` with 7 arguments: material file (elastic moduli), body file (geometry and mesh), impact file (amplitude of impact and time-width), output filename, simulation time (in microseconds) and time step.

For example, run two commands:
- `python simulation.py params/ps_retarded_nonlin params/rect_bar_225_10_10_medium params/impact_medium test_retarded 90 1`
- `python simulation.py params/ps_increased_young params/rect_bar_225_10_10_medium params/impact_medium test_lin 90 1`

This tool will simulate strains from the impact on the end of the rectangular polystyrene bar 225 mm long and 10 mm thick.
Then execute all cells (except the last two) in the ResultsVisualization notebook to see the results.
