# Spectral

This computational tool simulates 3+1D deformations of nonlinear viscoelastic waveguides using the multidomain pseudospectral method. It is designed for modeling of nonlinear strain waves and solitons.


**Features**
- Simulate the evolution of initial waves and waves generated by pressure applied on the waveguide surface.
- Leverages a multidomain pseudospectral method for high accuracy and efficiency.
- Includes Jupyter notebooks for simulation and result visualization.

### Installation
1. Clone the repository
2. Create a Conda environment using the provided `spectral.yml` file.


### Usage
The repository provides two examples for simulating wave propagation:

1. **Evolution of an initial wave**  
   Use the notebook `SimulateIC` to simulate the evolution of an initial wave.

2. **Wave propagation generated by impact**  
   Use the notebook `SimulateImpact` to model waves resulting from an impact (sudden pressure applied to the waveguide's end).

To **visualize** the simulation results, refer to the `ResultsVisualization*` notebooks, which demonstrate post-processing and graphical representation of the data.


## Example Workflow

1. Open and run the simulation notebook (`SimulateIC` or `SimulateImpact`) to generate the results.
2. Use the corresponding `ResultsVisualization*` notebook to analyze and visualize the output.
