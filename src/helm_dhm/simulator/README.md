# DHM Data Simulator
The DHM simulator creates synthetic digital holographic microscopy data. This
tool is meant to be used in a two-step process:
1. simulate particle tracks (specifying position, velocity through time) and then
2. simulate DHM images from tracks (creating 2D tif hologram images)

## CLI
To run the simulator, specify a configuration file (or files), the number of
experiments to simulate, and an output directory to save results to.

```bash
# Single config
python src/cli/HELM_simulator.py \
--configs src/cli/configs/helm_simulator_config.yml \
--n_exp 2 \
--sim_outdir <local_output_dir>

# Multiple configs
python src/cli/HELM_simulator.py \
--configs src/cli/configs/sim_config_v*.yml \
--n_exp 2 \
--sim_outdir <local_output_dir>
```

## Config
Within the configuration file, you can set items like
* image parameters (e.g., resolution, chamber size, etc.)
* experiment parameters (number of motile/non-motile particles, length of recording, drift, etc.)
* particle parameters (e.g., shape/size/brightness of particle, movement distribution, etc.)
