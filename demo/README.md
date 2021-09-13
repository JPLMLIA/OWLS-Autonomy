# OWLS Autonomy Demonstration Script

## Scope
This script is intended to provide a basic set of examples for each of the OWLS autonomy pipelines.  The `owls_demo.sh` script provided in this directory downloads a collection of raw data and processes it with the default processing configuration for each autonomy pipeline. This script is intended as a first activity for new users of the pipeline to ensure proper installation and provide a collection of example output products.


## Execution
1. Ensure the OWLS autonomy code base is installed by following the instruction in the [`README.md`](../README.md) at the root of this repository.

2. Run the demo by executing the following commands.
```bash
cd OWLS-Autonomy/demo
./owls_demo.sh
```
Note: This script utilizes the bash shell, which requires a \*nix system.

3. A new folder named `owls_demo_data` will be created, with subdirectories for each autonomy pipeline.  Raw input data and processed results will be available within the `owls_demo_data` folder at the completion of the script.  A `logs` folder will also be created with a processing log for each run of the demonstration script. 
- Read more about validation products in the validate [`README.md`](../src/helm_dhm/validate/README.md)
- Read more about tracker products in the tracker [`README.md`](../src/helm_dhm/tracker/README.md)
- Read more about feature products in the feature [`README.md`](../src/helm_dhm/features/README.md)
- Read more about classification products in the classifier [`README.md`](../src/helm_dhm/classifier/README.md)
- Read more about ASDP products in the ASDP [`README.md`](../src/helm_dhm/asdp/README.md)

4. This demonstration script provides a generic example of each pipeline.  If you would like to build upon these examples, please see the documentation in the CLI [`README.md`](../src/cli/README.md) for advanced usage examples and documentation.
