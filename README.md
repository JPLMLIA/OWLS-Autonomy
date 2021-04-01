# Ocean Worlds Life Surveyor autonomous algorithms

⚠️ **Installation instructions have changed as of 2021/03/18**

## Scope
The Ocean Worlds Life Surveyor (OWLS) project is aimed at autonomously detecting signatures of life in water at the molecular and cellular scale. The ACME, FAME, HELM, and JEWEL components of OWLS (this package) are a set of software-based autonomy tools to process data from different instruments aboard the OWLS instrument. Their goal is to generate and prioritize Autonomous Data Science Products (ASDPs) to send only the most scientifically valuable subset of information back to Earth.

* ACME: Autonomous CE-ESI Mass-spectra Examination. Extracts ASDPs from mass-spectra data.
* FAME: FLFM Autonomous Motility Evaluation. Extracts ASDPs from Volumetric Flourescent Imaging (VFI) microscopy data.
* HELM: Holographic Examination for Life-like motility. Extracts ASDPs from Digital Holography Microscopy (DHM) data.
* JEWEL: Joint Examination for Water-based Extant Life. Prioritizes ASDPs from all instruments to transmit data back to Earth in an "optimal" order.

## Installation

### Standard install requirements
1. **ImageMagick** \
  Full instructions for installing ImageMagick are
  [here](https://imagemagick.org/script/download.php). For MacOSX, use
  `brew install imagemagick`.
2. **ffmpeg** \
    Full instructions for installing ffmpeg are
    [here](https://ffmpeg.org/download.html). For MacOSX, use
    `brew install ffmpeg@4`.

### Python code
```bash
git clone https://github.com/JPLMLIA/OWLS-Autonomy.git
cd OWLS-Autonomy
pip install -e .  # use 'editable' to avoid repeated pip installs
```

### Development tools
For development or CI builds (e.g., on Jenkins).
```bash
cd OWLS-Autonomy
pip install -r requirements_dev.txt
```

## Code Usage
For ACME, HELM, and FAME, data processsing scripts that you can run from the
command line are located in the `src/cli` directory. Please see
the CLI [`README.md`](./src/cli/README.md) for examples.

## Docker
You can build a docker image from the source code if you can't pip install the
repository. This may be useful for Windows users.

### HELM Validate step
You can build a Docker image for using the HELM validate tool with the
 `Dockerfile_validate` Dockerfile located in this directory. Navigate a terminal
 to this dir and run the command below

```
# Build with `Dockerfile` and tag the image. Change the version of the tag as needed.
docker build . -f Dockerfile_validate --tag owls/validate:v1

# Run a container in the [Docker Dashboard GUI](https://docs.docker.com/desktop/dashboard/)
# or via the CLI. Be sure to mount your local data directory into the Docker
# container. The CLI approach is shown below. After starting the container, see the
# CLI [`README.md`](./src/cli/README.md) for examples on running the HELM validate
# step.
docker run -it --volume /path/to/data/locally:/path/to/data/docker owls/validate:v1 /bin/bash
```

## License
All software is licensed according to the [JPL Software License](LICENSE.md), while all documentation is Caltech/JPL Proprietary - Not for public release or distribution.
