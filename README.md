![image](docs/_static/Torchlbm_logo.svg)
# Lattice Boltzmann Simulation Framework Based on PyTorch

## Features

-   Fully differentiable implementation of the Lattice Boltzmann method
    using PyTorch
-   Usability features of PyTorch (Logging, Profiling, Graphical
    visualization of models) can be used
-   Implementation based on modules allows easy integration with
    machine-learning modules
-   Implementation degenerates to two- and one-dimensional setups
-   Simulation setup offers interface to XML- and JSON-based
    configuration files
-   Runs on GPUs. Currently, we only provide a single GPU
    implementation, but plan to extend it to multiple GPUs.

## Installation

-   It is recommended to use a conda virtual environment for this
    project. Conda can be installed following these
    [instructions](https://docs.conda.io/en/latest/miniconda.html).
    After successfully installing conda, the environment for this
    project can be created and activated:

```bash
conda create --name lbm python=3.12
conda activate lbm
```

-   After activating the environment, install
    [PyTorch](https://pytorch.org) as described in the link.
-   Finally, clone this repository using

```bash
git clone git@gitlab.lrz.de:qcaer/torchlbm.git # For ssh
git clone https://gitlab.lrz.de/qcaer/torchlbm.git # For https
```

and install the `torchlbm` package. Therefore, go to the folder of
this repository and use pip to install. For development, we recommend
installing the [torchlbm]{.title-ref} package in place. This can be done
using the [-e]{.title-ref} option for the pip install command.

```bash
cd torchlbm
pip install -e .
```

-   For development purposes, additional packages are required. They can
    be installed using the `requirements_dev.txt` file:

```bash
pip install -r requirements_dev.txt
```

-   With the successfully installed package, the \[example
    cases\](cases) are a good starting point.
-   In case a GPU is available, `torchlbm` uses it automatically.
    Nevertheless, it is possible to switch to CPU mode, or to select a
    specific GPU. This can be done using the following commands:

```bash
export CUDA_VISIBLE_DEVICES="" # For CPU mode
export CUDA_VISIBLE_DEVICES="number" # For GPU mode, where number is the ID of the GPU that should be used
```

## Example simulations
![image](docs/_static/karman.png) ![image](docs/_static/droplet.png)
![image](docs/_static/car.png)

## Documentation

The documentation of `torchlbm` can be build using the Makefile.
Therefore, go to the folder of `torchlbm` and use the Makefile:

```bash
cd /path/to/torchlbm
make docs
```

Note, to correctly build the documentation, the development requirements
of `torchlbm` have to be installed as explained in the installation
instructions.
