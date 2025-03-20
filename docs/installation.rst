.. highlight:: shell

============
Installation
============


* It is recommended to use a conda virtual environment for this project. Conda can be installed following these instructions_. After successfully installing conda, the environment for this project can be created and activated:

.. _instructions: https://docs.conda.io/en/latest/miniconda.html

.. code-block:: bash

    conda create --name lbm python=3.12
    conda activate lbm

* After activating the environment, install PyTorch_ as described in the link.

.. _PyTorch: https://pytorch.org

* Finally, clone this repository using

.. code-block:: bash

    git clone git@gitlab.lrz.de:qcaer/torchlbm.git # For ssh
    git clone https://gitlab.lrz.de/qcaer/torchlbm.git # For https

* and install the ``torchlbm`` package. Therefore, go to the folder of this repository and use
pip to install. For development, we recommend installing the `torchlbm` package in place. This can be done using the `-e` option for the pip install command.

.. code-block:: bash

    cd torchlbm
    pip install -e .

* For development purposes, additional packages are required. They can be installed using the ``requirements_dev.txt`` file:

.. code-block:: bash

    pip install -r requirements_dev.txt

* With the successfully installed package, the [example cases](cases) are a good starting point.

* In case a GPU is available, ``torchlbm`` uses it automatically. Nevertheless, it is possible to switch to CPU mode, or to select a specific GPU. This can be done using the following commands:

.. code-block:: bash

    export CUDA_VISIBLE_DEVICES="" # For CPU mode
    export CUDA_VISIBLE_DEVICES="number" # For GPU mode, where number is the ID of the GPU that should be used
