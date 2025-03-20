# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Update package list and install Python3, pip, and other dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1 \
    libegl1 \
    libglib2.0-0 \
    libxrender1 \
    git \
    wget && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip3 install --upgrade pip

RUN pip install torch && \
pip install torch_geometric && \
pip install lightning && \
pip install hydra-core && \
pip install termcolor && \
pip install wandb && \
pip install mlflow && \
pip install pydantic && \
pip install imageio && \
pip install moviepy && \
pip install tqdm && \
pip install omegaconf && \
pip install hydra-core && \
pip install numpy && \
pip install pandas && \
pip install vtk && \
pip install matplotlib && \
pip install pip && \
pip install flake8 && \
pip install coverage && \
pip install sphinx && \
pip install pytest && \
pip install black && \
pip install docutils && \
pip install sphinx_copybutton && \
pip install sphinxcontrib.katex && \
pip install opencv-python && \
pip install furo

