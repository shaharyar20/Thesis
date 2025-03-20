FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

RUN apt-get update && apt-get install -y libgl1 && apt-get install -y libegl1 && apt-get install -y libglib2.0-0 && apt-get install -y libxrender1 && apt-get install -y git

RUN pip install nvidia-modulus && \
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

