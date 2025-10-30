# Quick Start

```bash

# Verifiy GPU driver and CUDA (WSL)
nvidia-smi
nvcc --version  # e.g. 11.8 (if newer, modify PyTorch installation command)

# Minimal dependencies with GPU/CUDA availability (Linux / WSL2)
python -m pip install tensorflow[and-cuda] tensorflow-datasets pillow numpy matplotlib seaborn scikit-learn tqdm
python -m pip install ipykernel  # Jupyter kernel support for venv/conda
python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install torchinfo torchviz  # Model visualization libraries

# Verify GPU setup
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Register venv as Jupyter kernel
python -m ipykernel install --user --name=venv --display-name="Python (satellite-segmentation)"

# if relevant: initialize (general) your general conda working environment
# conda activate working

```