# Satellite Image Classification

This repo implements training on Sentinel-2 satellite image classification tasks using a selection of key techniques from the recent literature. The image data is obtained via the publically available EuroSAT dataset. 

After training, the best checkpoint of the smallest available ResNet model, i.e. ResNet18, is able to reach an average precision of ~98%. This result closely approaches the performance of the large ResNet50 model cited in the original EuroSAT paper (98.57%) while using significantly fewer parameters.

## Results

![cm](results/figures/confusion_matrices.png)

![loss_curves](results/figures/loss_curves.png)

![lr_schedule](results/figures/lr_schedule.png)

---

**Classification Report**: Best Model Checkpoint (Scikit-Learn)

|                 | precision     | recall      | f1-score      | support       |
|-----------------|----------|----------|----------|----------|
| accuracy            | -   | -        | **0.9802**        | 4050        |
| macro avg   | 0.9799        | 0.9794   | 0.9797   | 4050        |
| weighted avg        | **0.9803**        | 0.9802   | 0.9802   | 4050        |

                               

                                           
                              
                           

## Technique (Selection)

- **Transfer Learning:** **Finetuning modified ResNet18**
- **Learning Rate Scheduling** (**OneCycleLR**)
- **Mixed Precision Training**
- **Label Smoothing**
- **Data Augmentation**
- **Physics-Informed Data Augmentation** i.e. **Custom Hue Transform**

## Directory tree

```
.
├── data
│   └── samples
│   	└── eurosat
│   		└── *.jpg          		# Sample selection for visualization
├── notebooks
│   ├── data_processing.ipynb  		# Data pre-processing
│   └── train_eval.ipynb	   		# Complete training and model eval implementation
├── references                      
│   └── eurosat_helber_etal.pdf		# EuroSAT (Helber 2019)
├── results
│   └── figures
│		├── loss_curves.png 		# Training and Validation loss
│		├── lr_schedule.png			# Backbone/Classifier learning rate
│   	└── confusion_matrices.png	# Checkpoint eval via confusion matrices
├── README.md
└── .gitignore
```

## Quick Start

```bash
# Verifiy GPU driver and CUDA (WSL)
nvidia-smi
nvcc --version  # e.g. 11.8 (if newer, modify PyTorch installation command)

# Setup Python virtual environment
python -m venv venv
source venv/bin/activate  # Linux/WSL

# Minimal dependencies with GPU/CUDA availability (Linux / WSL2)
python -m pip install tensorflow[and-cuda] tensorflow-datasets pillow numpy matplotlib seaborn scikit-learn tqdm
python -m pip install ipykernel  # Jupyter kernel support for venv/conda
python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install torchinfo torchviz  # Model visualization libraries

# Verify GPU setup
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Register venv as Jupyter kernel
python -m ipykernel install --user --name=venv --display-name="Python (satellite-segmentation)"

# if relevant: initialize your (general) general conda working environment
conda activate my_conda_env
```

## References

EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification. Patrick Helber, Benjamin Bischke, Andreas Dengel, Damian Borth. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019

