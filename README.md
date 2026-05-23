
Code for the research paper "Approaching physical limits on latent dimensionality in optical computing".
## System Requirements

### Core Dependencies
- **Python**: Version 3.8
- **Python Packages**:
  - `splayout` == 0.5.16
  - `numpy` == 1.24.3
  - `torch` == 2.3.0
  - `torchvision` == 0.18.0
  - `tqdm` == 4.46.4

### Required Software
- **Ansys Lumerical**: Version 2024 R1 or newer

## Execution Instructions
#### To run the iris classification task (Default on GPU): 
```bash
# go to the iris_classification directory
python inference.py
```
#### To run the digits classification task (On CPU, Require 64GB RAM to run): 
```bash
# go to the digits_classification directory
python inference.py
```

#### To run the generative task (Default on GPU, Require 24GB GPU RAM to run):
```bash
# first, install the package:
python setup.py install
# go to the mnist_diffusion_gen directory
python train.py
```
## Datasets

- **Iris Dataset**  
  The dataset located at `datasets/iris` is a standardized version of the classic Iris dataset:  
  Fisher, R. A. *Iris*. UCI Machine Learning Repository. https://doi.org/10.24432/C56C76 (1988).  

- **Optical Recognition of Handwritten Digits Dataset**  
  The dataset located at `datasets/ocr` is a standardized version of the Optical Recognition of Handwritten Digits dataset:  
  Alpaydin, E. & Kaynak, C. *Optical Recognition of Handwritten Digits*. UCI Machine Learning Repository. https://doi.org/10.24432/C50P49 (1998).  

## Citation (Preprint)
To be added.

## Project Structure
```shell
code/
│
├── datasets/                       # directory for datasets
│
├── mml/                            # implementation for photonic layers
│   ├── torchmodel.py               
│   └── utils.py                    
│
├── iris_classification/            # directory for iris classification task        
│   ├── mux-demux                   # transfer functions for mux and demux
│   ├── iris.gds                    # structures for the iris classification task
│   └── inference.py                # script to run the iris classification task
│
├── digits_classification/          # directory for digits classification task         
│   ├── digits.gds                  # structures for the digits classification task
│   └── inference.py                # script to run the digits classification task
│
├── mnist_diffusion_gen/            # directory for generative task         
│   ├── network.py                  # network definition of the generative model
│   └── train.py                    # script to train the generative model  
│
├── setup.py                        # setup for the package of implementation for photonic layers
├── LICENSE                         # license
└── README.md                       # project overview
```
