# FaceDuplicationFilter

[中文](./README_CN.md)

## Content

- [FaceDuplicationFilter](#faceduplicationfilter)
  - [Content](#content)
  - [Project Overview](#project-overview)
  - [Technical Innovations](#technical-innovations)
  - [References](#references)
  - [System Requirements](#system-requirements)
  - [Environment Setup and Usage](#environment-setup-and-usage)
    - [Method 1: Complete Package](#method-1-complete-package)
    - [Method 2: Conda Installation](#method-2-conda-installation)
  - [Usage](#usage)
  - [License](#license)

## Project Overview

This project is based on the [eDifFIQA](https://github.com/LSIbabnikz/eDifFIQA) model, combining its face quality assessment capabilities to develop a face quality-enhanced duplicate data filtering system for identifying and removing duplicate face images.

Face recognition technology has long been a hot topic in cutting-edge academic research. With the advancement of AI technology, various face recognition techniques heavily rely on large-scale training datasets. However, these datasets often contain significant redundant data which, if not properly filtered, may become high-frequency noise that affects model training. This could lead to suboptimal training results, slower convergence, or even abnormal gradient changes during training. Therefore, establishing a reliable face data cleaning system is crucial.

Our system addresses this need by utilizing a carefully trained face quality assessment model to effectively extract the most distinctive face data while filtering out low-distinctiveness redundant data. This process not only improves training efficiency but also significantly accelerates model convergence, thereby enhancing overall system performance and reliability.

[Program Flowchart Here]

## Technical Innovations

1. **K-Fold Cross Validation for Optimal Face Selection**:

   - We employ k-fold cross validation to obtain optimal face combinations
   - The evaluation criteria is based on distances between faces
   - Particularly effective for our product's scenario of small-batch, high-duplication applications

2. **Enhanced Face Representation with eDifFIQA**:

   - Compared to traditional average distance methods, we introduce the eDifFIQA model for quality assessment
   - Use quality-weighted averages as face representation
   - Considers factors like face angle, noise, brightness, and camera distortion

3. **Advanced Quality Assessment Methodology**:
   - Diffusion process using a custom UNet model for generating noisy and reconstructed images
   - Process repeated on horizontally flipped images to capture pose impact
   - Quality score calculation through embedding comparison
   - Enhanced with knowledge distillation and label optimization:
     - Quality label optimization using relative position information from FR model embedding space
     - Representation consistency loss (Lrc) and quality loss (Lq) for improved prediction

## References

```bibtex
@article{babnikTBIOM2024,
  title={{eDifFIQA: Towards Efficient Face Image Quality Assessment based on Denoising Diffusion Probabilistic Models}},
  author={Babnik, {\v{Z}}iga and Peer, Peter and {\v{S}}truc, Vitomir},
  journal={IEEE Transactions on Biometrics, Behavior, and Identity Science (TBIOM)},
  year={2024},
  publisher={IEEE}
}
```

## System Requirements

- Windows environment (Python embedded package)
- Linux environment (theoretically supported, requires Python configuration)

## Environment Setup and Usage

### Method 1: Complete Package

1. Download the complete package via HuggingFace link[Download](https://huggingface.co/scolenchris/FaceDuplicationFilter/blob/main/DJ_folder_main1.zip)
2. Extract the package
3. Run `start.bat` on Windows

### Method 2: Conda Installation

1. Create a Python 3.10 environment
2. Install packages from `requirements.txt` in the project root
3. Download model weights according to instructions from [eDifFIQA](https://github.com/LSIbabnikz/eDifFIQA):
   - Recommended weights: `r100.pth` and `ediffiqaL.pth`
   - Place them in the `weights` folder
4. Run `allmain.py`

## Usage

[Add usage instructions here if needed]

## License

Follow the original project [eDifFIQA](https://github.com/LSIbabnikz/eDifFIQA), open source licenses.
