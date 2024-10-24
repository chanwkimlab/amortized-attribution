# Stochastic Amortization for feature attributions
This repository implements an amortized approach to feature attribution from [this paper](https://arxiv.org/abs/2401.15866) published in NeurIPS 2024. The code for data valuation is available in [this repo](https://github.com/iancovert/amortized-valuation).

## Installation

```bash
conda create -n xai-amortization python=3.9
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install transformers==4.32.0 evaluate==0.4.0 datasets==2.14.3 accelerate==0.20.3 scikit-learn==1.3.0
pip install git+https://github.com/huggingface/transformers@e9ad51306fdcc3fb79d837d667e21c6d075a2451
pip install fsspec==2023.6.0
pip install jupyterlab jupyter_contrib_nbextensions wandb ipdb gpustat seaborn
ipython kernel install --name "xai-amortization" --user

# Set up git filter if you want to strip the output of Jupyter notebooks before committing
git config filter.strip-notebook-output.clean 'jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR'
```


## Usage


### Train amortization models
*  `scripts/imagenette/classifier.sh`: train classifiers.
* `scripts/imagenette/shapley.sh`: train amortization models for Shapley value feature attribution.
* `sripts/imagenette/lime.sh`: train amortization models for LIME feature attribution.
* `scripts/imagenette/banzhaf.sh`: train amortization models for Banzhaf feature attribution.

### Evaluate amortization models
* `notebooks/visualize_results.ipynb`: evaluate and visualize the results of the amortization models.

## Note

This repository was inspired by the following repositories:

* [FastSHAP](https://github.com/iancovert/fastshap/tree/main/fastshap)
* [ViT-Shapley](https://github.com/suinleelab/vit-shapley)
* [Huggingface example - Image classification](https://github.com/huggingface/transformers/tree/149cb0cce2df3f932de58c6d05cec548600553e2/examples/pytorch/image-classification)