# vit-base on imagenette
# train classifier
CUDA_VISIBLE_DEVICES=3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_classifier  \
python train_classifier.py configs/vitbase_imagenette_classifier.json

# train surrogate (GPU utilization: train=80%, val=50%)
CUDA_VISIBLE_DEVICES=3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate \
python train_surrogate.py configs/vitbase_imagenette_surrogate.json