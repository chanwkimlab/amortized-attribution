# vit-base on imagenette
# train classifier
CUDA_VISIBLE_DEVICES=2 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette  \
python train_classifier.py configs/vitbase_imagenette.json

# train surrogate (GPU util: train=80%, val=50%)
CUDA_VISIBLE_DEVICES=2 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate \
python train_surrogate.py configs/vitbase_imagenette_surrogate.json

CUDA_VISIBLE_DEVICES=4,5,6,7 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_eval \
python train_surrogate.py configs/vitbase_imagenette_surrogate_eval.json

# train explainer 
CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_explainer \
python train_explainer.py configs/vitbase_imagenette_explainer.json

CUDA_VISIBLE_DEVICES=4,5,6,7 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_explainer_precomputed \
python train_explainer_precomputed.py configs/vitbase_imagenette_explainer_precomputed.json

CUDA_VISIBLE_DEVICES=4,5,6,7 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_explainer_objective \
python train_explainer_objective.py configs/vitbase_imagenette_explainer_objective.json

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_explainer_regression \
python train_explainer_regression.py configs/vitbase_imagenette_explainer_regression.json

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_explainer_regression_0 \
python train_explainer_regression.py configs/vitbase_imagenette_explainer_regression_0.json

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_explainer_regression_512 \
python train_explainer_regression.py configs/vitbase_imagenette_explainer_regression_512.json

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_explainer_regression_1024 \
python train_explainer_regression.py configs/vitbase_imagenette_explainer_regression_1024.json




CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_segexplainer \
python train_segexplainer.py configs/segexplainer.json

# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# WANDB_PROJECT=xai-amortization \
# WANDB_NAME=segment \
# python train_segmentation.py segment.json

# resnet50 on imagenette
CUDA_VISIBLE_DEVICES=2 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=resnet50_imagenette \
python train_classifier.py configs/resnet50_imagenette.json      

CUDA_VISIBLE_DEVICES=2 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=resnet50_imagenette_surrogate \
python train_surrogate.py configs/resnet50_imagenette_surrogate.json