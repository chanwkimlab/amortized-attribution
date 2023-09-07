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

CUDA_VISIBLE_DEVICES=4,5,6,7 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_eval_train \
python train_surrogate.py configs/vitbase_imagenette_surrogate_eval_train.json

python calculate_shapley.py \
--input_path logs/vitbase_imagenette_surrogate_eval_train/extract_output/train \
--batch_size 512 \
--normalize_function softmax \
--num_players 196

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_eval_validation \
python train_surrogate.py configs/vitbase_imagenette_surrogate_eval_validation.json

python calculate_shapley.py \
--input_path logs/vitbase_imagenette_surrogate_eval_validation/extract_output/validation \
--batch_size 512 \
--normalize_function softmax \
--num_players 196

CUDA_VISIBLE_DEVICES=4,5,6,7 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_eval_test \
python train_surrogate.py configs/vitbase_imagenette_surrogate_eval_test.json

python calculate_shapley.py \
--input_path logs/vitbase_imagenette_surrogate_eval_test/extract_output/test \
--batch_size 512 \
--normalize_function softmax \
--num_players 196


# train explainer
CUDA_VISIBLE_DEVICES=4,5,6,7 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_objexplainer \
python train_objexplainer.py configs/vitbase_imagenette_objexplainer.json



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