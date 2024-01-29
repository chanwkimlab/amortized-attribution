# Evaluate surrogate model 

## surrogate evaluate (train set)

CUDA_VISIBLE_DEVICES=4,5,6,7 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_shapley_eval_train \
python train_surrogate.py configs/vitbase_imagenette_surrogate_shapley_eval_train.json

## get feature attribution (train set - KernelSHAP)

python calculate_feature_attribution_using_extracted.py \
--input_path logs/vitbase_imagenette_surrogate_shapley_eval_train/extract_output/train \
--batch_size 512 \
--normalize_function softmax \
--num_players 196

## get feature attribution  (train set - permutation)

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_shapley_eval_train_permutation \
python calculate_feature_attribution.py configs/vitbase_imagenette_surrogate_shapley_eval_train_permutation.json

## get feature attribution  (train set - SGD-Shapley)

CUDA_VISIBLE_DEVICES=4,5,6 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_shapley_eval_train_SGD_antithetical \
python calculate_feature_attribution.py configs/vitbase_imagenette_surrogate_shapley_eval_train_SGD_antithetical.json

## surrogate evaluate (validation set)

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_eval_validation \
python train_surrogate.py configs/vitbase_imagenette_surrogate_eval_validation.json


## get feature attribution (validation set - KernelSHAP)

python calculate_feature_attribution_using_extracted.py \
--input_path logs/vitbase_imagenette_surrogate_eval_validation/extract_output/validation \
--batch_size 512 \
--normalize_function softmax \
--num_players 196

## get feature attribution (validation set - permutation)

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_eval_validation \
python calculate_feature_attribution.py configs/vitbase_imagenette_surrogate_eval_validation_permutation.json

## get feature attribution (validation set - SGD-Shapley)

CUDA_VISIBLE_DEVICES=4,5,6 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_shapley_eval_validation_SGD_antithetical \
python calculate_feature_attribution.py configs/vitbase_imagenette_surrogate_shapley_eval_validation_SGD_antithetical.json


## surrogate evaluate (test set)

CUDA_VISIBLE_DEVICES=4,5,6,7 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_eval_test \
python train_surrogate.py configs/vitbase_imagenette_surrogate_eval_test.json

## get feature attribution (test set - KernelSHAP)

python calculate_feature_attribution_using_extracted.py \
--input_path logs/vitbase_imagenette_surrogate_eval_test/extract_output/test \
--batch_size 512 \
--normalize_function softmax \
--num_players 196

### ground truth

CUDA_VISIBLE_DEVICES=1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_shapley_eval_train_regression \
python calculate_feature_attribution.py configs/vitbase_imagenette_surrogate_shapley_eval_train_regression.json

CUDA_VISIBLE_DEVICES=0,1,2 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_shapley_eval_test_regression \
python calculate_feature_attribution.py configs/vitbase_imagenette_surrogate_shapley_eval_test_regression.json


# Train AO models

## train explainer (KernelSHAP)
CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_shapley_regexplainer_upfront_512 \
python train_regexplainer.py configs/vitbase_imagenette_shapley_regexplainer_upfront_512.json

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_shapley_regexplainer_upfront_1024 \
python train_regexplainer.py configs/vitbase_imagenette_shapley_regexplainer_upfront_1024.json

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_shapley_regexplainer_upfront_1536 \
python train_regexplainer.py configs/vitbase_imagenette_shapley_regexplainer_upfront_1536.json

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_shapley_regexplainer_upfront_2048 \
python train_regexplainer.py configs/vitbase_imagenette_shapley_regexplainer_upfront_2048.json

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_shapley_regexplainer_upfront_2560 \
python train_regexplainer.py configs/vitbase_imagenette_shapley_regexplainer_upfront_2560.json

CUDA_VISIBLE_DEVICES=4,5,6,7 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_shapley_regexplainer_upfront_3072 \
python train_regexplainer.py configs/vitbase_imagenette_shapley_regexplainer_upfront_3072.json

## train explainer (Permutation sampling)

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_shapley_regexplainer_permutation_upfront_196 \
python train_regexplainer.py configs/vitbase_imagenette_shapley_regexplainer_permutation_upfront_196.json

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_shapley_regexplainer_permutation_upfront_392 \
python train_regexplainer.py configs/vitbase_imagenette_shapley_regexplainer_permutation_upfront_392.json

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_shapley_regexplainer_permutation_upfront_588 \
python train_regexplainer.py configs/vitbase_imagenette_shapley_regexplainer_permutation_upfront_588.json

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_shapley_regexplainer_permutation_upfront_1176 \
python train_regexplainer.py configs/vitbase_imagenette_shapley_regexplainer_permutation_upfront_1176.json

CUDA_VISIBLE_DEVICES=4,5,6,7 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_shapley_regexplainer_permutation_upfront_3136 \
python train_regexplainer.py configs/vitbase_imagenette_shapley_regexplainer_permutation_upfront_3136.json

## train explainer (Shapley-SGD)

CUDA_VISIBLE_DEVICES=1,2,3,5 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_shapley_regexplainer_SGD_antithetical_upfront_9986 \
python train_regexplainer.py configs/vitbase_imagenette_shapley_regexplainer_SGD_antithetical_upfront_9986.json


## train explainer (compute match)

python calculate_feature_attribution_using_extracted.py \
--input_path logs/vitbase_imagenette_surrogate_shapley_eval_train/extract_output/train \
--batch_size 512 \
--normalize_function softmax \
--num_players 196 \
--target_subset_size 2440

python calculate_feature_attribution_using_extracted.py \
--input_path logs/vitbase_imagenette_surrogate_shapley_eval_train/extract_output/train \
--batch_size 512 \
--normalize_function softmax \
--num_players 196 \
--target_subset_size 2257

python calculate_feature_attribution_using_extracted.py \
--input_path logs/vitbase_imagenette_surrogate_shapley_eval_validation/extract_output/validation \
--batch_size 512 \
--normalize_function softmax \
--num_players 196 \
--target_subset_size 2257

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_shapley_regexplainer_upfront_2257_numtrain_100 \
python train_regexplainer.py configs/vitbase_imagenette_shapley_regexplainer_upfront_2257_numtrain_100.json

CUDA_VISIBLE_DEVICES=0,4,5,6 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_shapley_regexplainer_upfront_2257_numtrain_250 \
python train_regexplainer.py configs/vitbase_imagenette_shapley_regexplainer_upfront_2257_numtrain_250.json

CUDA_VISIBLE_DEVICES=4,5,6,7 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_shapley_regexplainer_upfront_2257_numtrain_500 \
python train_regexplainer.py configs/vitbase_imagenette_shapley_regexplainer_upfront_2257_numtrain_500.json

CUDA_VISIBLE_DEVICES=3,5,6,7 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_shapley_regexplainer_upfront_2257_numtrain_1000 \
python train_regexplainer.py configs/vitbase_imagenette_shapley_regexplainer_upfront_2257_numtrain_1000.json

CUDA_VISIBLE_DEVICES=3,5,6,7 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_shapley_regexplainer_upfront_2257_numtrain_2000 \
python train_regexplainer.py configs/vitbase_imagenette_shapley_regexplainer_upfront_2257_numtrain_2000.json

CUDA_VISIBLE_DEVICES=0,2,3,4 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_shapley_regexplainer_upfront_2257_numtrain_5000 \
python train_regexplainer.py configs/vitbase_imagenette_shapley_regexplainer_upfront_2257_numtrain_5000.json


CUDA_VISIBLE_DEVICES=4,5,6 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_shapley_eval_train_antithetical \
python train_surrogate.py configs/vitbase_imagenette_surrogate_shapley_eval_train_antithetical.json