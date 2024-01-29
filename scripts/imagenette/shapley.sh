# Evaluate surrogate model 

## surrogate evaluate (train set)

CUDA_VISIBLE_DEVICES=4,5,6,7 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_shapley_eval_train \
python train_surrogate.py configs/vitbase_imagenette_surrogate_shapley_eval_train.json

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_shapley_eval_train_antithetical \
python train_surrogate.py configs/vitbase_imagenette_surrogate_shapley_eval_train_antithetical.json

## surrogate evaluate (train set - regression)

python calculate_feature_attribution_using_extracted.py \
--input_path logs/vitbase_imagenette_surrogate_shapley_eval_train/extract_output/train \
--batch_size 512 \
--normalize_function softmax \
--num_players 196

### new sample
python calculate_feature_attribution_using_extracted.py \
--input_path logs/vitbase_imagenette_surrogate_shapley_eval_train/extract_output/train \
--batch_size 512 \
--normalize_function softmax \
--num_players 196 \
--target_subset_size 512

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
--target_subset_size 2330

python calculate_feature_attribution_using_extracted.py \
--input_path logs/vitbase_imagenette_surrogate_shapley_eval_train/extract_output/train \
--batch_size 512 \
--normalize_function softmax \
--num_players 196 \
--target_subset_size 2257



## surrogate evaluate (train set - antithetical)

python calculate_feature_attribution_using_extracted.py \
--input_path logs/vitbase_imagenette_surrogate_shapley_eval_train_antithetical/extract_output/train \
--batch_size 512 \
--normalize_function softmax \
--num_players 196

### for evaluation

CUDA_VISIBLE_DEVICES=1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_shapley_eval_train_regression \
python calculate_feature_attribution.py configs/vitbase_imagenette_surrogate_shapley_eval_train_regression.json

# CUDA_VISIBLE_DEVICES=5,6,7 \
# WANDB_PROJECT=xai-amortization \
# WANDB_NAME=vitbase_imagenette_surrogate_shapley_eval_train_regression_antithetical \
# python calculate_feature_attribution.py configs/vitbase_imagenette_surrogate_shapley_eval_train_regression_antithetical.json

## surrogate evaluate (train set - permutation)

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_shapley_eval_train_permutation \
python calculate_feature_attribution.py configs/vitbase_imagenette_surrogate_shapley_eval_train_permutation.json

CUDA_VISIBLE_DEVICES=3,5,6,7 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_shapley_eval_train_permutation_antithetical \
python calculate_feature_attribution.py configs/vitbase_imagenette_surrogate_shapley_eval_train_permutation_antithetical.json

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_shapley_eval_train_permutation_newsample_196 \
python calculate_feature_attribution.py configs/vitbase_imagenette_surrogate_shapley_eval_train_permutation_newsample_196.json

## surrogate evaluate (train set - SGD-Shapley)

CUDA_VISIBLE_DEVICES=4,5,6 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_shapley_eval_train_SGD_antithetical \
python calculate_feature_attribution.py configs/vitbase_imagenette_surrogate_shapley_eval_train_SGD_antithetical.json

## surrogate evaluate (validation set)

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_eval_validation \
python train_surrogate.py configs/vitbase_imagenette_surrogate_eval_validation.json

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_shapley_eval_validation_antithetical \
python train_surrogate.py configs/vitbase_imagenette_surrogate_shapley_eval_validation_antithetical.json


## surrogate evaluate (validation set - regression)

python calculate_feature_attribution_using_extracted.py \
--input_path logs/vitbase_imagenette_surrogate_eval_validation/extract_output/validation \
--batch_size 512 \
--normalize_function softmax \
--num_players 196

python calculate_feature_attribution_using_extracted.py \
--input_path logs/vitbase_imagenette_surrogate_eval_validation/extract_output/validation \
--batch_size 512 \
--normalize_function softmax \
--num_players 196 \
--target_subset_size 512

python calculate_feature_attribution_using_extracted.py \
--input_path logs/vitbase_imagenette_surrogate_shapley_eval_validation/extract_output/validation \
--batch_size 512 \
--normalize_function softmax \
--num_players 196 \
--target_subset_size 2330

python calculate_feature_attribution_using_extracted.py \
--input_path logs/vitbase_imagenette_surrogate_shapley_eval_validation/extract_output/validation \
--batch_size 512 \
--normalize_function softmax \
--num_players 196 \
--target_subset_size 2257

## surrogate evaluate (validation set - antithetical)

python calculate_feature_attribution_using_extracted.py \
--input_path logs/vitbase_imagenette_surrogate_shapley_eval_validation_antithetical/extract_output/validation \
--batch_size 512 \
--normalize_function softmax \
--num_players 196

## surrogate evaluate (validation set - permutation)

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_eval_validation \
python calculate_feature_attribution.py configs/vitbase_imagenette_surrogate_eval_validation_permutation.json

CUDA_VISIBLE_DEVICES=0,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_shapley_eval_validation_permutation_antithetical \
python calculate_feature_attribution.py configs/vitbase_imagenette_surrogate_shapley_eval_validation_permutation_antithetical.json


CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_shapley_eval_validation_permutation_newsample_196 \
python calculate_feature_attribution.py configs/vitbase_imagenette_surrogate_shapley_eval_validation_permutation_newsample_196.json

## surrogate evaluate (validation set - SGD-Shapley)

CUDA_VISIBLE_DEVICES=4,5,6 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_shapley_eval_validation_SGD_antithetical \
python calculate_feature_attribution.py configs/vitbase_imagenette_surrogate_shapley_eval_validation_SGD_antithetical.json


## surrogate evaluate (test set)

CUDA_VISIBLE_DEVICES=4,5,6,7 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_eval_test \
python train_surrogate.py configs/vitbase_imagenette_surrogate_eval_test.json

## surrogate evaluate (test regression)

python calculate_feature_attribution_using_extracted.py \
--input_path logs/vitbase_imagenette_surrogate_eval_test/extract_output/test \
--batch_size 512 \
--normalize_function softmax \
--num_players 196

python calculate_feature_attribution_using_extracted.py \
--input_path logs/vitbase_imagenette_surrogate_eval_test/extract_output/test \
--batch_size 512 \
--normalize_function softmax \
--num_players 196 \
--target_subset_size 512

## surrogate evaluate (test regression - antithetical)

CUDA_VISIBLE_DEVICES=0,1,2 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_shapley_eval_test_regression \
python calculate_feature_attribution.py configs/vitbase_imagenette_surrogate_shapley_eval_test_regression.json

CUDA_VISIBLE_DEVICES=1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_shapley_eval_test_regression_antithetical \
python calculate_feature_attribution.py configs/vitbase_imagenette_surrogate_shapley_eval_test_regression_antithetical.json

# Train AO models

## train explainer (objective-AO)
CUDA_VISIBLE_DEVICES=4,5,6,7 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_objexplainer_newsample_32 \
python train_objexplainer.py configs/vitbase_imagenette_objexplainer_newsample_32.json

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_objexplainer_upfront_3200 \
python train_objexplainer.py configs/vitbase_imagenette_objexplainer_upfront_3200.json

scp -r chanwkim@l3:/sdata/chanwkim/xai-amortization/logs_0901/vitbase_imagenette_surrogate_shapley_eval_train/* /sdata/chanwkim/xai-amortization/logs_0901/vitbase_imagenette_surrogate_shapley_eval_train/;
scp -r chanwkim@l3:/homes/gws/chanwkim/xai-amortization/logs/vitbase_imagenette_surrogate_shapley_eval_train_antithetical/* /sdata/chanwkim/xai-amortization/logs_0901/vitbase_imagenette_surrogate_shapley_eval_train_antithetical/;
scp -r chanwkim@l3:/homes/gws/chanwkim/xai-amortization/logs/vitbase_imagenette_surrogate_shapley_eval_validation_antithetical/* /sdata/chanwkim/xai-amortization/logs_0901/vitbase_imagenette_surrogate_shapley_eval_validation_antithetical/;

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_shapley_objexplainer_antithetical_newsample_32 \
python train_objexplainer.py configs/vitbase_imagenette_shapley_objexplainer_antithetical_newsample_32.json

## train explainer (regression-AO) upfront with various subset sizes
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

## train explainer (regression-AO) regression newsample with 512 subset size

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_shapley_regexplainer_newsample_512 \
python train_regexplainer.py configs/vitbase_imagenette_shapley_regexplainer_newsample_512.json

## train explainer (regression-AO) regression upfront-512 with various batch size

CUDA_VISIBLE_DEVICES=4 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_regexplainer_upfront_512_batch_2 \
python train_regexplainer.py configs/vitbase_imagenette_regexplainer_upfront_512_batch_2.json

CUDA_VISIBLE_DEVICES=5 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_regexplainer_upfront_512_batch_4 \
python train_regexplainer.py configs/vitbase_imagenette_regexplainer_upfront_512_batch_4.json

CUDA_VISIBLE_DEVICES=6 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_regexplainer_upfront_512_batch_16 \
python train_regexplainer.py configs/vitbase_imagenette_regexplainer_upfront_512_batch_16.json

CUDA_VISIBLE_DEVICES=7 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_regexplainer_upfront_512_batch_32 \
python train_regexplainer.py configs/vitbase_imagenette_regexplainer_upfront_512_batch_32.json


## train explainer (regression-AO) regression upfront-512-antithetical

CUDA_VISIBLE_DEVICES=0,4,5,6 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_shapley_regexplainer_antithetical_upfront_512 \
python train_regexplainer.py configs/vitbase_imagenette_shapley_regexplainer_antithetical_upfront_512.json


## train explainer (regression-AO) permutation

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

CUDA_VISIBLE_DEVICES=4,5,6,7 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_shapley_regexplainer_permutation_newsample_196 \
python train_regexplainer.py configs/vitbase_imagenette_shapley_regexplainer_permutation_newsample_196.json


CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_shapley_regexplainer_permutation_antithetical_upfront_196 \
python train_regexplainer.py configs/vitbase_imagenette_shapley_regexplainer_permutation_antithetical_upfront_196.json



## train explainer (regression-AO) Shapley_SGD (to-do)

CUDA_VISIBLE_DEVICES=1,2,3,5 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_shapley_regexplainer_SGD_antithetical_upfront_9986 \
python train_regexplainer.py configs/vitbase_imagenette_shapley_regexplainer_SGD_antithetical_upfront_9986.json

# flops
CUDA_VISIBLE_DEVICES=7 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_shapley_eval_train_regression_antithetical_flops \
dlprof --mode pytorch python calculate_feature_attribution.py configs/vitbase_imagenette_surrogate_shapley_eval_train_regression_antithetical_flops.json


dlprof --mode pytorch --force=true python python untitled.py
nsys profile -f true -o net --export sqlite python python untitled.py

# fp16
# gradient_accumulation_steps=2

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_shapley_regexplainer_upfront_512_flops \
python train_regexplainer.py configs/vitbase_imagenette_shapley_regexplainer_upfront_512_flops.json

