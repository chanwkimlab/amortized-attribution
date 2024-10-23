CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_binomial_eval_train \
python train_surrogate.py configs/vitbase_imagenette_surrogate_binomial_eval_train.json

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_binomial_eval_validation \
python train_surrogate.py configs/vitbase_imagenette_surrogate_binomial_eval_validation.json


python calculate_feature_attribution_using_extracted.py \
--input_path logs/vitbase_imagenette_surrogate_binomial_eval_train/extract_output/train \
--batch_size 128 \
--normalize_function softmax \
--num_players 196 \
--attribution_name lime

python calculate_feature_attribution_using_extracted.py \
--input_path logs/vitbase_imagenette_surrogate_binomial_eval_validation/extract_output/validation \
--batch_size 128 \
--normalize_function softmax \
--num_players 196 \
--attribution_name lime


CUDA_VISIBLE_DEVICES=0,1,2 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_lime_eval_train_regression_long \
python calculate_feature_attribution.py configs/vitbase_imagenette_surrogate_lime_eval_train_regression_long.json

CUDA_VISIBLE_DEVICES=0,1,2 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_lime_eval_test_regression_long \
python calculate_feature_attribution.py configs/vitbase_imagenette_surrogate_lime_eval_test_regression_long.json


CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_lime_regexplainer_upfront_global_256 \
python train_regexplainer_normalize.py configs/vitbase_imagenette_lime_regexplainer_upfront_global_256.json

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_lime_regexplainer_upfront_global_512 \
python train_regexplainer_normalize.py configs/vitbase_imagenette_lime_regexplainer_upfront_global_512.json


CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_lime_regexplainer_upfront_perinstanceperclass_256 \
python train_regexplainer_normalize.py configs/vitbase_imagenette_lime_regexplainer_upfront_perinstanceperclass_256.json

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_lime_regexplainer_upfront_perinstanceperclass_512 \
python train_regexplainer_normalize.py configs/vitbase_imagenette_lime_regexplainer_upfront_perinstanceperclass_512.json
