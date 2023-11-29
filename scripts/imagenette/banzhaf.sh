CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_banzhaf_eval_train \
python train_surrogate.py configs/vitbase_imagenette_surrogate_banzhaf_eval_train.json

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_banzhaf_eval_validation \
python train_surrogate.py configs/vitbase_imagenette_surrogate_banzhaf_eval_validation.json


CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_banzhaf_eval_test_regression \
python calculate_feature_attribution.py configs/vitbase_imagenette_surrogate_banzhaf_eval_test_regression.json



CUDA_VISIBLE_DEVICES=4,5,6 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_banzhaf_eval_train_sampling_antithetical \
python calculate_feature_attribution.py configs/vitbase_imagenette_surrogate_banzhaf_eval_train_sampling_antithetical.json

CUDA_VISIBLE_DEVICES=0,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_banzhaf_eval_train_sampling_long \
python calculate_feature_attribution.py configs/vitbase_imagenette_surrogate_banzhaf_eval_train_sampling_long.json

CUDA_VISIBLE_DEVICES=4,5,6 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_banzhaf_eval_test_sampling_long \
python calculate_feature_attribution.py configs/vitbase_imagenette_surrogate_banzhaf_eval_test_sampling_long.json

CUDA_VISIBLE_DEVICES=4,5,6 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_banzhaf_eval_train_sampling \
python calculate_feature_attribution.py configs/vitbase_imagenette_surrogate_banzhaf_eval_train_sampling.json

CUDA_VISIBLE_DEVICES=4,5,6 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate_banzhaf_eval_validation_sampling \
python calculate_feature_attribution.py configs/vitbase_imagenette_surrogate_banzhaf_eval_validation_sampling.json


CUDA_VISIBLE_DEVICES=4,5,6,7 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_banzhaf_regexplainer_upfront_5000 \
python train_regexplainer.py configs/vitbase_imagenette_banzhaf_regexplainer_upfront_5000.json

CUDA_VISIBLE_DEVICES=4,5,6,7 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_banzhaf_regexplainer_upfront_100 \
python train_regexplainer.py configs/vitbase_imagenette_banzhaf_regexplainer_upfront_100.json

CUDA_VISIBLE_DEVICES=4,5,6,7 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_banzhaf_regexplainer_upfront_500 \
python train_regexplainer.py configs/vitbase_imagenette_banzhaf_regexplainer_upfront_500.json

CUDA_VISIBLE_DEVICES=4,5,6,7 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_banzhaf_regexplainer_upfront_500_normalize \
python train_regexplainer.py configs/vitbase_imagenette_banzhaf_regexplainer_upfront_500_normalize.json

CUDA_VISIBLE_DEVICES=4,5,6,7 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_banzhaf_regexplainer_upfront_500_normalize_max \
python train_regexplainer.py configs/vitbase_imagenette_banzhaf_regexplainer_upfront_500_normalize_max.json

CUDA_VISIBLE_DEVICES=4,5,6,7 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_banzhaf_regexplainer_upfront_normalize_constantfactor_500 \
python train_regexplainer_normalize.py configs/vitbase_imagenette_banzhaf_regexplainer_upfront_normalize_constantfactor_500.json

CUDA_VISIBLE_DEVICES=4,5,6,7 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_banzhaf_regexplainer_upfront_normalize_constantfactor_1000_500 \
python train_regexplainer_normalize.py configs/vitbase_imagenette_banzhaf_regexplainer_upfront_normalize_constantfactor_1000_500.json

CUDA_VISIBLE_DEVICES=4,5,6,7 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_banzhaf_regexplainer_upfront_normalize_constantfactor_1e-5_500 \
python train_regexplainer_normalize.py configs/vitbase_imagenette_banzhaf_regexplainer_upfront_normalize_constantfactor_1e-5_500.json

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_banzhaf_regexplainer_upfront_normalize_squaredtransformation_500 \
python train_regexplainer_normalize.py configs/vitbase_imagenette_banzhaf_regexplainer_upfront_normalize_squaredtransformation_500.json


tar zcvf logs/vitbase_imagenette_surrogate_banzhaf_eval_train_sampling.tar.gz logs/vitbase_imagenette_surrogate_banzhaf_eval_train_sampling/
tar zcvf logs/vitbase_imagenette_surrogate_banzhaf_eval_validation_sampling.tar.gz logs/vitbase_imagenette_surrogate_banzhaf_eval_validation_sampling/
tar zcvf logs/vitbase_imagenette_surrogate_banzhaf_eval_train_sampling_long.tar.gz logs/vitbase_imagenette_surrogate_banzhaf_eval_train_sampling_long/





# 89/9456 1807
265 in progress