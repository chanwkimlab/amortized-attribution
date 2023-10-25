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