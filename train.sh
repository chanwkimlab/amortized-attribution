
CUDA_VISIBLE_DEVICES=2 python train_classifier.py \
    --model_name_or_path microsoft/resnet-50 \
    --ignore_mismatched_sizes True \
    --dataset_name beans \
    --output_dir ./logs/resnet50_beans \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 42
# vit-base on imagenette
# train classifier
CUDA_VISIBLE_DEVICES=2 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette  \
python train_classifier.py \
    --model_name_or_path google/vit-base-patch16-224 \
    --ignore_mismatched_sizes True \
    --dataset_name frgfm/imagenette \
    --dataset_config_name 160px \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --fp16 True \
    --learning_rate 2e-5 \
    --num_train_epochs 25 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 42 \
    --output_dir ./logs/vitbase_imagenette_classifier \
    --report_to wandb 

# train surrogate (GPU util: train=80%, val=50%)
CUDA_VISIBLE_DEVICES=2 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_surrogate \
python train_surrogate.py \
    --classifier_model_name_or_path ./logs/vitbase_imagenette \
    --classifier_ignore_mismatched_sizes True \
    --surrogate_model_name_or_path ./logs/vitbase_imagenette \
    --surrogate_ignore_mismatched_sizes True \
    --dataset_name frgfm/imagenette \
    --dataset_config_name 160px \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --fp16 True \
    --learning_rate 2e-5 \
    --num_train_epochs 25 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 42 \
    --output_dir ./logs/vitbase_imagenette_surrogate \
    --report_to wandb


# train explainer 
CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_explainer \
python train_explainer.py \
    --surrogate_model_name_or_path ./logs/vitbase_imagenette_surrogate \
    --surrogate_ignore_mismatched_sizes True \
    --explainer_model_name_or_path ./logs/vitbase_imagenette_surrogate \
    --explainer_ignore_mismatched_sizes True \
    --dataset_name frgfm/imagenette \
    --dataset_config_name 160px \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --fp16 False \
    --learning_rate 1e-4 \
    --num_train_epochs 100 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 42 \
    --output_dir ./logs/vitbase_imagenette_explainer \
    --report_to wandb

CUDA_VISIBLE_DEVICES=4,5,6,7 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=segment \
python train_segmentation.py \
    --model_name_or_path nvidia/mit-b0 \
    --dataset_name segments/sidewalk-semantic \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --max_steps 10000 \
    --learning_rate 0.00006 \
    --lr_scheduler_type polynomial \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --logging_strategy steps \
    --logging_steps 100 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --seed 42 \
    --output_dir ./logs/segformer_outputs

CUDA_VISIBLE_DEVICES=4,5,6,7 \
WANDB_PROJECT=xai-amortization \
WANDB_NAME=vitbase_imagenette_segexplainer \
python train_segexplainer.py \
    --surrogate_model_name_or_path ./logs/vitbase_imagenette_surrogate \
    --surrogate_ignore_mismatched_sizes True \
    --explainer_model_name_or_path nvidia/mit-b0 \
    --explainer_ignore_mismatched_sizes True \
    --dataset_name frgfm/imagenette \
    --dataset_config_name 160px \
    --dataset_cache_dir /sdata/chanwkim/huggingface_cache \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --fp16 True \
    --learning_rate 1e-3 \
    --num_train_epochs 100 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 42 \
    --output_dir ./logs/vitbase_imagenette_segexplainer_ \
    --report_to none

# resnet50 on imagenette
CUDA_VISIBLE_DEVICES=2 python train_classifier.py \
    --model_name_or_path microsoft/resnet-50 \
    --ignore_mismatched_sizes True \
    --dataset_name frgfm/imagenette \
    --dataset_config_name 160px \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --learning_rate 2e-5 \
    --num_train_epochs 25 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 42 \
    --output_dir ./logs/resnet50_imagenette \
    --report_to wandb       


CUDA_VISIBLE_DEVICES=3 python train_surrogate.py \
    --classifier_model_name_or_path ./logs/resnet50_imagenette \
    --classifier_ignore_mismatched_sizes True \
    --surrogate_model_name_or_path ./logs/resnet50_imagenette \
    --surrogate_ignore_mismatched_sizes True \
    --dataset_name frgfm/imagenette \
    --dataset_config_name 160px \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --learning_rate 2e-5 \
    --num_train_epochs 25 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 42 \
    --output_dir ./logs/resnet50_imagenette_surrogate \
    --report_to wandb         


CUDA_VISIBLE_DEVICES=4 python train_surrogate.py \
    --classifier_model_name_or_path ./logs/resnet50_imagenette \
    --classifier_ignore_mismatched_sizes True \
    --surrogate_model_name_or_path ./logs/resnet50_imagenette \
    --surrogate_ignore_mismatched_sizes True \
    --dataset_name frgfm/imagenette \
    --dataset_config_name 160px \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --learning_rate 1e-4 \
    --num_train_epochs 25 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 42 \
    --output_dir ./logs/resnet50_imagenette_surrogate_ \
    --report_to wandb       