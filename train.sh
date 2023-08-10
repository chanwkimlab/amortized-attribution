
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
CUDA_VISIBLE_DEVICES=2 \
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
    --fp16 True \
    --learning_rate 1e-4 \
    --num_train_epochs 25 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 42 \
    --output_dir ./logs/vitbase_imagenette_explainer \
    --report_to none        


CUDA_VISIBLE_DEVICES=4,5,6,7 \
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
    --fp16 True \
    --learning_rate 1e-4 \
    --num_train_epochs 25 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 42 \
    --output_dir ./logs/vitbase_imagenette_explainer_ \
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