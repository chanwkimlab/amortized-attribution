CUDA_VISIBLE_DEVICES=2 python train_classifier.py \
    --dataset_name beans \
    --output_dir ./logs/beans \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --push_to_hub \
    --push_to_hub_model_id vit-base-beans \
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


CUDA_VISIBLE_DEVICES=2 python train_surrogate.py \
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
    --num_train_epochs 1 \
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