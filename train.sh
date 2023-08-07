CUDA_VISIBLE_DEVICES=2 python train_classifier.py \
    --dataset_name beans \
    --output_dir ./output/beans \
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
--dataset_name beans \
--output_dir ./output/beans \
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