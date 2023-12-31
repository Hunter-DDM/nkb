CUDA_VISIBLE_DEVICES=5 python3 cbqa_baseline.py \
    --model_name_or_path ../../data/t5-base \
    --train_file ../../data/cbqa_hq_std/train_wocxt.json \
    --validation_file ../../data/cbqa_hq_std/dev_wocxt.json \
    --dropout 0.2 \
    --do_train \
    --do_eval \
    --predict_with_generate \
    --metric_for_best_model exact_match \
    --overwrite_cache \
    --question_column question \
    --answer_column answers \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --optim adafactor \
    --learning_rate 1e-3 \
    --num_train_epochs 20 \
    --lr_scheduler_type constant \
    --max_seq_length 512 \
    --pad_to_max_length False \
    --output_dir output_dir \
    --overwrite_output_dir \
    --save_strategy no \
    --evaluation_strategy steps \
    --eval_steps 50 \
    --logging_strategy steps \
    --logging_steps 10 \
    --seed 1234