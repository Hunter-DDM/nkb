CUDA_VISIBLE_DEVICES=6 python3 summarization.py \
    --model_name_or_path ../../data/t5-base \
    --do_train \
    --do_eval \
    --train_file ../../data/xsum/train.json \
    --validation_file ../../data/xsum/dev.json \
    --kb_layer 11 \
    --ex_size 3072 \
    --optim_group all \
    --sec_lr zero \
    --text_column src \
    --summary_column tgt \
    --source_prefix "summarize: " \
    --output_dir output_dir \
    --overwrite_output_dir \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --max_train_samples 1000 \
    --max_eval_samples 10 \
    --save_strategy steps \
    --save_steps 1000 \
    --evaluation_strategy steps \
    --eval_steps 10 \
    --logging_strategy steps \
    --logging_steps 1 \
    --predict_with_generate