CUDA_VISIBLE_DEVICES=7 python3 kv_analysis.py \
    --model_name_or_path ../../data/t5_nkb_wqft_600 \
    --do_train \
    --do_eval \
    --train_file ../../data/cbqa_wq_std/train_short_ans.json \
    --validation_file ../../data/cbqa_wq_std/dev_short_ans.json \
    --kb_layer 11 \
    --ex_size 3072 \
    --overwrite_cache \
    --question_column question \
    --answer_column answers \
    --max_seq_length 512 \
    --pad_to_max_length False \
    --output_dir output_dir \
    --overwrite_output_dir \
    --save_strategy no \
    --seed 1234