CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 src/gpt2_beam.py      --data ./data/e2e/test.jsonl      --batch_size 1      --seq_len 512      --eval_len 64      --model_card gpt2.md      --init_checkpoint ./trained_models/GPT2_M_e2e_rank_4_100_s110/model.20000.pt      --platform local      --lora_dim 4      --lora_alpha 32      --beam 10      --length_penalty 0.8      --no_repeat_ngram_size 4      --repetition_penalty 1.0      --eos_token_id 628      --work_dir ./trained_models/GPT2_M_e2e_rank_4_100_s110      --output_file predict.20000.b10p08.jsonl

CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 src/gpt2_beam.py      --data ./data/e2e/test.jsonl      --batch_size 1      --seq_len 512      --eval_len 64      --model_card gpt2.md      --init_checkpoint ./trained_models/GPT2_M_e2e_rank_4_100_s1/model.20000.pt      --platform local      --lora_dim 4      --lora_alpha 32      --beam 10      --length_penalty 0.8      --no_repeat_ngram_size 4      --repetition_penalty 1.0      --eos_token_id 628      --work_dir ./trained_models/GPT2_M_e2e_rank_4_100_s1      --output_file predict.20000.b10p08.jsonl

CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 src/gpt2_beam.py      --data ./data/e2e/test.jsonl      --batch_size 1      --seq_len 512      --eval_len 64      --model_card gpt2.md      --init_checkpoint ./trained_models/GPT2_M_e2e_rank_4_100_s2/model.20000.pt      --platform local      --lora_dim 4      --lora_alpha 32      --beam 10      --length_penalty 0.8      --no_repeat_ngram_size 4      --repetition_penalty 1.0      --eos_token_id 628      --work_dir ./trained_models/GPT2_M_e2e_rank_4_100_s2      --output_file predict.20000.b10p08.jsonl



python src/gpt2_decode.py \
    --vocab ./vocab \
    --sample_file ./trained_models/GPT2_M_e2e_rank_4_100_s1/predict.20000.b10p08.jsonl \
    --input_file ./data/e2e/test_formatted.jsonl \
    --output_ref_file e2e_ref.txt \
    --output_pred_file e2e_pred.txt

python eval/e2e/measure_scores.py e2e_ref.txt e2e_pred.txt -p