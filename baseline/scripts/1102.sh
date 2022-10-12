
CUDA_VISIBLE_DEVICES=3 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8731 src/gpt2_beam_new_weights.py      --data ./data/e2e/test.jsonl      --batch_size 1      --seq_len 512      --eval_len 64      --model_card gpt2.md      --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin         --platform local      --lora_dim 4     --lora_alpha 32      --beam 10      --length_penalty 0.8      --no_repeat_ngram_size 4      --repetition_penalty 1.0      --eos_token_id 628      --work_dir ./trained_models/GPT2_M_e2e_lora/      --output_file predict.b10p08.jsonl > e2e.out &


python src/gpt2_decode.py     --vocab ./vocab     --sample_file ./trained_models/GPT2_M_e2e_lora/predict.b10p08.jsonl     --input_file ./data/e2e/test_formatted.jsonl     --output_ref_file e2e_ref.txt     --output_pred_file e2e_pred.txt

python eval/e2e/measure_scores.py e2e_ref.txt e2e_pred.txt -p