python src/gpt2_decode.py     --vocab ./vocab     --sample_file ./trained_models/GPT2_M_bilateral_rank1_1/e2e/predict.105155.b10p08.jsonl     --input_file ./data/e2e/test_formatted.jsonl     --output_ref_file e2e_ref.txt     --output_pred_file e2e_pred.txt

python eval/e2e/measure_scores.py e2e_ref.txt e2e_pred.txt -p