model_name=llama3_ultrachat
input_path=./data/llm_as_a_judge_${model_name}.jsonl


python ./judge_data_process/data_generate.py \
    --input_path ${input_path} \
    --loop1_path "./data/${model_name}/loop1_guidance.json" \
    --loop2_path "./data/${model_name}/loop2_guidance.json" \
    --loop3_path "./data/${model_name}/loop3_guidance.json" \
    --loop4_path "./data/${model_name}/loop4_guidance.json" \
    --loop5_path "./data/${model_name}/loop5_guidance.json"
