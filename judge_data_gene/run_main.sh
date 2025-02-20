
export CUDA_VISIBLE_DEVICES=0,1,2,3

model_name=llama3_ultrachat
model=/path/llama3_ultrachat
guidance_model=/path/llama3_70b_instruct
py_path=./judge_data_gene/main.py

input_path=./data/dolma_init_process.jsonl
few_shot_constraints_path=./data/few_shot_constraints.jsonl
few_shot_constraints_combine_path=./data/few_shot_constraints_combine.jsonl
output_path=./data/llm_as_a_judge_${model_name}.jsonl


python ${py_path} \
    -i ${input_path} \
    -o ${output_path} \
    -m ${guidance_model} \
    -few_shot_constraints ${few_shot_constraints_path} \
    -few_shot_constraints_combine_path ${few_shot_constraints_combine_path}


python ${py_path} \
    -i ${output_path} \
    -o ${output_path} \
    -m ${model} \
    -few_shot_constraints ${few_shot_constraints_path} \
    -few_shot_constraints_combine_path ${few_shot_constraints_combine_path}


for i in {1..5}; do
    python ${py_path} \
        -i ${output_path} \
        -o ${output_path} \
        -m ${guidance_model} \
        -few_shot_constraints ${few_shot_constraints_path} \
        -few_shot_constraints_combine_path ${few_shot_constraints_combine_path}

    python ${py_path} \
        -i ${output_path} \
        -o ${output_path} \
        -m ${model} \
        -few_shot_constraints ${few_shot_constraints_path} \
        -few_shot_constraints_combine_path ${few_shot_constraints_combine_path}
    
    python ${py_path} \
        -i ${output_path} \
        -o ${output_path} \
        -m ${guidance_model} \
        -few_shot_constraints ${few_shot_constraints_path} \
        -few_shot_constraints_combine_path ${few_shot_constraints_combine_path}
done



python ${py_path} \
    -i ${output_path} \
    -o ${output_path} \
    -m ${guidance_model} \
    -few_shot_constraints ${few_shot_constraints_path} \
    -few_shot_constraints_combine_path ${few_shot_constraints_combine_path}

