import json
import argparse
from tqdm import tqdm
import re
import random
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from rouge import Rouge
import torch


def vllm_generate(llm, tokenizer, messages_all, params, output_text=None):
    prompts = []
    for messages in messages_all:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        if output_text:
            text += output_text
        prompts.append(text)
    params["stop_token_ids"] = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    sampling_params = SamplingParams(**params)
    all_results = llm.generate(prompts, sampling_params)
    return all_results

def remove_tags(text):
    if not text:
        return text
    cleaned_text = text.strip()
    return cleaned_text


def result_search_score_constraint(text, finish_reason, columns_consts, keep_score1):
    if finish_reason == 'length':
        return "", "", ""
    if keep_score1:
        score_want_keep = ["3"] # ["2", "3"]
    else:
        score_want_keep = ["0", "1", "2"] # ["0", "1"]
    scores = []
    consts_right = []
    try:
        judges = text.split("\n\n")
        criteria = columns_consts.split("\n\n\n\n")
        assert len(criteria) == len(judges)
        for idx, judge in enumerate(judges):
            crit, score = judge.split("\t")
            score = score.strip()
            assert float(score) in [0, 1, 2, 3]    
            scores.append(score)
            if score in score_want_keep:
                consts_right.append(criteria[idx])
    except:
        print("------result_search_score_constraint error------")
        return "", "", ""
    return "\n\n".join(scores), "\n\n\n\n".join(consts_right), "\n\n".join(consts_right)



def result_search_simply(text, finish_reason):
    if finish_reason == 'length':
        return ""
    if len(text)>=3:
        text = text.split("\n\nNote:")[0]
        return remove_tags(text)
    else:
        return ""


def llm_infer_params_generate(max_output_tokens=4096, temperature=0.2):
    params = {
        'n': 1,
        'temperature':temperature, 
        'top_p':0.9, 
        'top_k':-1,
        'ignore_eos':False,
        'max_tokens':max_output_tokens,
        'skip_special_tokens':True,
        'logprobs': 1
    }
    return params

def output_improve_data_generate_1221_1(data, columns_ins="instruction", columns_output="output", columns_cons=None): # 对output进行润色
    messages_all = []

    for item in data:

        if columns_cons:
            cons = item[columns_cons].split("\n\n\n\n")
            cons = "\n\n".join(cons)
            ins = item[columns_ins] + "\n\n" + cons
        else:
            ins = item[columns_ins]

        prompt_template = """You are a professional editor. Given an instruction and an original response, your task is to improve the response while ensuring it aligns well with the instruction.\n\nThe improvement should focus on:\n- Better alignment with the instruction\n- Enhanced clarity and coherence\n- Aligns with AI assistant response style\n- Maintaining the core message while improving expression.\n\n"""

        prompt_template += f"""Now, this is your task. Please directly present your modifications, without using ANY headings or prefixes.\n\nInstruction: {ins}\nOriginal Response: {item[columns_output]}\nEnhanced Response:\n"""

        messages = [
            {"role": "system", "content": "You are a helpful assistant."}, 
            {"role": "user", "content": prompt_template}
        ]

        messages_all.append(messages)
    return messages_all


def constraint_combine_data_generate(data, few_shot_doc_ins_data, columns_ins, columns_cons, level=1):
    messages_all = []
    for item in data:

        prompt_template = f"""You are a skilled writing specialist who excels at blending different elements into cohesive, natural-sounding instructions.

Fusion guidelines:
- Consolidates overlapping constraints and resolves any conflicts
- Craft a cohesive instruction that naturally integrates ALL appropriate constraints
- AVOID expanding constraints

Here are some examples of merging instructions with constraints:"""

        # few-shot
        selected_examples = random.sample(few_shot_doc_ins_data, min(3, len(few_shot_doc_ins_data)))
        
        constraints_name = f"constraint_{level}"
        new_ins_name = f"new_ins_{level}"

        for idx, example in enumerate(selected_examples, 1):
            prompt_template += f"""

Example {idx}:
[Original Input]
{example['instruction']}

[Original Constraints]
{example[constraints_name]}

[Merged Instruction]
{example[new_ins_name]}"""

        prompt_template += f"""

Now it's your turn. Please merge the following input and constraints, do not output anything else, including response to the merged instruction:

[Original Input]
{item[columns_ins]}

[Original Constraints]
{item[columns_cons]}

"""

        messages = [
            {"role": "user", "content": prompt_template}
        ]

        messages_all.append(messages)
    
    return messages_all



def llm_data_generate(data, columns_ins="instruction", columns_cons=None):
    messages_all = []
    for examples in data:
        if columns_cons:
            cons = examples[columns_cons].split("\n\n\n\n")
            cons = "\n\n".join(cons)
            ins = examples[columns_ins] + "\n\n" + cons
        else:
            ins = examples[columns_ins]
            
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": ins}
        ]
        messages_all.append(messages)
    return messages_all



def doc_to_llm_answer_cons_data_generate_1221_model_select_1cons(data, constraints_by_type, constraint_field_pattern=None, constraint_field_range=None, columns_ins="instruction", columns_output1="output1", columns_output2="output2"):

    messages_all = []
    for item in data:

        ins = item[columns_ins]

        constraint_sections = []
        constraint_types_order = [
            'Data Format Constraint',
            'Document Structure Constraint',
            'Domain-Specific Format Constraint',
            'Exclusion Constraint',
            'Inclusion Constraint',
            'Citation Constraint',
            'Prior Condition Constraint',
            'Target Audience Constraint',
            'Tone and Style Constraint',
            'Emotion Constraint',
            'Linguistic Characteristics Constraint',
            'Multilingual Constraint'
        ]

        if constraint_field_range:
            for i in constraint_field_range:
                constraint_field = constraint_field_pattern.format(i=i)
                constraints_text = item.get(constraint_field, "")
                for constraints_text_key in constraints_text.keys():
                    if constraints_text_key in constraint_types_order:
                        constraint_types_order.remove(constraints_text_key)

        random.shuffle(constraint_types_order)
        random.shuffle(constraints_by_type)

        for idx, constraint_type in enumerate(constraint_types_order, start=1):
            if constraint_type == 'Data Format Constraint':
                description = 'The generated content may need to conform to specific data structure formats, such as JSON, Markdown, Table, CSV, etc.'
            elif constraint_type == 'Document Structure Constraint':
                description = 'The generated content may need to follow specific document organization patterns, including Numbered lists, Bullet points (•, -, *), Custom templates with predefined sections, Headers, or simply flow writing without any specific format, etc.'
            elif constraint_type == 'Domain-Specific Format Constraint':
                description = 'Content may need to follow specific format rules for different industries.'
            elif constraint_type == 'Inclusion Constraint':
                description = 'Specific elements or information that appear in Output1 but not in Output2.'
            elif constraint_type == 'Exclusion Constraint':
                description = 'Specific elements or information that appear in Output2 but not in Output1.'
            elif constraint_type == 'Citation Constraint':
                description = 'The generated content may need to include citations to sources, providing reliable sources and literature support; may need to follow specific citation formats or reference styles.'
            elif constraint_type == 'Prior Condition Constraint':
                description = 'When a specific intention is met, a particular process may need to be followed to perform an operation or output specific content.'
            elif constraint_type == 'Target Audience Constraint':
                description = 'The generated content may need to target a specific audience, which might affect the terminology used, the level of detail provided, and the complexity of the content.'
            elif constraint_type == 'Tone and Style Constraint':
                description = 'The generated content may need to adopt a specific tone and style, such as formal, polite, academic, concise, literary, romantic, or sci-fi.'
            elif constraint_type == 'Emotion Constraint':
                description = 'The generated content may need to express a specific emotion or mood, such as being positive, inspiring, or empathetic.'
            elif constraint_type == 'Linguistic Characteristics Constraint':
                description = 'May need to use specific linguistic features, such as metaphors, personification, and other rhetorical devices.'
            elif constraint_type == 'Multilingual Constraint':
                description = 'The generated content may need to be written in a specific language, such as English, Mandarin, or Spanish.'
            else:
                assert False, f"Unknown constraint type: {constraint_type}"
            

            constraints_temp = [temp['constraint'] for temp in constraints_by_type if temp['constraint_type'] == constraint_type]
            constraints_temp = constraints_temp[:1]
            if constraints_temp:
                formatted_constraints = "\n".join([f"- {constraint}" for constraint in constraints_temp])
                examples_text = f"Examples:\n{formatted_constraints}"
            else:
                examples_text = ""

            section = f"** {constraint_type}: {description}"
            if examples_text:
                section += f"\n{examples_text}"
            constraint_sections.append(section)

        constraints_text = "\n\n".join(constraint_sections)


        prompt_template = f"""Based on the provided instruction, I obtained Output1 and Output2 from two different models. Please analyze both outputs carefully to identify the MOST CRITICAL constraint type that Output2 needs to improve to match Output1's quality.

Available Constraint Types:
{constraints_text}

Task Requirements:
1. [Analysis] Compare Output1 and Output2 to identify differences
2. [Selection] Choose the SINGLE most critical constraint type where Output2 shows the biggest gap
3. [Constraint] Create ONE specific constraint that:
   - Addresses ONLY the selected constraint type
   - Exists in Output1 but is missing in Output2
   - Is written in a clear and concise sentence (10-20 words)
   - Avoids references to "Output1" or "Output2"
4. If no significant differences match the available types, specify "None"

Required Response Format:
**Analysis**: [Brief analysis]
**Selected Type**: [Single most critical type]
**Constraint**: [ONE specific constraint]

Context:
#Instruction#
{ins}

#Output1#
{item[columns_output1]}

#Output2#
{item[columns_output2]}

#Your Response#
"""

        messages = [
            {"role": "system", "content": "You are a helpful assistant."}, 
            {"role": "user", "content": prompt_template}]

        messages_all.append(messages)
    return messages_all




def model_select_1cons_dedup(data, constraint_field_pattern, constraint_field_range, columns_ins="instruction", columns_cons="cons"):

    messages_all = []
    for item in data:

        ins = item[columns_ins]

        pre_cons = []
        for i in constraint_field_range:
            constraint_field = constraint_field_pattern.format(i=i)
            constraints_text = item.get(constraint_field, "")
            for constraints_text_value in constraints_text.values():
                pre_cons.append(constraints_text_value)
        pre_cons = "\n\n".join(pre_cons)


        prompt_template = f"""Please analyze whether the new constraint is duplicated or very similar to any existing constraints. Compare them carefully in terms of their core meaning and requirements.

Task Requirements:
1. Compare the new constraint with all existing constraints
2. Consider constraints as duplicates if they:
   - Express the same core requirement
   - Have very similar meaning with different wording
   - Target the same aspect of the output
3. Provide detailed reasoning for your conclusion

Required Response Format:
**Reasoning**: [Explain your analysis]
**Answer**: [Yes if duplicated/very similar, No if significantly different]


Context:
#Instruction#
{ins}

#Existing Constraints#
{pre_cons}

#New Constraint#
{item[columns_cons]}

#Your Response#
"""

        messages = [
            {"role": "system", "content": "You are a helpful assistant."}, 
            {"role": "user", "content": prompt_template}
        ]

        messages_all.append(messages)
    
    return messages_all





def constraint_quality_evaluator(data, columns_ins, columns_consts, columns_output):

    messages_all = []
    for item in data:

        formatted_constraints = item[columns_consts]
        formatted_constraints = formatted_constraints.split("\n\n\n\n")
        formatted_constraints = "\n\n".join(formatted_constraints)

        ins_combie_constraints = f"{item[columns_ins]}\n\n{formatted_constraints}"
        prompt_template = f"""I want you to act as a quality evaluator. You need to evaluate the model answer by combining [User Instructions], [Model Answer], and [Evaluation Criteria] and score with 0-3.

Specifically, [Model Answer] is the response to [User Instructions], and [Evaluation Criteria] defines the points that the model answer should satisfy and needs to be evaluated. You need to strictly score the [Model Answer] according to each evaluation point in [Evaluation Criteria].

Scoring Rules:
- Score 0: Does not meet the evaluation criteria
- Score 1: Meets the evaluation criteria with acceptable response
- Score 2: Meets the evaluation criteria with high quality and comprehensive response
- Score 3: Meets the evaluation criteria with exceptional and flawless response

Output format: 1. Strictly output one line at a time according to the order of evaluation points in [Evaluation Criteria], with lines separated by "\n\n";
               2. Each line first outputs the corresponding content in [Evaluation Criteria], then uses "\t" to separate, and outputs the corresponding score(0-3) after it;
               3. Please output your evaluation directly without any other content;
               4. Note that if a criteria states like "do not include X", the score should be 0 if the answer includes X.

[Example]:
    [User Instructions]: Generate some poems by Li Bai from the Tang Dynasty about friendship. The output should be in JSON format with two keys: "poem" and "title". Provide three different poems. Your output should not contain the word "moon".
    
    [Model Answer]: '"poem": ["poem": "Moon shines on friend's heart", "title": "Friendship like the moon","poem": "Mountains and waters flow", "title": "Hard to find soulmate","poem": "Drink with close friends", "title": "Laughing drunk in battlefield"]'

    [Evaluation Criteria]: Generate poems by Li Bai from Tang Dynasty.\n\nThe poems should be about friendship.\n\nOutput should be in JSON format with two keys: "poem" and "title".\n\nProvide three different poems.\n\nYour output should not contain the word "moon".

    [Your Evaluation]: Generate poems by Li Bai from Tang Dynasty.\t0\n\nThe poems should be about friendship.\t1\n\nOutput should be in JSON format with two keys: "poem" and "title".\t0\n\nProvide three different poems.\t3\n\nYour output should not contain the word "moon".\t0

[User Instructions]: {ins_combie_constraints}

[Model Answer]: {item[columns_output]}

[Evaluation Criteria]: {formatted_constraints}

[Your Evaluation]: """

        messages = [
            {"role": "system", "content": "You are a helpful assistant."}, 
            {"role": "user", "content": prompt_template}
        ]
        messages_all.append(messages)

    return messages_all






def extract_constraints(text, finish_reason):
    if finish_reason == 'length':
        return {}
    
    text = text.split("\n\nNote:")[0]

    result = {}

    # constraint_type_pattern = r'\*\*(.*?Constraint Type)\*\*:\s*(.*?)(?=(?:\*\*|$))'
    # constraint_type_pattern = r'(?:\*\*)?\bConstraint Type\b(?:\*\*)?:\s*(\*{0,2})(.*?)\*{0,2}(?=\*\*|\n|$)'
    # constraint_content_pattern = r'\*\*(.*?Constraint Content)\*\*:\s*(.*?)(?=(?:\*\*|$))'

    constraint_type_pattern = r'(?:\*\*)?\bSelected Type\b(?:\*\*)?:\s*(\*{0,2})(.*?)\*{0,2}(?=\*\*|\n|$)'
    constraint_content_pattern = r'\*\*(.*?Constraint)\*\*:\s*(.*?)(?=(?:\*\*|$))'

    res1 = ""
    res2 = ""
    matches = re.findall(constraint_type_pattern, text, re.DOTALL)
    if matches:
        constraint_type = matches[0][1].strip()
        if constraint_type.startswith("\n"):
            constraint_type = constraint_type.split("\n")[1].strip()
        if constraint_type.endswith("\n"):
            constraint_type = constraint_type.split("\n")[-2].strip()
        res1 = constraint_type

    content_match = re.findall(constraint_content_pattern, text, re.DOTALL)
    if content_match:
        constraint_content = content_match[0][1].strip()
        res2 = constraint_content

    if (26 >= len(res2.split()) >= 3) and ("Output1" not in res2) and ("Output2" not in res2) and ("original instruction" not in res2) and ("ese constraints" not in res2) and 6>=len(res1.split())>=2 and "None" not in res2:
        result[res1] = res2

    return result



def extract_dedup(text):

    text = text.split("\n\nNote:")[0]
    constraint_content_pattern = r'\*\*(.*?Answer)\*\*:\s*(.*?)(?=(?:\*\*|$))'

    res2 = ""
    content_match = re.findall(constraint_content_pattern, text, re.DOTALL)
    if content_match:
        constraint_content = content_match[0][1].strip()
        res2 = constraint_content
        res2 = res2.strip()

        if res2.startswith("**"):
            res2 = res2[2:]
        if res2.lower()[:2] == "no":
            return False
        elif res2.lower()[:3] == "yes":
            return True

    temp = text.split("**Answer**:")[-1]
    temp = temp.strip()
    if len(temp.split("**")) >= 2:
        temp = temp.split("**")[1]
        temp = temp.replace(",", "")
    if "no" in [temp_i.lower() for temp_i in temp.split()]:
        return False

    return True



def main(args):

    # Read input data
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = f.readlines()
        data = [json.loads(data_i) for data_i in data]
    # data = data[:10]

    # Read few-shot constraints data, format: constraint type - specific constraint
    constraints_by_type = []
    with open(args.few_shot_constraints_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                constraint_entry = json.loads(line)
                temp = {
                    "constraint_type": constraint_entry.get('constraint_type'),
                    "constraint": constraint_entry.get('constraint')
                }
                constraints_by_type.append(temp)
    
    # Read few-shot constraints combination data
    with open(args.few_shot_constraints_combine_path, 'r', encoding='utf-8') as f:
        constraints_combine_data = [json.loads(line) for line in f]


    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(
        model=args.model_path,
        max_model_len=8192,
        dtype='bfloat16',
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.96,
    )

    # Polish documents
    if "document_polish" not in data[0].keys():
        print(f"using {args.model_path} to generate document_polish...")

        messages_all = output_improve_data_generate_1221_1(data, columns_ins=args.instruction, columns_output=args.output)
        params = llm_infer_params_generate()
        outputs = vllm_generate(llm, tokenizer, messages_all, params)
        for i, output in enumerate(outputs):
            data[i]["document_polish"] = result_search_simply(output.outputs[0].text, output.outputs[0].finish_reason)
        
        with open(args.output_file, 'w', encoding='utf-8') as f:    
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"using {args.model_path} to generate document_polish over")
        return


    iter_num = 1 # Current iteration number
    idx_list = list(range(len(data))) # idx_list contains indices of data to be processed
    
    if "output_student_loop1" not in data[0].keys():
        print(f"using {args.model_path} to generate output_student_loop1...")
     
        output_student_var = f"output_student_loop{iter_num}"
        messages_all = llm_data_generate(data, columns_ins=args.instruction)

        params = llm_infer_params_generate()
        outputs = vllm_generate(llm, tokenizer, messages_all, params)
        for idx, output in zip(idx_list, outputs):
            data[idx][output_student_var] = output.outputs[0].text
        
        with open(args.output_file, 'w', encoding='utf-8') as f:    
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Using {args.model_path} to generate {output_student_var} over")
        return

    # Five iteration loops
    while idx_list and iter_num <= 5:

        print(f"Starting iteration {iter_num}, length of data: {len(idx_list)}")

        filtered_data = [data[i] for i in idx_list] # Get current iteration data

        output_student_var = f"output_student_loop{iter_num}" # Student model's response to initial instruction
        output_student_next_var = f"output_student_loop{iter_num+1}" # Student model's response to final instruction

        instruction_var = f"instruction_loop{iter_num-1}" if iter_num > 1 else args.instruction # Initial instruction
        instruction_next_var = f"instruction_loop{iter_num}" # Final instruction for current iteration
        
        constraints_dict_var = f"constraints_dict_loop{iter_num}_all" # All constraints in this iteration in dict format

        constraints_document_check_var = f"constraints_loop{iter_num}_document_check" # Constraints that match document_polish
        constraints_student_check_var = f"constraints_loop{iter_num}_student_check" # Constraints that are difficult for 8b LLM
        constraints_student_check_temp_var = f"constraints_loop{iter_num}_student_check_temp"

        """
        Generate constraints: document_polish should meet constraints but output_student should not
        """
        # 1. Generate constraints
        if f"constraints_loop{iter_num}_0" not in filtered_data[0].keys():
            print(f"using {args.model_path} to generate constraint_loop{iter_num}_0")

            constraints_var = f"constraints_loop{iter_num}_0"
            messages_all = doc_to_llm_answer_cons_data_generate_1221_model_select_1cons(
                filtered_data, constraints_by_type,
                constraint_field_pattern="constraints_dict_loop{i}_all", 
                constraint_field_range = range(1, iter_num) if iter_num >= 2 else None,
                columns_ins=args.instruction,
                columns_output1="document_polish",
                columns_output2=output_student_var
            )

            params = llm_infer_params_generate()
            outputs = vllm_generate(llm, tokenizer, messages_all, params)
            for idx, output in zip(idx_list, outputs):
                data[idx][constraints_dict_var] = extract_constraints(output.outputs[0].text, output.outputs[0].finish_reason)
                data[idx][constraints_var] = output.outputs[0].text
            


            # 2. Check for duplicate constraints
            if iter_num > 1:
                constraints_dedup_var = f"constraints_loop{iter_num}_0_dedup_text"
                messages_all = model_select_1cons_dedup(
                    filtered_data,
                    constraint_field_pattern="constraints_dict_loop{i}_all", 
                    constraint_field_range = range(1, iter_num) if iter_num >= 2 else None,
                    columns_ins=args.instruction,
                    columns_cons=constraints_dict_var
                )

                params = llm_infer_params_generate()
                outputs = vllm_generate(llm, tokenizer, messages_all, params)
                for idx, output in zip(idx_list, outputs):
                    data[idx][constraints_dedup_var] = output.outputs[0].text
            
            for idx in idx_list:

                if iter_num <= 1 or not extract_dedup(data[idx][constraints_dedup_var]):
                    data[idx][constraints_document_check_var] = "\n\n\n\n".join(list(data[idx][constraints_dict_var].values()))
                    data[idx][instruction_next_var] = data[idx][instruction_var] + "\n\n" +  "\n\n".join(list(data[idx][constraints_dict_var].values()))
                else:
                    data[idx][constraints_document_check_var] = ""
                    data[idx][instruction_next_var] = data[idx][instruction_var] + "\n\n"
            
            with open(args.output_file, 'w', encoding='utf-8') as f:    
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

            print(f"using {args.model_path} to generate constraint_loop{iter_num}_0 over")
            return


        # 4. Generate answer using 8b SFT LLM
        """
        Student model generates output based on instruction_next_var
        """
        if output_student_next_var not in filtered_data[0].keys():
            print(f"using {args.model_path} to generate {output_student_next_var}")
            
            messages_all = llm_data_generate(filtered_data, columns_ins=instruction_next_var)

            params = llm_infer_params_generate()
            outputs = vllm_generate(llm, tokenizer, messages_all, params)
            for idx, output in zip(idx_list, outputs):
                data[idx][output_student_next_var] = output.outputs[0].text

            with open(args.output_file, 'w', encoding='utf-8') as f:    
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"using {args.model_path} to generate {output_student_next_var} over")
            return
        


        """
        Check which constraints were not met by 8b SFT LLM generated answer
        """
        if constraints_student_check_var not in filtered_data[0].keys():
            print(f"using {args.model_path} to generate {constraints_student_check_var}")
            messages_all = constraint_quality_evaluator(filtered_data, columns_ins=args.instruction, columns_consts=constraints_document_check_var, columns_output=output_student_next_var)
            params = llm_infer_params_generate()
            outputs = vllm_generate(llm, tokenizer, messages_all, params)
            for idx, output in zip(idx_list, outputs):
                item = data[idx]
                output1, output2, output3 = result_search_score_constraint(output.outputs[0].text, output.outputs[0].finish_reason, columns_consts = item[constraints_document_check_var], keep_score1=False)
                item[constraints_student_check_var] = output2
                item[constraints_student_check_temp_var] = output.outputs[0].text

            with open(args.output_file, 'w', encoding='utf-8') as f:    
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"using {args.model_path} to generate {constraints_student_check_var} over")
            return



        # List indices that can enter next iteration
        new_idx_list = []
        for idx in idx_list:
            item = data[idx]
            if len(item.get(constraints_document_check_var, "").split()) >= 3:
                new_idx_list.append(idx)
        idx_list = new_idx_list
        iter_num += 1



    # Final dataset generation
    def process_constraints(data, constraint_field_pattern, constraint_field_range, few_shot, result_constraint_key, new_instruction, llm_output_key=None):
        
        # 1. Collect constraints
        for idx in range(len(data)):
            item = data[idx]
            all_constraints = []
            if constraint_field_range is None:
                constraints_text = item.get(constraint_field_pattern, "")
                if constraints_text:
                    constraints_split = constraints_text.strip().split("\n\n\n\n")
                    all_constraints.extend(constraints_split)
            else:
                for i in constraint_field_range:
                    constraint_field = constraint_field_pattern.format(i=i)
                    constraints_text = item.get(constraint_field, "")
                    if constraints_text:
                        constraints_split = constraints_text.strip().split("\n\n\n\n")
                        all_constraints.extend(constraints_split)            
            unique_constraints = list(set(all_constraints))
            random.shuffle(unique_constraints)
            level = min(max(len(unique_constraints), 1), 5)
            item[result_constraint_key] = "\n\n\n\n".join(unique_constraints)


        # 2. Merge constraints and instructions into new instruction
        messages_all = constraint_combine_data_generate(data, few_shot, args.instruction, columns_cons=result_constraint_key, level=level)
        params = llm_infer_params_generate(max_output_tokens=800)
        outputs = vllm_generate(llm, tokenizer, messages_all, params, output_text="[Merged Instruction]\n")

        for i, output in enumerate(outputs):
            data[i][new_instruction] = result_search_simply(output.outputs[0].text, output.outputs[0].finish_reason)


        # 3. Generate response based on final instruction
        messages_all = llm_data_generate(data, columns_ins=new_instruction)
        params = llm_infer_params_generate()
        outputs = vllm_generate(llm, tokenizer, messages_all, params)
        for idx, output in enumerate(outputs):
            data[idx][llm_output_key] = output.outputs[0].text

    """
    Generate final instructions and responses
    """
    if 'output_loop1_final_guidance' not in data[0].keys():

        # All constraints from iteration 1
        process_constraints(
            data=data,
            constraint_field_pattern='constraints_loop1_document_check',
            constraint_field_range=None,
            few_shot=constraints_combine_data,
            result_constraint_key='constraints_loop1_final',
            new_instruction='instruction_loop1_final',
            llm_output_key='output_loop1_final_guidance'
        )

        # All constraints from iteration 2
        process_constraints(
            data=data,
            constraint_field_pattern='constraints_loop{i}_document_check',
            constraint_field_range=range(1, 3),
            few_shot=constraints_combine_data,
            result_constraint_key='constraints_loop2_final',
            new_instruction='instructions_loop2_final',
            llm_output_key='output_loop2_final_guidance'
        )

        # All constraints from iteration 3
        process_constraints(
            data=data,
            constraint_field_pattern='constraints_loop{i}_document_check',
            constraint_field_range=range(1, 4),
            few_shot=constraints_combine_data,
            result_constraint_key='constraints_loop3_final',
            new_instruction='instruction_loop3_final',
            llm_output_key='output_loop3_final_guidance'
        )

        # All constraints from iteration 4
        process_constraints(
            data=data,
            constraint_field_pattern='constraints_loop{i}_document_check',
            constraint_field_range=range(1, 5),
            few_shot=constraints_combine_data,
            result_constraint_key='constraints_loop4_final',
            new_instruction='instruction_loop4_final',
            llm_output_key='output_loop4_final_guidance'
        )

        # All constraints from iteration 5
        process_constraints(
            data=data,
            constraint_field_pattern='constraints_loop{i}_document_check',
            constraint_field_range=range(1, 6),
            few_shot=constraints_combine_data,
            result_constraint_key='constraints_loop5_final',
            new_instruction='instruction_loop5_final',
            llm_output_key='output_loop5_final_guidance'
        )


        with open(args.output_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"using {args.model_path} to generate output_loop1_final_72b_llm over")
        print("Finished")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", type=str)
    parser.add_argument("-o", "--output_file", type=str)
    parser.add_argument("-m", "--model_path", type=str)
    parser.add_argument("-few_shot_constraints", "--few_shot_constraints_path")
    parser.add_argument("-few_shot_constraints_combine_path", "--few_shot_constraints_combine_path", type=str)
    parser.add_argument("--instruction", type=str, default="instruction")
    parser.add_argument("--output", type=str, default="document")

    args = parser.parse_args()
    main(args)

