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


def result_search_init_ins(text, finish_reason):
    if finish_reason == 'length':
        return ""

    text = text.split("\n")
    text_2 = [i for i in text if "This instruction" not in i and "Here is a" not in i and len(i) >= 5]
    if text_2:
        text_2 = text_2[-1].strip()
        if text_2[0] == '"' and text_2[-1] == '"':
            return text_2[1:-1]
        else:
            return text_2
    else:
        text = text[-1].strip()
        if text[0] == '"' and text[-1] == '"':
            return text[1:-1]
        else:
            return text
        

def result_search_instruction_score(text, finish_reason):
    if finish_reason == 'length':
        return ""

    match = re.search(r'(?i)Score: (\d+)/5', text)
    if match and int(match.group(1)) >= 0 and int(match.group(1)) <= 5:
        return int(match.group(1))
    else:
        return ""
        

def back_translation(data, output):
    messages_all = []
    for item in data:
        prompt_template = f"""Please generate a single instruction that would lead to the given text as a response. 
- The instruction should not be a question. Instead, it should be a more general task.
- The instruction should not cover all details of the response. Instead, it should be concise and only focus on the main aspect.

Text: {item[output]}
Instruction: """

        messages =  [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_template}
        ]

        messages_all.append(messages)
    
    return messages_all



def instruction_score_data_generate_1221(data, columns):

    messages_all = []
    for item in data:

        prompt = f"""Review the user's instruction using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

Award 1 point for containing a basic question or task.
Add 1 point if the instruction can be addressed using the language model's existing knowledge base without requiring external resources or current event information.
Add 1 point if the instruction does NOT require analyzing specific texts, documents, or specific person's perspective.
Add 1 point if the instruction effectively communicates both the core question and key preferences, demonstrating clear intent while being self-contained.
Add 1 point if the instruction pertains to general topics or advice that are widely applicable and within the common knowledge base, rather than requiring specialized or niche information about specific individuals or events.

After examining the instruction:
- Briefly justify your total score, up to 100 words.
- Conclude with the score using the format: "Score: <total points>/5"

Example 1:
Instruction: What was the impact of Gary Gilmour's career and his life in the years following his cricketing career?
Answer: The instruction poses a basic question about Gary Gilmour's impact after his cricketing career (1 point). It can be answered using the language model's existing knowledge (1 point). It doesn't require analyzing specific texts, documents, or a specific person's perspective (1 point). The question is clear, self-contained, and demonstrates clear intent (1 point). However, since it involves information about a specific individual, which requires specialized or niche knowledge, the last point is not awarded.\nScore: 4/5

Example 2:
Instruction: What's the most helpful advice you have for students who are awaiting their college admission decision?
Answer: The instruction asks for the most helpful advice for students awaiting their college admission decisions, which is a basic question (1 point). It can be answered using the language model's existing knowledge (1 point). It does not require analyzing specific texts, documents, or a specific person's perspective (1 point). The question is clear, self-contained, and demonstrates clear intent (1 point). It pertains to a general topic that is widely applicable and within the common knowledge base (1 point).\nScore: 5/5

Example 3:
Instruction: What are the staff's recommendations and discoveries for shows, books, and music on What's the Buzz?
Answer: The instruction poses a basic question about the staff's recommendations and discoveries for shows, books, and music on "What's the Buzz" (1 point). It can be addressed using the language model's existing knowledge (1 point).  However, this is about the staff's specific perspectives, thus failing the third criterion.
Score: 2/5

Example 4:
Instruction: What's the response from Penn State alumni and the broader community to the recent tragic events?
Answer: The instruction poses a basic question about the response from Penn State alumni and the broader community to recent tragic events (1 point). However, it refers to "recent tragic events" without specifying them, and they may have occurred after the model's knowledge cutoff, requiring current event information, so it does not satisfy the second criterion.
Score: 1/5

This is your task:
Instruction: {item[columns]}
Answer: """

        messages = [
                {"role": "system", "content": "You are a helpful assistant."}, 
                {"role": "user", "content": prompt}
            ]
        messages_all.append(messages)
    return messages_all



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


def main(args):

    # 读取输入数据
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = f.readlines()
        data = [json.loads(data_i) for data_i in data]
    # data = data[:10]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(
        model=args.model_path,
        max_model_len=8192,
        dtype='bfloat16',
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.96,
    )

    # back translation
    messages_all = back_translation(data, args.output)
    params = llm_infer_params_generate()
    outputs = vllm_generate(llm, tokenizer, messages_all, params)
    for i, output in enumerate(outputs):
        data[i][args.instruction] = result_search_init_ins(output.outputs[0].text, output.outputs[0].finish_reason)

    # instruction score
    messages_all = instruction_score_data_generate_1221(data, args.instruction)
    params = llm_infer_params_generate()
    outputs = vllm_generate(llm, tokenizer, messages_all, params)
    for i, output in enumerate(outputs):
        data[i][args.instruction+"_score"] = result_search_instruction_score(output.outputs[0].text, output.outputs[0].finish_reason)
    

    with open(args.output_file, 'w', encoding='utf-8') as f:    
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", type=str)
    parser.add_argument("-o", "--output_file", type=str)
    parser.add_argument("-m", "--model_path", type=str)
    parser.add_argument("--instruction", type=str, default="instruction")
    parser.add_argument("--output", type=str, default="document")

    args = parser.parse_args()
    main(args)

