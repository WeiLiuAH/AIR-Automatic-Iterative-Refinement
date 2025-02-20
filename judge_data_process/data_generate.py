
import json
import numpy as np
import random
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--loop1_guidance_path', type=str, required=True)
    parser.add_argument('--loop2_guidance_path', type=str, required=True)
    parser.add_argument('--loop3_guidance_path', type=str, required=True)
    parser.add_argument('--loop4_guidance_path', type=str, required=True)
    parser.add_argument('--loop5_guidance_path', type=str, required=True)
    return parser.parse_args()



def main(args):

    with open(args.input_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
        data = [json.loads(data_i) for data_i in data]

    loop1_guidance = []
    loop2_guidance = []
    loop3_guidance = []
    loop4_guidance = []
    loop5_guidance = []

    instruction_len_min = 3
    instruction_len_max = 150
    init_instruction_len_max = 50



    for item in data:

        if not instruction_len_min <= len(item.get('instruction', '').split()) <= init_instruction_len_max:
            continue

        if type(item.get("instruction_score", '')) == int and item['instruction_score'] <= 3:
            continue

        if len(item.get("document_polish", '').split()) < 3:
            continue
        

        # Check if this data item actually generated constraints in round 1
        if len(item.get("output_student_loop1", '').split()) < 3 or len(item.get("constraints_dict_loop1_all", '')) == 0 or len(item.get("constraints_loop1_document_check", '').split()) < 3:
            continue
        # For iteration 1: Check if instruction and output meet length requirements
        if len(item.get("constraints_loop1_final", '').split()) > 3 and instruction_len_max>len(item.get("instruction_loop1_final", '').split())>instruction_len_min and len(item.get("output_loop1_final_guidance", '').split()) > 3:
            
            item_new = {}
            item_new["instruction"] = item["instruction_loop1_final"]
            item_new["input"] = ""
            item_new["output"] = item["output_loop1_final_guidance"]
            loop5_guidance.append(item_new)
    


        # Check if this data item actually generated constraints in round 2
        if len(item.get("output_student_loop2", '').split()) < 3 or len(item.get("constraints_dict_loop2_all", '')) == 0 or len(item.get("constraints_loop2_document_check", '').split()) < 3:
            continue
        # For iteration 2: Check if instruction and output meet length requirements
        if len(item.get("constraints_loop2_final", '').split()) > 3 and instruction_len_max>len(item.get("instructions_loop2_final", '').split())>instruction_len_min and len(item.get("output_loop2_final_guidance", '').split()) > 3:
            
            item_new = {}
            item_new["instruction"] = item["instruction_loop2_final"]
            item_new["input"] = ""
            item_new["output"] = item["output_loop2_final_guidance"]
            loop2_guidance.append(item_new)



        # Check if this data item actually generated constraints in round 3
        if len(item.get("output_student_loop3", '').split()) < 3 or len(item.get("constraints_dict_loop3_all", '')) == 0 or len(item.get("constraints_loop3_document_check", '').split()) < 3:
            continue
        # For iteration 3: Check if instruction and output meet length requirements
        if len(item.get("constraints_loop3_final", '').split()) > 3 and instruction_len_max>len(item.get("instruction_loop3_final", '').split())>instruction_len_min and len(item.get("output_loop3_final_guidance", '').split()) > 3:
            
            item_new = {}
            item_new["instruction"] = item["instruction_loop3_final"]
            item_new["input"] = ""
            item_new["output"] = item["output_loop3_final_guidance"]
            loop3_guidance.append(item_new)




        # Check if this data item actually generated constraints in round 4
        if len(item.get("output_student_loop4", '').split()) < 3 or len(item.get("constraints_dict_loop4_all", '')) == 0 or len(item.get("constraints_loop4_document_check", '').split()) < 3:
            continue
        # For iteration 4: Check if instruction and output meet length requirements
        if len(item.get("constraints_loop4_final", '').split()) > 3 and instruction_len_max>len(item.get("instruction_loop4_final", '').split())>instruction_len_min and len(item.get("output_loop4_final_guidance", '').split()) > 3:
            
            item_new = {}
            item_new["instruction"] = item["instruction_loop4_final"]
            item_new["input"] = ""
            item_new["output"] = item["output_loop4_final_guidance"]
            loop4_guidance.append(item_new)




        # Check if this data item actually generated constraints in round 5
        if len(item.get("output_student_loop5", '').split()) < 3 or len(item.get("constraints_dict_loop5_all", '')) == 0 or len(item.get("constraints_loop5_document_check", '').split()) < 3:
            continue
        # For iteration 5: Check if instruction and output meet length requirements
        if len(item.get("constraints_loop5_final", '').split()) > 3 and instruction_len_max>len(item.get("instruction_loop5_final", '').split())>instruction_len_min and len(item.get("output_loop5_final_guidance", '').split()) > 3:

            item_new = {}
            item_new["instruction"] = item["instruction_loop5_final"]
            item_new["input"] = ""
            item_new["output"] = item["output_loop5_final_guidance"]
            loop5_guidance.append(item_new)

 

    print("final length")
    print("1. ：", len(loop1_guidance))
    print("2. ：", len(loop2_guidance))
    print("3. ：", len(loop3_guidance))
    print("4. ：", len(loop4_guidance))
    print("5. ：", len(loop5_guidance))

    loop1_guidance = loop1_guidance[:10000]
    loop2_guidance = loop2_guidance[:10000]
    loop3_guidance = loop3_guidance[:10000]
    loop4_guidance = loop4_guidance[:10000]
    loop5_guidance = loop5_guidance[:10000]

    with open(args.loop1_guidance_path, 'w', encoding='utf-8') as f:
        json.dump(loop1_guidance, f, ensure_ascii=False, indent=2)
    with open(args.loop2_guidance_path, 'w', encoding='utf-8') as f:
        json.dump(loop2_guidance, f, ensure_ascii=False, indent=2)
    with open(args.loop3_guidance_path, 'w', encoding='utf-8') as f:
        json.dump(loop3_guidance, f, ensure_ascii=False, indent=2)
    with open(args.loop4_guidance_path, 'w', encoding='utf-8') as f:
        json.dump(loop4_guidance, f, ensure_ascii=False, indent=2)
    with open(args.loop5_guidance_path, 'w', encoding='utf-8') as f:
        json.dump(loop5_guidance, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    args = parse_args()
    main(args)
