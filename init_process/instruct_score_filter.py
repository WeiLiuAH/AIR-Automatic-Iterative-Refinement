
import json
import argparse

def main(args):
    with open(args.input_path, "r") as f:
        data = [json.loads(line) for line in f]

    res = []
    for item in data:
        if type(item.get("instruction_score")) == int and item["instruction_score"] >= 4:
            res.append({"document": item["document"], "instruction": item["instruction"]})
    print(len(res))

    with open(args.output_path, 'w', encoding='utf-8') as f:
        for item in res:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()
    main(args)

