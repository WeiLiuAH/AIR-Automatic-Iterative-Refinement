import pandas as pd
import json
import argparse
import re

def check_symbol_ratio(text):
    # Calculate the ratio of symbols to total text length
    symbols = len(re.findall(r'[^\w\s]', text))
    total_chars = len(text)
    if total_chars == 0:
        return False
    return symbols / total_chars <= 0.5  # Symbol ratio threshold set to 50%

def check_redundancy(text):
    # Check for duplicate paragraphs
    paragraphs = text.split('\n')
    if len(paragraphs) - len(set(paragraphs)) > len(paragraphs) * 0.1:
        return False
    
    # Check for excessive symbol repetition
    if re.search(r'(.)\1{10,}', text):  # Same character repeated more than 10 times
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='Input directory path')
    parser.add_argument('--output_path', type=str, required=True, help='Output file path')
    parser.add_argument('--only_filter_length', type=bool, default=True, help='Whether to filter only by length')
    args = parser.parse_args()

    new_data = []
    for i in range(3):
        try:
            data_temp = pd.read_parquet(f"{args.input_path}/00{i}_00000.parquet")
            
            # Filter by text length
            data_temp['length'] = data_temp['text'].apply(lambda x: len(x.split()))
            data_temp = data_temp[data_temp['length'] >= 50]
            data_temp = data_temp[data_temp['length'] <= 2048]
            
            # Check symbol ratio and redundancy
            if not args.only_filter_length:
                data_temp = data_temp[data_temp['text'].apply(check_symbol_ratio)]
                data_temp = data_temp[data_temp['text'].apply(check_redundancy)]
            
            print(f"File {i}: {len(data_temp)} documents after filtering")
            
            # Sampling
            sample_size = min(int(10000*2*5), len(data_temp))
            data_temp = data_temp.sample(n=sample_size)

            for _, item in data_temp.iterrows():
                new_item = {
                    'document': item['text'].strip(),
                }
                new_data.append(new_item)

        except Exception as e:
            print(f"Error processing file {i}: {str(e)}")

    print(f"Total documents: {len(new_data)}")

    with open(args.output_path, 'w', encoding='utf-8') as f:
        for item in new_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Successfully written to {args.output_path}")


if __name__ == "__main__":
    main()
