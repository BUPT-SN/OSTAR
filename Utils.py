import csv
import json

input_csv = "/data/Content_Moderation/paper_data/checkgpt/test.csv"
output_authorship = "/data/Content_Moderation/paper_data/checkgpt/test_Authorship.jsonl"
output_robust = "/data/Content_Moderation/paper_data/checkgpt/test_Robust.jsonl"

def process_labels():
    authorship_data = []
    robust_data = []
    
    with open(input_csv, mode='r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        
        for row in csv_reader:
            original_label = row.get('label', '').strip().lower()
            
            if original_label in {'human', 'machine_origin'}:
                authorship_data.append(row.copy())
                robust_data.append(row.copy())
            
            elif original_label == 'paraphrase':
                modified_row = row.copy()
                modified_row['label'] = 'machine'
                authorship_data.append(modified_row)
            
            else:
                modified_row = row.copy()
                modified_row['label'] = 'machine'
                robust_data.append(modified_row)
    
    with open(output_authorship, 'w', encoding='utf-8') as f:
        for item in authorship_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    with open(output_robust, 'w', encoding='utf-8') as f:
        for item in robust_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    process_labels()