from transformers import BertTokenizer, BartForConditionalGeneration
from modeling_cpt import CPTForConditionalGeneration
import torch
from tqdm import tqdm
from preprocess import read_text_pairs
from utils import seed_everything
import thwpy
import argparse
from pprint import pprint


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='d1_cpt-base_128_10_best')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--file', type=str, default='data/d1_gen_test.txt')
    parser.add_argument('--score', action="store_true")
    return parser.parse_args()


seed_everything()
args = set_args()
pprint(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = args.model_name
batch_size = args.batch_size
top_k = args.top_k
num_beams = args.num_beams
score = args.score

tokenizer = BertTokenizer.from_pretrained(model_name)
model = CPTForConditionalGeneration.from_pretrained(model_name) if 'cpt' in model_name \
    else BartForConditionalGeneration.from_pretrained(model_name)

model = model.to(device)
model.eval()

split = thwpy.check_split(args.file)
file_prefix = args.file.split('/')[-1].split('.')[0]

full_names, abbrs = read_text_pairs(args.file)

correct = 0
total = 0

results = []
bad_cases = []
with torch.no_grad():
    for i in range(0, len(abbrs), batch_size):
        batch_full_names, batch_abbrs = full_names[i: i + batch_size], abbrs[i: i + batch_size]
        encoded = tokenizer(batch_full_names, return_tensors='pt', padding=True, truncation=True)
        if 'bart' in model_name:
            encoded.pop('token_type_ids')
        encoded = {k: v.to(device) for k, v in encoded.items()}
        outputs = model.generate(**encoded, num_beams=num_beams, num_return_sequences=top_k, temperature=0.05,
                                 repetition_penalty=1.2, output_scores=score, return_dict_in_generate=score)
        batch_generates = tokenizer.batch_decode(outputs.sequences if score else outputs, skip_special_tokens=True)
        if score:
            batch_scores = outputs.sequences_scores.cpu().tolist()
        for j, abbr in enumerate(batch_abbrs):
            generate = batch_generates[j * top_k: (j + 1) * top_k]
            generate = [w.strip().replace(' ', '') for w in generate]
            if score:
                scores = batch_scores[j * top_k: (j + 1) * top_k]
                print(f"Generate: {generate} Score: {scores} Abbr: {abbr}")
                results.append([batch_full_names[j], abbr, ';'.join(generate), ';'.join([str(s) for s in scores])])
            else:
                print(f"Generate: {generate} Abbr: {abbr}")
                results.append([batch_full_names[j], abbr, ';'.join(generate)])
            if abbr in generate:
                correct += 1
            else:
                bad_cases.append([batch_full_names[j], abbr, ';'.join(generate)])
            total += 1
    print(batch_full_names)
    print(batch_abbrs)

thwpy.save_csv(results, f'eval/{model_name}_{split}_beam_{num_beams}_candidates_{top_k}{"_score" if score else ""}.txt', sep='\t')

print(f"Model({split}): {model_name}")
print(f"Hit@{top_k}(beam={num_beams}): {correct} / {total} = ", correct / total)
with open('eval/eval_result.txt', 'a+') as f:
    f.write(f"Model({split}): {model_name}\n")
    f.write(f"Hit@{top_k}(beam={num_beams}): {correct} / {total} = {correct / total}\n")
