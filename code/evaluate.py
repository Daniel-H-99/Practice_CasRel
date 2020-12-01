import numpy as np
from tqdm import tqdm
import json

import torch

from utils import match_tokenizer

BERT_MAX_LEN = 512


def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1


@torch.no_grad()
def extract_items(model, tokenizer, data_point, id2rel, h_bar=0.5, t_bar=0.5):
    text_in = data_point['text']

    tokens = tokenizer.tokenize(match_tokenizer(text_in))
    if len(tokens) > BERT_MAX_LEN:
        tokens = tokens[:BERT_MAX_LEN]

    encoded_input = tokenizer(match_tokenizer(text_in, encode=True), return_tensors='pt')
    token_ids = encoded_input['input_ids'].cuda()
    segment_ids = encoded_input['token_type_ids'].cuda()
    attention_mask = encoded_input['attention_mask'].cuda()

    if len(token_ids[0]) > BERT_MAX_LEN:
        token_ids = token_ids[:,:BERT_MAX_LEN]
        segment_ids = segment_ids[:,:BERT_MAX_LEN]
        attention_mask = attention_mask[:,:BERT_MAX_LEN]

    bert_embedding = model.get_embedding(token_ids, segment_ids, attention_mask)

    # TODO: fill subjects
    # HINT: sub_head and sub_tail are np.array of int

    ### YOUR CODE ###

    subjects = []
    for sub_head in sub_heads:
        sub_tail = sub_tails[sub_tails >= sub_head]
        if len(sub_tail) > 0:
            sub_tail = sub_tail[0]

            ### YOUR CODE ###

    if subjects:
        triple_list = []

        # TODO: fill triple_list with (subject, relation, object) tuples
        # e.g. triple_list = [("Bananaman", "creator", "Bright"), ("Bananaman", "starring", "Brooke-Taylor"), ...]

        ### YOUR CODE ###

        return triple_list
    else:
        return []


def partial_match(pred_set, gold_set):
    pred = {(i[0].split(' ')[0] if len(i[0].split(' ')) > 0 else i[0], i[1], i[2].split(' ')[0] if len(i[2].split(' ')) > 0 else i[2]) for i in pred_set}
    gold = {(i[0].split(' ')[0] if len(i[0].split(' ')) > 0 else i[0], i[1], i[2].split(' ')[0] if len(i[2].split(' ')) > 0 else i[2]) for i in gold_set}
    return pred, gold


def metric(model, eval_data, id2rel, tokenizer, exact_match=False, output_path=None):
    if output_path:
        F = open(output_path, 'w')
    orders = ['subject', 'relation', 'object']
    correct_num, predict_num, gold_num = 1e-10, 1e-10, 1e-10
    for line in tqdm(eval_data):
        Pred_triples = set(extract_items(model, tokenizer, line, id2rel))
        Gold_triples = set(line['triple_list'])

        Pred_triples_eval, Gold_triples_eval = partial_match(Pred_triples, Gold_triples) if not exact_match else (Pred_triples, Gold_triples)

        correct_num += len(Pred_triples_eval & Gold_triples_eval)
        predict_num += len(Pred_triples_eval)
        gold_num += len(Gold_triples_eval)

        if output_path:
            result = json.dumps({
                'text': line['text'],
                'triple_list_gold': [
                    dict(zip(orders, triple)) for triple in Gold_triples
                ],
                'triple_list_pred': [
                    dict(zip(orders, triple)) for triple in Pred_triples
                ],
                'new': [
                    dict(zip(orders, triple)) for triple in Pred_triples - Gold_triples
                ],
                'lack': [
                    dict(zip(orders, triple)) for triple in Gold_triples - Pred_triples
                ]
            }, ensure_ascii=False, indent=4)
            F.write(result + '\n')
    if output_path:
        F.close()

    # TODO: compute precision, recall, f1_score from correct_num and predict_num

    ### YOUR CODE ###

    return precision, recall, f1_score