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

    assert token_ids.size()[0] == 1
    if len(token_ids[0]) > BERT_MAX_LEN:
        token_ids = token_ids[:,:BERT_MAX_LEN]
        segment_ids = segment_ids[:,:BERT_MAX_LEN]
        attention_mask = attention_mask[:,:BERT_MAX_LEN]

    bert_embedding = model.get_embedding(token_ids, segment_ids, attention_mask)

    subjects = []      # list of subject words
    
    # extract indice of candidates for start/end spots
    
    preds_sub_head, preds_sub_tail = model.tag_subject(bert_embedding)
    sub_heads = (preds_sub_head >= h_bar).squeeze(0).nonzero().squeeze(1).cpu().numpy()     # extract indice of possible start/end spots   
    sub_tails = (preds_sub_tail >= t_bar).squeeze(0).nonzero().squeeze(1).cpu().numpy()
    sub_head_batch = [] 
    sub_tail_batch = []
    for sub_head in sub_heads:
        sub_tail = sub_tails[sub_tails >= sub_head]
        if len(sub_tail) > 0:
            sub_tail = sub_tail[0]
            ### YOUR CODE ###
            sub_tokens = tokenizer.convert_ids_to_tokens(token_ids[0, sub_head:sub_tail + 1])
            sub_words = tokenizer.convert_tokens_to_string(sub_tokens).replace(' [unused1] ', ' ').replace(' [unused1]', '').replace('[unused1] ', '')
            subjects.append(sub_words)
            sub_head_batch.append(sub_head)
            sub_tail_batch.append(sub_tail)
            
    # for each subjects, find corresponding tripples (s, r, o)
    if subjects:
        triple_list = []
        
        # e.g. triple_list = [("Bananaman", "creator", "Bright"), ("Bananaman", "starring", "Brooke-Taylor"), ...]

        sub_head_batch, sub_tail_batch = torch.Tensor([sub_head_batch]).type(torch.LongTensor), torch.Tensor([sub_tail_batch]).type(torch.LongTensor)
        pred_obj_heads, pred_obj_tails = model.tag_object(bert_embedding, sub_head_batch, sub_tail_batch)   # predict probs for objects
        pred_obj_heads, pred_obj_tails = (pred_obj_heads.transpose(1, 2) >= h_bar), (pred_obj_tails.transpose(1, 2) >= t_bar)    # (num_subject x num_rel x num_tokens)
        d = pred_obj_heads.size()
        assert len(d) == 3
        assert d[0] == len(subjects)
        for sub_index in range(d[0]):     # select subject
            sub_words = subjects[sub_index]
            for rel_index in range(d[1]):     # select relation
                rel_words = id2rel[rel_index]
                obj_heads, obj_tails = pred_obj_heads[sub_index][rel_index].nonzero().squeeze(1).cpu().numpy(), pred_obj_heads[sub_index][rel_index].nonzero().squeeze(1).cpu().numpy()     # extract candidates for object start/end spots
                for obj_head in obj_heads:
                    obj_tail = obj_tails[obj_tails >= obj_head]
                    if len(obj_tail) > 0:
                        obj_tail = obj_tail[0]
                        obj_tokens = tokenizer.convert_ids_to_tokens(token_ids[0, obj_head:obj_tail + 1])
                        obj_words = tokenizer.convert_tokens_to_string(obj_tokens).replace(' [unused1] ', ' ').replace(' [unused1]', '').replace('[unused1] ', '')
                        triple_list.append((sub_words, rel_words, obj_words))    
        
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

    # compute precision, recall, f1_score from correct_num and predict_num
    
    TP = correct_num
    FN = gold_num - correct_num
    FP = predict_num - correct_num
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 / (1 / precision + 1 / recall)
    
    return precision, recall, f1_score