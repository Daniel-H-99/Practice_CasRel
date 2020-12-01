import os
import argparse
from tqdm import tqdm
import numpy as np

import torch
from transformers import BertTokenizer

from data_loader import load_data, data_generator
from model import E2EModel
from evaluate import metric
from utils import set_seed, str2bool

parser = argparse.ArgumentParser()

parser.add_argument('--train', type=str2bool)

parser.add_argument('--dataset', type=str)
parser.add_argument('--max_sentences', type=int)
parser.add_argument('--max_tokens', type=int)

parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--exact_match', type=str2bool, default=False)
parser.add_argument('--min_delta', type=float, default=1e-4)

parser.add_argument('--save_result', type=str2bool)

parser.add_argument('--data_path', type=str)
parser.add_argument('--ckpt_path', type=str)
parser.add_argument('--log_path', type=str)
parser.add_argument('--results_path', type=str)

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--max_len', type=int, default=100)

args = parser.parse_args()


set_seed(args.seed)

train_path = f'{args.data_path}/{args.dataset}/train_triples.json'
dev_path = f'{args.data_path}/{args.dataset}/valid_triples.json'
test_path = f'{args.data_path}/{args.dataset}/test_triples.json'
rel_dict_path = f'{args.data_path}/{args.dataset}/rel2id.json'

if args.train:
    os.makedirs(f'{args.log_path}', exist_ok=True)
    os.makedirs(f'{args.ckpt_path}', exist_ok=True)

if args.save_result:
    os.makedirs(f'{args.results_path}', exist_ok=True)


train_data, dev_data, test_data, id2rel, rel2id, num_rels = load_data(train_path, dev_path, test_path, rel_dict_path)

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
tokenizer.do_basic_tokenize = False
train_loader = data_generator(train_data, tokenizer, rel2id, num_rels, args.max_len, args.max_sentences, args.max_tokens)

model = E2EModel(num_rels).cuda()
criterion = torch.nn.BCELoss(reduction='none')
optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)


if args.train:
    best_val_f1 = -np.Inf
    best_test_f1 = -np.Inf

    for epoch in range(1, args.epochs+1):
        print(f'Epoch: {epoch}')
        model.train()
        for data_batch in tqdm(train_loader):
            tokens_batch = torch.from_numpy(data_batch[0][0]).cuda()
            segments_batch = torch.from_numpy(data_batch[0][1]).cuda()
            attention_masks_batch = torch.from_numpy(data_batch[0][2]).cuda()
            sub_heads_batch = torch.from_numpy(data_batch[0][3]).type(torch.FloatTensor).cuda()
            sub_tails_batch = torch.from_numpy(data_batch[0][4]).type(torch.FloatTensor).cuda()
            sub_head_batch = torch.from_numpy(data_batch[0][5]).cuda()
            sub_tail_batch = torch.from_numpy(data_batch[0][6]).cuda()
            obj_heads_batch = torch.from_numpy(data_batch[0][7]).type(torch.FloatTensor).cuda()
            obj_tails_batch = torch.from_numpy(data_batch[0][8]).type(torch.FloatTensor).cuda()

            padding_mask_batch = tokens_batch.ne(0)

            pred_sub_heads, pred_sub_tails, pred_obj_heads, pred_obj_tails = model(tokens_batch, segments_batch, attention_masks_batch, sub_head_batch, sub_tail_batch)

            sub_heads_loss = criterion(pred_sub_heads, sub_heads_batch)
            sub_heads_loss = sub_heads_loss[padding_mask_batch].mean()

            sub_tails_loss = criterion(pred_sub_tails, sub_tails_batch)
            sub_tails_loss = sub_tails_loss[padding_mask_batch].mean()

            obj_heads_loss = criterion(pred_obj_heads, obj_heads_batch)
            obj_heads_loss = obj_heads_loss.sum(dim=-1)
            obj_heads_loss = obj_heads_loss[padding_mask_batch].mean()

            obj_tails_loss = criterion(pred_obj_tails, obj_tails_batch)
            obj_tails_loss = obj_tails_loss.sum(dim=-1)
            obj_tails_loss = obj_tails_loss[padding_mask_batch].mean()

            loss = (sub_heads_loss + sub_tails_loss) + (obj_heads_loss + obj_tails_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_prec, val_rec, val_f1 = metric(model, dev_data, id2rel, tokenizer, exact_match=args.exact_match)
        test_prec, test_rec, test_f1 = metric(model, test_data, id2rel, tokenizer, exact_match=args.exact_match)

        if np.greater(val_f1, best_val_f1 + args.min_delta) or np.greater(args.min_delta, val_f1):
            best_val_prec = val_prec
            best_val_rec = val_rec
            best_val_f1 = val_f1
            report_test_prec = test_prec
            report_test_rec = test_rec
            report_test_f1 = test_f1

            torch.save(model.state_dict(), f'{args.ckpt_path}/{args.dataset}_best_model.ckpt')
        
        if np.greater(test_f1, best_test_f1 + args.min_delta) or np.greater(args.min_delta, test_f1):
            best_test_f1 = test_f1
        
        with open(f'{args.log_path}/{args.dataset}_train_log.tsv', 'a') as log_f:
            log_f.write(f'{epoch:5}\t{val_f1:.5f}\t{test_f1:.5f}\t'
                        + f'{best_val_prec:.5f}\t{best_val_rec:.5f}\t{best_val_f1:.5f}\t'
                        + f'{report_test_prec:.5f}\t{report_test_rec:.5f}\t{report_test_f1:.5f}\t'
                        + f'{best_test_f1:.5f}\n')

        print(f'f1: {val_f1:.4f}, best f1: {best_val_f1:.4f}, test f1: {report_test_f1:.4f}, best test f1: {best_test_f1:.4f}\n')

else:
    best_state_dict = torch.load(f'{args.ckpt_path}/{args.dataset}_best_model.ckpt')
    model.load_state_dict(best_state_dict)
    model.eval()

    if args.save_result:
        test_result_path = f'{args.results_path}/{args.dataset}_test_result.json'
    else:
        test_result_path = None

    test_prec, test_rec, test_f1 = metric(model, test_data, id2rel, tokenizer, exact_match=args.exact_match, output_path=test_result_path)

    print(f'test precision: {test_prec:.4f}, test recall: {test_rec:.4f}, test f1: {test_f1:.4f}')
