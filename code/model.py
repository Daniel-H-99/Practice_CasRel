import torch
import torch.nn as nn

from transformers import BertModel

class E2EModel(nn.Module):
    def __init__(self, num_rels, embed_dim=768):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        # TODO: define subject tagger
        # TODO: define object tagger

    def forward(self, tokens_batch, segments_batch, attention_masks_batch, sub_head_batch, sub_tail_batch):
        # TODO: fill forward module

        ### YOUR CODE ###

        return pred_sub_heads, pred_sub_tails, pred_obj_heads, pred_obj_tails
    
    def get_embedding(self, tokens_batch, segments_batch, attention_masks_batch):
        output = self.bert(input_ids=tokens_batch, token_type_ids=segments_batch, attention_mask=attention_masks_batch)
        bert_embedding = output[0]

        return bert_embedding

    def tag_subject(self, bert_embedding):
        # TODO: fill subject tagging module

        ### YOUR CODE ###

        return pred_sub_heads, pred_sub_tails

    def tag_object(self, bert_embedding, sub_head_batch, sub_tail_batch):
        # TODO: fill object tagging module

        ### YOUR CODE ###

        return pred_obj_heads, pred_obj_tails