import torch
import torch.nn as nn

from transformers import BertModel

# Class for Subject/Object Tagger: receive embeddings and predict probabilities for each tokens.
class Tagger(nn.Module):
    def __init__(self, num_rels, embed_dim=768, is_sub=False):
        super().__init__()
        self.num_rels = num_rels
        self.embed_dim = embed_dim
        self.is_sub = is_sub
        self.fc = nn.Linear(embed_dim, 2 * num_rels)    # prediction layer for prediction of both start/end spots.
        self.fn = nn.Sigmoid()
    def forward(self, embeddings):
        preds = self.fn(self.fc(embeddings))
        d = preds.size()
        assert d[2] == 2 * self.num_rels
        preds = preds.view(d[0], d[1], self.num_rels, 2)
        preds_head = preds[:, :, :, 0]        # separate starte/end prediction
        preds_tail = preds[:, :, :, 1]
        if self.is_sub:
            preds_head, preds_tail = preds_head.squeeze(2), preds_tail.squeeze(2)
            
        return preds_head, preds_tail
        
class E2EModel(nn.Module):
    def __init__(self, num_rels, embed_dim=768):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        # define subject/object tagger
        self.sub_tagger = Tagger(1, is_sub=True)
        self.obj_tagger = Tagger(num_rels)
        
    def forward(self, tokens_batch, segments_batch, attention_masks_batch, sub_head_batch, sub_tail_batch):
        embeddings = self.get_embedding(tokens_batch, segments_batch, attention_masks_batch)
        pred_sub_heads, pred_sub_tails = self.tag_subject(embeddings)
        pred_obj_heads, pred_obj_tails = self.tag_object(embeddings, sub_head_batch, sub_tail_batch)
        
        return pred_sub_heads, pred_sub_tails, pred_obj_heads, pred_obj_tails
    
    def get_embedding(self, tokens_batch, segments_batch, attention_masks_batch):
        output = self.bert(input_ids=tokens_batch, token_type_ids=segments_batch, attention_mask=attention_masks_batch)
        bert_embedding = output[0]

        return bert_embedding

    def tag_subject(self, bert_embedding):
        pred_sub_heads, pred_sub_tails = self.sub_tagger(bert_embedding)
        
        return pred_sub_heads, pred_sub_tails

    def tag_object(self, bert_embedding, sub_head_batch, sub_tail_batch):
        assert sub_head_batch.shape == sub_tail_batch.shape
        new_embeds = []      # collect revise embeddings for specific subject vocabs
        d = bert_embedding.size()
        for batch_idx in range(d[0]):    # select a sentence
            original_embed = bert_embedding[batch_idx]
            sub_heads, sub_tails = sub_head_batch[batch_idx], sub_tail_batch[batch_idx]    # list of subjects in the sentence
            for sub_idx in range(len(sub_heads)):      # select a subject
                sub_head, sub_tail = sub_heads[sub_idx], sub_tails[sub_idx]
                sub_vocab = original_embed[sub_head:sub_tail + 1].mean(dim=0)
                new_embeds.append(original_embed + sub_vocab)     # produce subject-specific embedding of the sentence
        new_embeds = torch.stack(new_embeds)      # construct new batch of revised embeddings
        pred_obj_heads, pred_obj_tails = self.obj_tagger(new_embeds)  # probability of begin start/end spots in size (num_subject x num_words x num_rel), respectivley.
        
        return pred_obj_heads, pred_obj_tails