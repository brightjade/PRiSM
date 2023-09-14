import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PRISM(nn.Module):

    def __init__(self, args, config, relation_features, encoder1, encoder2):
        super().__init__()
        self.args = args
        self.config = config
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.relation_features = relation_features
        args.embedding_size = config.hidden_size    # 768 for BERT, 1024 for RoBERTa

        self.head_extractor = nn.Linear(config.hidden_size, args.embedding_size)
        self.tail_extractor = nn.Linear(config.hidden_size, args.embedding_size)
        self.pair_extractor = nn.Linear(2 * args.embedding_size, config.hidden_size)
        self.loss_fnt = nn.BCEWithLogitsLoss(reduction="mean")

        if args.group_bilinear:
            self.bilinear = nn.Linear(args.embedding_size * args.block_size, args.num_labels)
        else:
            self.bilinear = nn.Bilinear(args.embedding_size, args.embedding_size, args.num_labels)


    def __process_long_input(self, input_ids, attention_mask):
        N, T = input_ids.shape
        new_input_ids, new_attention_mask = [], []
        num_chunks = math.ceil(T / self.args.max_seq_length)
        # Split into chunks for training
        for i in range(N):
            for c in range(num_chunks):
                start = c * self.args.max_seq_length
                end = (c+1) * self.args.max_seq_length
                # pad the last chunk
                if len(input_ids[i, start:end]) < self.args.max_seq_length:
                    to_pad = self.args.max_seq_length - len(input_ids[i, start:end])
                    new_input_ids.append(F.pad(input_ids[i, start:end], (0, to_pad), value=self.config.pad_token_id))
                    new_attention_mask.append(F.pad(attention_mask[i, start:end], (0, to_pad), value=0))
                else:
                    new_input_ids.append(input_ids[i, start:end])
                    new_attention_mask.append(attention_mask[i, start:end])

        input_ids = torch.stack(new_input_ids, dim=0)
        attention_mask = torch.stack(new_attention_mask, dim=0)
        out = self.encoder1(input_ids, attention_mask, output_attentions=True)
        out.last_hidden_state = out.last_hidden_state.reshape(N, num_chunks * self.args.max_seq_length, -1)
        return out


    def __get_entity_embeddings(self, output, ent_pos, ent_pairs):
        offset = 1  # cls token shifts ent_pos by 1
        last_hidden = output.last_hidden_state
        N, T, D = last_hidden.shape 
        head_embeddings, tail_embeddings = [], []

        for i in range(N):
            ent_embs = []
            for ent in ent_pos[i]:
                if len(ent) == 0:
                    e_emb = last_hidden.new_zeros(D)
                elif len(ent) > 1:    # more than 1 mention
                    e_emb = []
                    for start, end in ent:
                        if start + offset < T:  # In case the entity mention is truncated due to limited max seq length
                            if self.args.mark_entities:
                                e_emb.append(last_hidden[i, start+offset])
                            else:   # max-pool all token embeddings to represent entity embedding
                                m_emb, _ = torch.max(last_hidden[i, start+offset:end+offset], dim=0)
                                e_emb.append(m_emb)
                    
                    if len(e_emb) > 0:
                        e_emb = torch.stack(e_emb, dim=0)
                        if self.args.ent_pooler == "logsumexp":
                            e_emb = torch.logsumexp(e_emb, dim=0)
                        elif self.args.ent_pooler == "max":
                            e_emb, _ = torch.max(e_emb, dim=0)
                        elif self.args.ent_pooler == "sum":
                            e_emb = torch.sum(e_emb, dim=0)
                        elif self.args.ent_pooler == "avg":
                            e_emb = torch.mean(e_emb, dim=0)
                        else:
                            raise ValueError("Supported pooling operations: logsumexp, max, sum, avg")
                    else:
                        e_emb = last_hidden.new_zeros(D)
                else:
                    start, end = ent[0]
                    if start + offset < T:
                        if self.args.mark_entities:
                            e_emb = last_hidden[i, start+offset]
                        else:   # max-pool all token embeddings to represent entity embedding
                            e_emb, _ = torch.max(last_hidden[i, start+offset:end+offset], dim=0)
                    else:
                        e_emb = last_hidden.new_zeros(D)
                
                ent_embs.append(e_emb)
        
            ent_embs = torch.stack(ent_embs, dim=0)     # (num_ents, D)

            # Get embeddings of all possible entity pairs
            ent_pairs_i = torch.tensor(ent_pairs[i], dtype=torch.long, device=last_hidden.device)
            head_embs = torch.index_select(ent_embs, dim=0, index=ent_pairs_i[:, 0])
            tail_embs = torch.index_select(ent_embs, dim=0, index=ent_pairs_i[:, 1])

            head_embeddings.append(head_embs)
            tail_embeddings.append(tail_embs)

        head_embeddings = torch.cat(head_embeddings, dim=0)
        tail_embeddings = torch.cat(tail_embeddings, dim=0)
        
        return head_embeddings, tail_embeddings


    def __get_relation_embeddings(self, r_out):
        if self.args.rel_pooler == "pooler":
            return r_out.pooler_output
        elif self.args.rel_pooler == "cls":
            return r_out.last_hidden_state[:,0,:]
        else:
            raise ValueError("Supported pooling operations: pooler, cls.")


    def __compute_ht_scores(self, head_embs, tail_embs):
        if self.args.group_bilinear:
            b1 = head_embs.view(-1, self.args.embedding_size // self.args.block_size, self.args.block_size).unsqueeze(3)
            b2 = tail_embs.view(-1, self.args.embedding_size // self.args.block_size, self.args.block_size).unsqueeze(2)
            bl = (b1 * b2).view(-1, self.args.embedding_size * self.args.block_size)
            scores = self.bilinear(bl)
        else:
            scores = self.bilinear(head_embs, tail_embs)
        return scores


    def __compute_pr_scores(self, pair_embs, rel_embs):
        scores = pair_embs @ rel_embs.T
        normalized_pair_embs = F.normalize(pair_embs, p=2, dim=-1)
        normalized_rel_embs = F.normalize(rel_embs, p=2, dim=-1)
        normalized_scores = normalized_pair_embs @ normalized_rel_embs.T
        return scores, normalized_scores


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        ent_pos=None,
        ent_pairs=None,
        labels=None,
    ):
        # multi-label classification
        N, T = input_ids.shape
        if self.args.long_seq and T > self.args.max_seq_length:
            out = self.__process_long_input(input_ids, attention_mask)
        else:
            out = self.encoder1(input_ids, attention_mask, output_attentions=True)

        h_embs, t_embs = self.__get_entity_embeddings(out, ent_pos, ent_pairs)
        h_embs = torch.tanh(self.head_extractor(h_embs))
        t_embs = torch.tanh(self.tail_extractor(t_embs))
        logits = self.__compute_ht_scores(h_embs, t_embs)

        # pair-relation similarity
        if self.args.share_params:
            r_out = self.encoder1(self.relation_features["input_ids"].to(input_ids),
                                  self.relation_features["attention_mask"].to(input_ids))
        else:
            r_out = self.encoder2(self.relation_features["input_ids"].to(input_ids),
                                  self.relation_features["attention_mask"].to(input_ids))
        
        r_embs = self.__get_relation_embeddings(r_out)
        p_embs = torch.cat([h_embs, t_embs], dim=-1)
        p_embs = torch.tanh(self.pair_extractor(p_embs))
        pr_logits, normalized_pr_logits = self.__compute_pr_scores(p_embs, r_embs)

        if self.args.dist_fn == "inner":
            logits = logits + pr_logits
        elif self.args.dist_fn == "cosine":
            logits = logits + (normalized_pr_logits / self.args.temperature)

        model_output = (torch.sigmoid(logits),)

        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            loss = self.loss_fnt(logits, labels)
            model_output = (loss,) + model_output + (labels,)
        
        return model_output
