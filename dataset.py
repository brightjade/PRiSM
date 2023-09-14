import logging
import os
import os.path as osp
import json
import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from utils import seed_worker


def load_data(args, config, tokenizer, split="train"):

    if args.dataset_name in {"DocRED", "Re-DocRED"}:
        dataset = DocREDDataset(args, config, tokenizer, split)
    elif args.dataset_name == "DWIE":
        dataset = DWIEDataset(args, config, tokenizer, split)
    else:
        raise ValueError("Dataset must be DocRED, Re-DocRED, or DWIE.")
    
    if split == "train":
        dataloader = DataLoader(dataset,
                                batch_size=args.train_batch_size,
                                collate_fn=dataset.collate_fn,
                                worker_init_fn=seed_worker,
                                num_workers=args.num_workers,
                                shuffle=True,
                                drop_last=True,
                                pin_memory=True)
    elif split == "dev":
        dataloader = DataLoader(dataset,
                                batch_size=args.eval_batch_size,
                                collate_fn=dataset.collate_fn,
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True)
    elif split =="test":
        dataloader = DataLoader(dataset,
                                batch_size=args.test_batch_size,
                                collate_fn=dataset.collate_fn,
                                shuffle=False,
                                drop_last=False)
    else:
        raise ValueError("Data split must be either train/dev/test.")
    
    return dataloader, dataset.features


def load_and_cache_relations(args, config, tokenizer):
    save_dir = osp.join(args.data_dir, "cached")
    save_path = osp.join(save_dir, f"{args.model_name_or_path}_reldesc{args.num_labels}.pt")

    os.makedirs(save_dir, exist_ok=True)
    if osp.exists(save_path):
        logging.info(f"Loading relation features from {save_path}")
        return torch.load(save_path)

    with open(osp.join(args.data_dir, f"rel_desc.json")) as f:
        relations = json.load(f)
    
    with open(osp.join(args.data_dir, f"label_map.json")) as f:
        label_map = json.load(f)

    relation_features = [None] * args.num_labels
    for rel_id, relation in relations.items():
        input_ids = tokenizer.encode(relation)
        attention_mask = [1] * len(input_ids)
        relation_features[label_map[rel_id]] = {"input_ids": input_ids, "attention_mask": attention_mask}

    # Collate
    PAD = config.pad_token_id
    max_len = max([len(r["input_ids"]) for r in relation_features])
    input_ids = [r["input_ids"] + [PAD] * (max_len - len(r["input_ids"])) for r in relation_features]
    attention_mask = [r["attention_mask"] + [0] * (max_len - len(r["attention_mask"])) for r in relation_features]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)

    relation_features = {"input_ids": input_ids, "attention_mask": attention_mask}
    logging.info(f"Saving relation features to {save_path}")
    torch.save(relation_features, save_path)

    return relation_features


class DocREDDataset(Dataset):

    def __init__(self, args, config, tokenizer, split="train"):
        self.args = args
        self.config = config
        self.tokenizer = tokenizer
        self.split = split
        self.features = []

        self.ent_marked = "_entmarked" if args.mark_entities else ""
        self.save_dir = osp.join(args.data_dir, "cached")
        self.save_path = osp.join(self.save_dir, f"{split}_{args.model_name_or_path}{self.ent_marked}.pt")
        os.makedirs(self.save_dir, exist_ok=True)

        self.ner_map = {'PAD':0, 'ORG':1, 'LOC':2, 'NUM':3, 'TIME':4, 'MISC':5, 'PER':6}
        with open(osp.join(args.data_dir, "label_map.json"), "r") as f:
            self.label_map = json.load(f)

        self.__load_and_cache_examples()

        # Set up resource-constrained setting
        if self.split == "train" and args.num_train_ratio < 1:
            num_train = round(len(self.features) * self.args.num_train_ratio)
            # keep random sampling until label distribution resembles that of the full data
            if args.dataset_name == "DocRED":
                label_freq = [1163035, 264, 8921, 4193, 2004, 2689, 1044, 511, 79, 475, 79, 275, 356, 172, 76, 194, 539, 35, 583, 632, 414, 1052, 1142, 621, 95, 203, 316, 805, 196, 173, 210, 596, 85, 303, 74, 273, 360, 119, 155, 150, 238, 304, 104, 406, 96, 62, 335, 298, 246, 156, 82, 188, 192, 166, 108, 208, 185, 23, 163, 144, 299, 231, 152, 79, 63, 223, 110, 51, 36, 379, 320, 48, 111, 85, 137, 119, 191, 140, 144, 33, 66, 9, 77, 103, 95, 100, 172, 83, 92, 92, 2, 75, 36, 36, 18, 2, 4]
            elif args.dataset_name == "Re-DocRED":
                label_freq = [1125284, 263, 14401, 20402, 3369, 4665, 1172, 692, 155, 868, 181, 575, 761, 336, 178, 431, 948, 66, 923, 2313, 1299, 1773, 1621, 919, 200, 281, 503, 1000, 421, 340, 368, 2112, 178, 640, 168, 466, 703, 281, 366, 3055, 402, 460, 204, 403, 191, 102, 712, 1207, 341, 237, 152, 506, 506, 305, 191, 389, 356, 49, 370, 245, 669, 410, 264, 171, 145, 1168, 222, 105, 79, 379, 489, 83, 239, 174, 293, 249, 1168, 292, 357, 59, 107, 22, 152, 225, 192, 204, 298, 144, 230, 230, 2, 117, 65, 96, 96, 8, 8]
            label_dist = torch.tensor(label_freq) / sum(label_freq)

            sampled_features = random.sample(self.features, num_train)
            sampled_freq = torch.stack([torch.tensor(x["labels"]).sum(0) for x in sampled_features]).sum(0)
            sampled_dist = sampled_freq / sampled_freq.sum()
            while not torch.allclose(label_dist, sampled_dist, atol=1e-03):
                sampled_features = random.sample(self.features, num_train)
                sampled_freq = torch.stack([torch.tensor(x["labels"]).sum(0) for x in sampled_features]).sum(0)
                sampled_dist = sampled_freq / sampled_freq.sum()

            self.features = sampled_features 


    def __load_and_cache_examples(self):
        if osp.exists(self.save_path):
            logging.info(f"Loading features from {self.save_path}")
            self.features = torch.load(self.save_path)
            return
        
        logging.info(f"Creating features to {self.save_path}")
        with open(osp.join(self.args.data_dir, f"{self.split}.json")) as f:
            examples = json.load(f)

        num_pos_samples, num_neg_samples = 0, 0

        for ex in tqdm(examples, desc="Converting examples to features"):
            ents = ex["vertexSet"]
            
            # Locate start & end of entity mention for entity marking
            ent_start, ent_end = set(), set()
            if self.args.mark_entities:
                for ent in ents:
                    for ment in ent:
                        ent_start.add((ment["sent_id"], ment["pos"][0]))
                        ent_end.add((ment["sent_id"], ment["pos"][1]-1))

            # Map each word idx to subword idx
            input_tokens = []
            token_idx_map = []
            tok_to_sent = []
            for sent_idx, sent in enumerate(ex["sents"]):
                idx_map = {}
                for word_idx, word in enumerate(sent):
                    tokens = self.tokenizer.tokenize(word)
                    if (sent_idx, word_idx) in ent_start:
                        tokens = ["*"] + tokens
                    if (sent_idx, word_idx) in ent_end:
                        tokens = tokens + ["*"]
                    idx_map[word_idx] = len(input_tokens)
                    tok_to_sent += [sent_idx] * len(tokens)
                    input_tokens += tokens
                idx_map[word_idx+1] = len(input_tokens)
                token_idx_map.append(idx_map)

            input_tokens = input_tokens[:self.args.max_seq_length-2]                # truncate to max sequence length
            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)          # convert tokens to ids
            input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)  # add [CLS] & [SEP]
            tok_to_sent = [None] + tok_to_sent + [None]

            # Locate spans of each entity mention
            ent_pos = []
            for ent in ents:
                ent_pos.append([])
                for ment in ent:
                    # Get subword idx of the entity mention
                    token_start_pos = token_idx_map[ment["sent_id"]][ment["pos"][0]]
                    token_end_pos = token_idx_map[ment["sent_id"]][ment["pos"][1]]
                    ent_pos[-1].append((token_start_pos, token_end_pos))

            ground_truth_triples = defaultdict(list)
            if ex.get("labels"):    # test file does not have "labels"
                for label in ex["labels"]:
                    rel_id = self.label_map[label["r"]]
                    ground_truth_triples[(label["h"], label["t"])].append({"relation": rel_id, "evidence": label["evidence"]})
            
            # Create positive pairs
            ent_pairs, rel_vectors = [], []
            for (h, t), instances in ground_truth_triples.items():
                rel_vector = [0] * len(self.label_map)
                for instance in instances:
                    rel_vector[instance["relation"]] = 1
                rel_vectors.append(rel_vector)
                ent_pairs.append((h, t))
                num_pos_samples += 1

            # Create negative pairs
            for h in range(len(ents)):
                for t in range(len(ents)):
                    if h != t and (h, t) not in ent_pairs:
                        rel_vector = [1] + [0] * (len(self.label_map)-1)
                        rel_vectors.append(rel_vector)
                        ent_pairs.append((h, t))
                        num_neg_samples += 1

            assert len(rel_vectors) == len(ent_pairs) == (len(ents) * (len(ents)-1))

            self.features.append({
                "input_ids": input_ids,
                "ent_pos": ent_pos,
                "ent_pairs": ent_pairs,
                "title": ex["title"],   # needed for test submission
                "labels": rel_vectors,
            })

        logging.info(f"# of documents: {len(self.features)}")
        logging.info(f"# of positive pairs {num_pos_samples}")
        logging.info(f"# of negative pairs {num_neg_samples}")
        logging.info(f"Saving features to {self.save_path}")
        torch.save(self.features, self.save_path)


    def collate_fn(self, samples):
        PAD = self.config.pad_token_id
        max_len = max([len(x["input_ids"]) for x in samples])
        input_ids = [x["input_ids"] + [PAD] * (max_len - len(x["input_ids"])) for x in samples]
        attention_mask = [[1] * len(x["input_ids"]) + [0] * (max_len - len(x["input_ids"])) for x in samples]
        
        ent_pos = [x["ent_pos"] for x in samples]
        ent_pairs = [x["ent_pairs"] for x in samples]
        labels = [x["labels"] for x in samples]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "ent_pos": ent_pos,
                "ent_pairs": ent_pairs,
                "labels": labels}

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]


class DWIEDataset(Dataset):
    
    def __init__(self, args, config, tokenizer, split="train"):
        self.args = args
        self.config = config
        self.tokenizer = tokenizer
        self.split = split
        self.features = []

        self.ent_marked = "_entmarked" if args.mark_entities else ""
        self.long = "_long" if args.long_seq == 1 else ""
        self.save_dir = osp.join(args.data_dir, "cached")
        self.save_path = osp.join(self.save_dir, f"{split}_{args.model_name_or_path}{self.ent_marked}{self.long}.pt")
        os.makedirs(self.save_dir, exist_ok=True)

        with open(osp.join(args.data_dir, "label_map.json"), "r") as f:
            self.label_map = json.load(f)

        self.__load_and_cache_examples()

        # Set up resource-constrained setting
        if self.split == "train":
            num_train = round(len(self.features) * self.args.num_train_ratio)
            # keep random sampling until label distribution resembles that of the full data            import pdb; pdb.set_trace()
            label_freq = [601051, 83, 133, 470, 1403, 751, 1572, 1518, 307, 291, 211, 193, 1255, 2005, 1597, 137, 184, 170, 1703, 1206, 158, 361, 326, 68, 5, 11, 99, 18, 242, 253, 51, 57, 367, 32, 123, 30, 4, 21, 126, 87, 16, 43, 25, 7, 27, 6, 16, 16, 16, 11, 43, 30, 12, 7, 5, 9, 2, 2, 3, 0, 2, 1, 0, 0, 1, 1]
            label_dist = torch.tensor(label_freq) / sum(label_freq)
            sampled_features = random.sample(self.features, num_train)
            sampled_freq = torch.stack([torch.tensor(x["labels"]).sum(0) for x in sampled_features]).sum(0)
            sampled_dist = sampled_freq / sampled_freq.sum()
            while not torch.allclose(label_dist, sampled_dist, atol=1e-03):
                sampled_features = random.sample(self.features, num_train)
                sampled_freq = torch.stack([torch.tensor(x["labels"]).sum(0) for x in sampled_features]).sum(0)
                sampled_dist = sampled_freq / sampled_freq.sum()

            self.features = sampled_features


    def __load_and_cache_examples(self):
        if osp.exists(self.save_path):
            logging.info(f"Loading features from {self.save_path}")
            self.features = torch.load(self.save_path)
            return

        num_pos_samples, num_neg_samples = 0, 0

        logging.info(f"Creating features to {self.save_path}")
        for filename in tqdm(os.listdir(osp.join(self.args.data_dir, self.split)), desc="Converting examples to features"):
            if osp.isfile(osp.join(self.args.data_dir, self.split, filename)):
                with open(osp.join(self.args.data_dir, self.split, filename), 'r') as f:
                    ex = json.load(f)

                start = 0
                input_tokens = []
                token_idx_map = defaultdict(list)
                for ment in ex["mentions"]:
                    # tokenize text up to entity mention
                    end = ment["begin"]
                    words = ex["content"][start:end].strip()
                    before_tokens = self.tokenizer.tokenize(words)
                    
                    # tokenize entity mention
                    start, end = ment["begin"], ment["end"]
                    ment_word = ex["content"][start:end].strip()
                    ment_tokens = self.tokenizer.tokenize(ment_word)
                    if self.args.mark_entities:
                        ment_tokens = ["*"] + ment_tokens + ["*"]
                    
                    # For each entity, store the token position (start, end) of mention
                    token_start_pos = len(input_tokens) + len(before_tokens)
                    token_end_pos = token_start_pos + len(ment_tokens)
                    token_idx_map[ment["concept"]].append((token_start_pos, token_end_pos))

                    input_tokens += before_tokens + ment_tokens
                    start = ment["end"]

                # Finish tokenizing the text
                after_tokens = self.tokenizer.tokenize(ex["content"][start:].strip())
                input_tokens += after_tokens
                if not self.args.long_seq:
                    input_tokens = input_tokens[:self.args.max_seq_length-2]                # truncate to max sequence length
                input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)          # convert tokens to ids
                input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)  # add [CLS] & [SEP]

                # Convert to ent_pos format (following DocRED)
                ent_pos = []
                for i in range(len(ex["concepts"])):
                    if token_idx_map.get(i):
                        ent_pos.append(token_idx_map[i])
                    else:   # there are annotated entities that do not exist in the document
                        ent_pos.append([])

                ground_truth_triples = defaultdict(list)
                if ex.get("relations"):
                    for label in ex["relations"]:
                        rel_id = self.label_map[label["p"]]
                        # Remove relations where entities do not exist in the document
                        if len(ent_pos[label["s"]]) != 0 and len(ent_pos[label["o"]]) != 0:
                            ground_truth_triples[(label["s"], label["o"])].append(rel_id)

                # Create positive pairs
                ent_pairs, rel_vectors = [], []
                for (h, t), relations in ground_truth_triples.items():
                    rel_vector = [0] * len(self.label_map)
                    for r in relations:
                        rel_vector[r] = 1
                    rel_vectors.append(rel_vector)
                    ent_pairs.append((h, t))
                    num_pos_samples += 1

                # Create negative pairs
                for h in range(len(ex["concepts"])):
                    for t in range(len(ex["concepts"])):
                        if h != t and (h, t) not in ent_pairs and len(ent_pos[h]) != 0 and len(ent_pos[t]) != 0:
                        # if h != t and (h, t) not in ent_pairs:
                            rel_vector = [1] + [0] * (len(self.label_map)-1)
                            rel_vectors.append(rel_vector)
                            ent_pairs.append((h, t))
                            num_neg_samples += 1

                assert len(rel_vectors) == len(ent_pairs)

                self.features.append({
                    "input_ids": input_ids,
                    "ent_pos": ent_pos,
                    "ent_pairs": ent_pairs,
                    "labels": rel_vectors,
                })

        logging.info(f"# of documents: {len(self.features)}")
        logging.info(f"# of positive pairs {num_pos_samples}")
        logging.info(f"# of negative pairs {num_neg_samples}")
        logging.info(f"Saving features to {self.save_path}")
        torch.save(self.features, self.save_path)


    def collate_fn(self, samples):
        PAD = self.config.pad_token_id
        max_len = max([len(x["input_ids"]) for x in samples])
        input_ids = [x["input_ids"] + [PAD] * (max_len - len(x["input_ids"])) for x in samples]
        attention_mask = [[1] * len(x["input_ids"]) + [0] * (max_len - len(x["input_ids"])) for x in samples]
        
        ent_pos = [x["ent_pos"] for x in samples]
        ent_pairs = [x["ent_pairs"] for x in samples]
        labels = [x["labels"] for x in samples]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "ent_pos": ent_pos,
                "ent_pairs": ent_pairs,
                "labels": labels}

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]
