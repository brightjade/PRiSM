import argparse
import json
import csv
import os
os.environ["OMP_NUM_THREADS"] = "1"   # restraints the model to 1 cpu
import os.path as osp

import torch

from tqdm import tqdm

import config
from dataset import load_data
from models.utils import load_config, load_tokenizer, load_model
from logger import FileLogger
from evaluation import *
from utils import *


class Evaluator:

    def __init__(self):
        ### Load config / tokenizer / model ###
        self.config = load_config(args)
        self.tokenizer = load_tokenizer(args)

        ### Load data ###
        self.valid_loader, self.valid_features = load_data(args, self.config, self.tokenizer, split="dev")
        self.test_loader, self.test_features = load_data(args, self.config, self.tokenizer, split="test")
        self.theta = args.theta

        ### Load trained parameter weights ###
        ckpt_model_path = osp.join(args.train_output_dir, "best_valid_f1.pt")
        if osp.exists(ckpt_model_path):
            log.console(f"Loading model checkpoint from {ckpt_model_path}...")
            ckpt = torch.load(ckpt_model_path)
            log.console(f"Validation loss was {ckpt['loss']:.4f}")
            log.console(f"Validation avg theta was {ckpt['theta']:.4f}")
            log.console(f"Validation F1 was {ckpt['f1']:.4f}")
            pretrained_dict = {key.replace("module.", ""): value for key, value in ckpt['model_state_dict'].items()}
            self.theta = ckpt['theta']
            self.model = load_model(args, self.config, self.tokenizer)
            self.model.load_state_dict(pretrained_dict)
        else:
            log.event("Predicting with untrained model!")
            self.model = load_model(args, self.config, self.tokenizer)


    @torch.no_grad()
    def evaluate(self, split="dev"):
        self.model.eval()
        dataloader = self.valid_loader if split == "dev" else self.test_loader
        features = self.valid_features if split == "dev" else self.test_features
        total = len(dataloader)
        logits, labels = [], []

        with tqdm(desc="Evaluating", total=total, ncols=100) as pbar:
            for step, inputs in enumerate(dataloader, 1):
                inputs["input_ids"] = inputs["input_ids"].to(args.device)
                inputs["attention_mask"] = inputs["attention_mask"].to(args.device)

                ### Forward pass ###
                outputs = self.model(**inputs)
                _, logit, label = outputs
                logits.append(logit)
                labels.append(label)

                pbar.update(1)
                del outputs

        logits = torch.cat(logits, dim=0)
        labels = torch.cat(labels, dim=0)

        # Remove "no relation" label (idx=0) b/c it was a "fake" label => should not be counted in F1
        logits_eval = logits[:,1:]
        labels_eval = labels[:,1:]

        score_dict = unofficial_evaluate(logits_eval, labels_eval, dataset_name=args.dataset_name)
        if split == "dev":
            self.theta = score_dict["theta"]
        best_f1 = score_dict["F1"]

        ece, ace, prob_true, prob_pred = calibrate(logits, labels)
        log.console(f"ECE: {ece}, ACE: {ace}")

        if args.dataset_name in {"DocRED", "Re-DocRED"}:
            ans = to_official(logits_eval, features, self.theta)
            best_f1, _, best_f1_ign, _ = official_evaluate(ans, args.data_dir, split=split)

        with open(osp.join(args.train_output_dir, "evaluation.txt"), "a") as f:
            f.write(f"{split} F1: {best_f1}\n")
            if args.dataset_name in {"DocRED", "Re-DocRED"}:
                f.write(f"{split} Ign F1: {best_f1_ign}\n")
            f.write(f"{split} Macro F1: {score_dict['macro_F1']}\n")
            f.write(f"{split} Macro F1@500: {score_dict['macro_F1_at_500']}\n")
            f.write(f"{split} Macro F1@200: {score_dict['macro_F1_at_200']}\n")
            f.write(f"{split} Macro F1@100: {score_dict['macro_F1_at_100']}\n")
            f.write(f"{split} ECE: {ece}\n")
            f.write(f"{split} ACE: {ace}\n")
            f.write(f"{split} F1 Per Class: {score_dict['F1_per_class']}\n")

        with open(osp.join(args.train_output_dir, f"calibration_curve_data.csv"), "a") as f:
            writer = csv.writer(f)
            writer.writerow(prob_true.tolist())
            writer.writerow(prob_pred.tolist())


    @torch.no_grad()
    def report(self):
        self.model.eval()
        total = len(self.test_loader)
        preds = []

        with tqdm(desc="Evaluating", total=total, ncols=100) as pbar:
            for step, inputs in enumerate(self.test_loader, 1):
                inputs["input_ids"] = inputs["input_ids"].to(args.device)
                inputs["attention_mask"] = inputs["attention_mask"].to(args.device)

                ### Forward pass ###
                outputs = self.model(**inputs)
                _, pred, _ = outputs
                preds.append(pred)

                pbar.update(1)
                del outputs

        preds = torch.cat(preds, dim=0)[:,1:]
        ans = to_official(preds, self.test_features, self.theta)

        with open(osp.join(args.train_output_dir, "result.json"), "w") as f:
            json.dump(ans, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate.py")
    config.model_args(parser)
    config.data_args(parser)
    config.predict_args(parser)
    args = parser.parse_args()
    args.n_gpu = torch.cuda.device_count()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(osp.join(args.data_dir, "label_map.json"), "r") as f:
        label_map = json.load(f)
    args.num_labels = len(label_map)

    os.makedirs(args.train_output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    log = FileLogger(args.train_output_dir, is_master=True, is_rank0=True, log_to_file=args.log_to_file)
    log.console(args)

    evaluator = Evaluator()
    evaluator.evaluate(split="dev")
    
    if args.dataset_name in {"Re-DocRED", "DWIE"}:
        evaluator.evaluate(split="test")
    elif args.dataset_name == "DocRED":
        evaluator.report()
