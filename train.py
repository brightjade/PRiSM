import argparse
import os
os.environ["OMP_NUM_THREADS"] = "1"   # restraints the model to 1 cpu
import os.path as osp
import time

import torch

from tqdm import tqdm
import wandb

import config
from dataset import load_data
from models.utils import load_config, load_tokenizer, load_model
from logger import FileLogger
from evaluation import *
from utils import *


class Trainer:

    def __init__(self):
        ### Load config / tokenizer / model ###
        self.config = load_config(args)
        self.tokenizer = load_tokenizer(args)

        ### Load data ###
        self.train_loader, _ = load_data(args, self.config, self.tokenizer, split="train")
        self.valid_loader, self.valid_features = load_data(args, self.config, self.tokenizer, split="dev")

        self.model = load_model(args, self.config, self.tokenizer)

        ### Calculate steps ###
        args.total_steps = int(len(self.train_loader) * args.epochs // args.gradient_accumulation_steps)
        args.warmup_steps = int(args.total_steps * args.warmup_ratio)
        log.console(f"warmup steps: {args.warmup_steps}, total steps: {args.total_steps}")

        ### scaler / optimizer / scheduler ###
        self.scaler = init_scaler(args)
        self.optimizer = init_optimizer(args, self.model)
        self.scheduler = init_scheduler(args, self.optimizer)

        self.best_valid_loss = float("inf")
        self.best_valid_f1 = float("-inf")
        self.start_epoch = 0
        self.tolerance = 0
        self.global_step = 0

        ### Resume training ###
        ckpt_model_path = osp.join(args.train_output_dir, "best_valid_f1.pt")
        if args.resume and osp.exists(ckpt_model_path):
            log.console(f"Loading model checkpoint from {ckpt_model_path}...")
            ckpt = torch.load(ckpt_model_path)
            self.best_valid_loss = ckpt["loss"]
            self.best_valid_f1 = ckpt["f1"]
            self.start_epoch = ckpt["epoch"]
            self.global_step = ckpt["steps"]
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            log.console(f"Validation loss was {ckpt['loss']:.4f}")
            log.console(f"Validation F1 was {ckpt['f1']:.4f}")
        else:
            log.console(f"Training model from scratch")


    def train(self):
        for epoch in range(self.start_epoch, args.epochs):
            avg_train_loss = self.__epoch_train(epoch)
            avg_valid_loss, score_dict = self.__epoch_valid()

            log.console(f"epoch: {epoch+1}, " +
                        f"steps: {self.global_step}, " +
                        f"current lr: {self.optimizer.param_groups[0]['lr']:.8f}, " +
                        f"train loss: {avg_train_loss:.4f}, " +
                        f"valid loss: {avg_valid_loss:.4f}, " +
                        f"best theta: {score_dict['theta']}")
            log.console(f"P ({score_dict['num_matches']}/{score_dict['num_preds']}): {score_dict['P']:.5f}, " +
                        f"R ({score_dict['num_matches']}/{score_dict['num_labels']}): {score_dict['R']:.5f}, " +
                        f"F1: {score_dict['F1']:.5f}")
            
            if args.wandb_on:
                wandb.log({"Train Loss": avg_train_loss, "Validation Loss": avg_valid_loss,
                           "Precision": score_dict['P'], "Recall": score_dict['R'], "F1": score_dict['F1']})

            if score_dict["F1"] > self.best_valid_f1:
                self.tolerance = 0
                self.best_valid_f1 = score_dict["F1"]
                log.console(f"Saving best valid F1 checkpoint to {args.train_output_dir}...")
                torch.save({'epoch': epoch,
                            'steps': self.global_step,
                            'loss': avg_valid_loss,
                            'p': score_dict['P'],
                            'r': score_dict['R'],
                            'f1': score_dict['F1'],
                            'theta': score_dict['theta'],
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict()
                            }, osp.join(args.train_output_dir, "best_valid_f1.pt"))
                with open(osp.join(args.train_output_dir, "hyparams.txt"), "w") as f:
                    f.write(f"Epoch: {epoch}\n" +
                            f"Total Steps: {self.global_step}\n" +
                            f"Train Loss: {avg_train_loss}\n" +
                            f"Valid Loss: {avg_valid_loss}\n" +
                            f"Theta: {score_dict['theta']}\n" +
                            f"Precision: {score_dict['P']}\n" +
                            f"Recall: {score_dict['R']}\n" +
                            f"F1: {score_dict['F1']}")
            else:
                self.tolerance += 1
                log.console(f"F1 did not improve, patience: {self.tolerance}/{args.max_tolerance}")

            if self.tolerance == args.max_tolerance: break


    def __epoch_train(self, epoch):
        self.model.train()
        train_loss = 0.
        total = len(self.train_loader)

        with tqdm(desc="Training", total=total, ncols=100, disable=args.hide_tqdm) as pbar:
            for step, inputs in enumerate(self.train_loader, 1):
                inputs["input_ids"] = inputs["input_ids"].to(args.device)
                inputs["attention_mask"] = inputs["attention_mask"].to(args.device)

                ### Forward pass ###
                with torch.cuda.amp.autocast(enabled=args.use_amp):
                    loss, _, _ = self.model(**inputs)

                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                
                train_loss += loss.item()

                ### Backward pass ###
                if step % args.gradient_accumulation_steps == 0:
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.global_step += 1

                    if self.global_step == 1 or self.global_step % args.logging_steps == 0:
                        log.console(f"epoch: {epoch+1}, " +
                                    f"steps: {self.global_step}, " +
                                    f"current lr: {self.optimizer.param_groups[0]['lr']:.8f}, " +
                                    f"train loss: {(train_loss / step):.4f}")

                pbar.update(1)
                del loss

        return train_loss / total


    @torch.no_grad()
    def __epoch_valid(self):
        self.model.eval()
        valid_loss = 0.
        total = len(self.valid_loader)
        preds, labels = [], []

        with tqdm(desc="Evaluating", total=total, ncols=100, disable=args.hide_tqdm) as pbar:
            for step, inputs in enumerate(self.valid_loader, 1):
                inputs["input_ids"] = inputs["input_ids"].to(args.device)
                inputs["attention_mask"] = inputs["attention_mask"].to(args.device)

                ### Forward pass ###
                outputs = self.model(**inputs)
                loss, pred, label = outputs
                preds.append(pred)
                labels.append(label)
                valid_loss += loss.item()

                pbar.update(1)
                del outputs

        # Remove "no relation" label (idx=0) b/c it was a "fake" label => should not be counted in F1
        preds = torch.cat(preds, dim=0)[:,1:]
        labels = torch.cat(labels, dim=0)[:,1:]

        score_dict = unofficial_evaluate(preds, labels, dataset_name=args.dataset_name)

        return valid_loss / total, score_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train.py")
    config.model_args(parser)
    config.data_args(parser)
    config.train_args(parser)
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
    if args.wandb_on:
        project_name = f"PRiSM-{args.dataset_name}"
        run_name = "/".join(args.train_output_dir.split("/")[2:])
        wandb.init(project=project_name, name=run_name)

    set_seed(args.seed)

    trainer = Trainer()
    start_time = time.time()
    trainer.train()
    log.console(f"Time for training: {time.time() - start_time:.1f} seconds")
