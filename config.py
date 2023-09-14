"""
Configurations.
"""

def model_args(parser):
    parser.add_argument("--model_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)
    parser.add_argument("--config_name", default="", type=str)
    parser.add_argument("--tokenizer_name", default="", type=str)
    parser.add_argument("--cache_dir", default=".cache/", type=str)
    parser.add_argument("--ent_pooler", default="logsumexp", type=str)
    parser.add_argument("--rel_pooler", default="cls", type=str)
    parser.add_argument("--dist_fn", default="", type=str)
    parser.add_argument("--temperature", default=0.1, type=float)
    parser.add_argument("--embedding_size", default=768, type=int)
    parser.add_argument("--block_size", default=64, type=int)
    parser.add_argument("--group_bilinear", default=True, type=bool)
    parser.add_argument("--share_params", default=1, type=int)
    parser.add_argument("--long_seq", default=0, type=int)


def data_args(parser):
    parser.add_argument("--log_dir", type=str, default=".logs/")
    parser.add_argument("--dataset_name", default="DocRED", type=str)
    parser.add_argument("--data_dir", default="data/DocRED", type=str)
    parser.add_argument("--train_output_dir", default=".checkpoints/", type=str)
    parser.add_argument("--test_output_dir", default=".checkpoints/", type=str)
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--num_labels", default=97, type=int)
    parser.add_argument("--num_train_ratio", default=1.0, type=float)
    parser.add_argument("--mark_entities", default=True, type=bool)
    parser.add_argument("--log_to_file", default=0, type=int)


def train_args(parser):
    parser.add_argument("--train_batch_size", default=4, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--logging_steps", default=100, type=str)
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("--clf_lr", default=1e-4, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--warmup_ratio", default=0.06, type=float)
    parser.add_argument("--max_tolerance", default=5, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--use_amp", default=True, type=bool)
    parser.add_argument("--hide_tqdm", default=False, type=bool)
    parser.add_argument("--wandb_on", default=0, type=int)
    parser.add_argument("--resume", action="store_true")


def predict_args(parser):
    parser.add_argument("--theta", default=0.5, type=float)
    parser.add_argument("--eval_batch_size", default=20, type=int)
    parser.add_argument("--test_batch_size", default=20, type=int)
