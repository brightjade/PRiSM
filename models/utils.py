from transformers import AutoConfig, AutoTokenizer, AutoModel
from dataset import load_and_cache_relations
from .prism import PRISM


def load_config(args):
    config = AutoConfig.from_pretrained(
                args.config_name if args.config_name else args.model_name_or_path,
                num_labels=args.num_labels,
                cache_dir=args.cache_dir if args.cache_dir else None)
    return config


def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
                    args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                    cache_dir=args.cache_dir if args.cache_dir else None)
    return tokenizer


def load_model(args, config, tokenizer):
    relation_features = load_and_cache_relations(args, config, tokenizer)

    encoder1 = AutoModel.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    cache_dir=args.cache_dir if args.cache_dir else None)
    
    if not args.share_params:
        encoder2 = AutoModel.from_pretrained(
                        args.model_name_or_path,
                        from_tf=bool(".ckpt" in args.model_name_or_path),
                        cache_dir=args.cache_dir if args.cache_dir else None)
    else:
        encoder2 = None
    
    model = PRISM(args, config, relation_features, encoder1, encoder2)

    return model.to(args.device)
