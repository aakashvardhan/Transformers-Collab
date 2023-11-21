from pathlib import Path


# ===================== BERT Config ===================== #

def bert_config():
    return {
        "batch_size": 1024,
        "seq_len": 20,
        "embed_size": 128,
        "inner_ff_size": bert_config().get("embed_size") * 4,
        "n_heads": 8,
        "n_code": 8,
        "n_vocab": 40000,
        "dropout": 0.1,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "betas": (.9, .999),
        "train_file": "dataset/training.txt",
        "vocab_file": "dataset/vocab.txt",
    }
    