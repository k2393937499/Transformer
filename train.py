import torchtext; torchtext.disable_torchtext_deprecation_warning()

import os
import spacy
import torch
import torchtext.datasets as datasets

from dataset import Multi30k
from model import Transformer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from loss import LabelSmoothing, SimpleLossCompute
from torchtext.vocab import build_vocab_from_iterator
from train_eval_utils import train_one_epoch, val_one_epoch

def yield_tokens(iter, tokenizer, index):
    for text in iter:
        yield [token.text for token in tokenizer.tokenizer(text[index])]
try:
    spacy_de = spacy.load("de_core_news_sm")
except IOError:
    os.system("python -m spacy download de_core_news_sm")
    spacy_de = spacy.load("de_core_news_sm")

try:
    spacy_en = spacy.load("en_core_web_sm")
except IOError:
    os.system("python -m spacy download en_core_web_sm")
    spacy_en = spacy.load("en_core_web_sm")
train, val, test = datasets.Multi30k(language_pair=("de", "en"))
src_vocab = build_vocab_from_iterator(yield_tokens(train+val+test, spacy_de, 0), specials=["<s>", "</s>", "<blank>", "<unk>"], min_freq=2)
tgt_vocab = build_vocab_from_iterator(yield_tokens(train+val+test, spacy_en, 1), specials=["<s>", "</s>", "<blank>", "<unk>"], min_freq=2)
src_vocab.set_default_index(src_vocab["<unk>"])
tgt_vocab.set_default_index(tgt_vocab["<unk>"])

train_data = Multi30k(dataset="train", tokenizer_de=spacy_de, tokenizer_en=spacy_en, vocab_de=src_vocab, vocab_en=tgt_vocab, language_pair=('de', 'en'))
valid_data = Multi30k(dataset="val", tokenizer_de=spacy_de, tokenizer_en=spacy_en, vocab_de=src_vocab, vocab_en=tgt_vocab, language_pair=('de', 'en'))
train_dataloader = DataLoader(train_data, batch_size=16)
valid_dataloader = DataLoader(valid_data, batch_size=16)

config = {
    "num_epochs": 10,
    "accum_iter": 10,
    "base_lr": 1.0,
    "max_padding": 72,
    "warmup": 3000,
    "file_prefix": "multi30k_model_",
}

model = Transformer(len(src_vocab), len(tgt_vocab), N=6)
model.to("cuda")

pad_idx = tgt_vocab["<blank>"]
criterion = LabelSmoothing(size=len(tgt_vocab), padding_idx=pad_idx, smoothing=0.1)
criterion.to("cuda")

def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))
optimizer = torch.optim.Adam(model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9)
lr_scheduler = LambdaLR(
    optimizer=optimizer,
    lr_lambda=lambda step: rate(step, model_size=512, factor=1, warmup=config["warmup"]),
)

for epoch in range(config["num_epochs"]):
    model.train()
    print(f"Epoch {epoch} Training ====", flush=True)
    sloss = train_one_epoch(
        train_dataloader,
        model,
        SimpleLossCompute(model.generator, criterion),
        optimizer,
        lr_scheduler,
        accum_iter=config["accum_iter"],
    )

    loss, bleu_score = val_one_epoch(valid_dataloader, model, src_vocab, tgt_vocab, SimpleLossCompute(model.generator, criterion),)

    with open('log.txt', 'a', encoding='utf-8') as file:
        text_to_append = "epochs:{}, loss:{}, bleu_score:{}\n".format(epoch, loss.data, bleu_score)
        file.write(text_to_append)

    if epoch % 10 == 0:
        file_path = "saves/%s%.2d.pt" % (config["file_prefix"], epoch)
        torch.save(model.state_dict(), file_path)
    torch.cuda.empty_cache()

file_path = "saves/%sfinal.pt" % config["file_prefix"]
torch.save(model.state_dict(), file_path)