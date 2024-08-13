import torchtext; torchtext.disable_torchtext_deprecation_warning()

import spacy
import torch
import torchtext.datasets as datasets

from model import Transformer
from dataset import Multi30k, Batch
from torch.utils.data import DataLoader
from train_eval_utils import greedy_decode
from torchtext.vocab import build_vocab_from_iterator

def yield_tokens(iter, tokenizer, index):
    for text in iter:
        yield [token.text for token in tokenizer.tokenizer(text[index])]
spacy_de = spacy.load("de_core_news_sm")
spacy_en = spacy.load("en_core_web_sm")
train, val, test = datasets.Multi30k(language_pair=("de", "en"))
src_vocab = build_vocab_from_iterator(yield_tokens(train+val+test, spacy_de, 0), specials=["<s>", "</s>", "<blank>", "<unk>"], min_freq=2)
tgt_vocab = build_vocab_from_iterator(yield_tokens(train+val+test, spacy_en, 1), specials=["<s>", "</s>", "<blank>", "<unk>"], min_freq=2)
src_vocab.set_default_index(src_vocab["<unk>"])
tgt_vocab.set_default_index(tgt_vocab["<unk>"])

test_data = Multi30k(dataset="test", tokenizer_de=spacy_de, tokenizer_en=spacy_en, vocab_de=src_vocab, vocab_en=tgt_vocab, language_pair=('de', 'en'))
test_dataloader = DataLoader(test_data, batch_size=4)

model = Transformer(len(src_vocab), len(tgt_vocab), N=6)
model.to("cuda")
model.eval()
model.load_state_dict(torch.load("saves/my_multi30k_model_140.pt"))

pad_idx = 2
eos_string="</s>"

from train_eval_utils import val_one_epoch
from loss import SimpleLossCompute, LabelSmoothing
pad_idx = tgt_vocab["<blank>"]
criterion = LabelSmoothing(size=len(tgt_vocab), padding_idx=pad_idx, smoothing=0.1)
criterion.to("cuda")
valid_data = Multi30k(dataset="test", tokenizer_de=spacy_de, tokenizer_en=spacy_en, vocab_de=src_vocab, vocab_en=tgt_vocab, language_pair=('de', 'en'))
valid_dataloader = DataLoader(test_data, batch_size=4)
loss, bleu_score = val_one_epoch(valid_dataloader, model, src_vocab, tgt_vocab, SimpleLossCompute(model.generator, criterion),)
print("Loss: ", loss.data)
print("BLEU: ", bleu_score)

for idx, batch in (enumerate((Batch(b[0], b[1], pad_idx) for b in test_dataloader))):
    out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)

    model_out = greedy_decode(model, batch.src, batch.src_mask, 72, 0)
    model_txt = []
    tgt_txt   = []
    for i in model_out:
        model_txt.append(" ".join([tgt_vocab.get_itos()[x] for x in i if x != pad_idx]).split(eos_string, 1)[0] + eos_string)
    for i in batch.tgt:
        tgt_txt.append(" ".join([tgt_vocab.get_itos()[x] for x in i if x != pad_idx]).split(eos_string, 1)[0] + eos_string)

    print("Test {}".format(idx))
    print(tgt_txt)
    print(model_txt)
    if idx == 4:
        break