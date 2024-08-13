import torchtext; torchtext.disable_torchtext_deprecation_warning()

import spacy
import torch
import torchtext.datasets as datasets
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset

class Multi30k():
    def __init__(self, dataset, tokenizer_de, tokenizer_en, vocab_de, vocab_en, language_pair, max_padding = 128, pad_id = 2):
        assert dataset in ['train', 'val', 'test'], 'dataset must be in [train, val, test]'
        assert language_pair in [('de', 'en'), ('en', 'de')], 'language pair must be (\'en\', \'de\') or (\'de\', \'en\') '
        self.tokenizer_de = tokenizer_de
        self.tokenizer_en = tokenizer_en
        self.vocab_de = vocab_de
        self.vocab_en = vocab_en
        self.language_pair = language_pair
        self.max_padding = max_padding
        self.pad_id = pad_id
        
        if dataset == 'train':
            self.orign_dataset, _, _ = datasets.Multi30k(language_pair=language_pair)
        elif dataset == 'val':
            _, self.orign_dataset, _ = datasets.Multi30k(language_pair=language_pair)
        else:
            _, _, self.orign_dataset = datasets.Multi30k(language_pair=language_pair)
        self.iter_map = to_map_style_dataset(self.orign_dataset)

    def __getitem__(self, idx):
        bs_id = torch.tensor([0], device=0)  # <s> token id
        eos_id = torch.tensor([1], device=0)  # </s> token id
        if self.language_pair == ('de', 'en'):
            origin_src = self.iter_map[idx][0]
            origin_tgt = self.iter_map[idx][1]
            src = torch.cat([bs_id, torch.tensor(self.vocab_de([token.text for token in self.tokenizer_de.tokenizer(origin_src)]), dtype=torch.int64, device=0), eos_id], dim=0)
            tgt = torch.cat([bs_id, torch.tensor(self.vocab_en([token.text for token in self.tokenizer_de.tokenizer(origin_tgt)]), dtype=torch.int64, device=0), eos_id], dim=0)
            src = pad(src, (0, self.max_padding - len(src)), value=self.pad_id)
            tgt = pad(tgt, (0, self.max_padding - len(tgt)), value=self.pad_id)
        else:
            origin_src = self.iter_map[idx][1]
            origin_tgt = self.iter_map[idx][0]
            src = torch.cat([bs_id, torch.tensor(self.vocab_de([token.text for token in self.tokenizer_de.tokenizer(origin_src)]), dtype=torch.int64, device=0), eos_id], dim=0)
            tgt = torch.cat([bs_id, torch.tensor(self.vocab_en([token.text for token in self.tokenizer_de.tokenizer(origin_tgt)]), dtype=torch.int64, device=0), eos_id], dim=0)
            src = pad(src, (0, self.max_padding - len(src)), value=self.pad_id)
            tgt = pad(tgt, (0, self.max_padding - len(tgt)), value=self.pad_id)
        return src, tgt

    def __len__(self):
        return len(self.iter_map)

class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        attn_shape = (1, tgt.size(-1), tgt.size(-1))
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
        subsequent_mask = (subsequent_mask == 0).type_as(tgt_mask.data)
        tgt_mask = tgt_mask & subsequent_mask
        return tgt_mask
    


if __name__ == "__main__":
    def yield_tokens(iter, tokenizer, index):
        for text in iter:
            yield [token.text for token in tokenizer.tokenizer(text[index])]
    spacy_de = spacy.load("de_core_news_sm")
    spacy_en = spacy.load("en_core_web_sm")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    src_vocab = build_vocab_from_iterator(yield_tokens(train+val+test, spacy_de, 0), specials=["<s>", "</s>", "<blank>", "<unk>"])
    tgt_vocab = build_vocab_from_iterator(yield_tokens(train+val+test, spacy_en, 1), specials=["<s>", "</s>", "<blank>", "<unk>"])
    src_vocab.set_default_index(src_vocab["<unk>"])
    tgt_vocab.set_default_index(tgt_vocab["<unk>"])

    origin_data = Multi30k(dataset="val", tokenizer_de=spacy_de, tokenizer_en=spacy_en, vocab_de=src_vocab, vocab_en=tgt_vocab, language_pair=('de', 'en'))
    data_iter = DataLoader(origin_data, batch_size=1)
    data = (Batch(b[0], b[1], pad=2) for b in data_iter)

    for i in data:
        print(i)
        print(i.src)
        print(i.src_mask)
        print(i.tgt)
        print(i.tgt_mask)