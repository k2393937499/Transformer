import time
import torch
import evaluate
from dataset import Batch

def train_one_epoch(data_iter, model, loss_compute, optimizer, scheduler, accum_iter=1, pad_idx = 2):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate((Batch(b[0], b[1], pad_idx) for b in data_iter)):
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        # loss_node = loss_node / accum_iter
        loss_node.backward()
        if i % accum_iter == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            n_accum += 1
        scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 100 == 0:
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                ("Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f " + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e")
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(src.size(0), 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.view(-1, 1)
        ys = torch.cat([ys, next_word], dim=1)
    return ys

bleu = evaluate.load("bleu")
@torch.no_grad()
def val_one_epoch(
    valid_dataloader,
    model,
    vocab_src,
    vocab_tgt,
    loss_compute,
    pad_idx=2,
    eos_string="</s>",
):
    model.eval()
    total_loss = 0
    total_tokens = 0
    prediction = []
    references = []
    for idx, batch in enumerate((Batch(b[0], b[1], pad_idx) for b in valid_dataloader)):
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens

        model_out = greedy_decode(model, batch.src, batch.src_mask, 72, 0)
        for i in model_out:
            prediction.append(" ".join([vocab_tgt.get_itos()[x] for x in i if x != pad_idx]).split(eos_string, 1)[0] + eos_string)
        for i in batch.tgt:
            references.append([" ".join([vocab_tgt.get_itos()[x] for x in i if x != pad_idx]).split(eos_string, 1)[0] + eos_string])

    bleu_score = bleu.compute(predictions=prediction, references=references)['bleu']

    return total_loss / total_tokens, bleu_score


if __name__ == "__main__":
    predictions = ["A man wearing a red shirt is sitting down with a small boat a b ."]
    references = [
        ["A man wearing a red life jacket is sitting in a small boat ."]
    ]
    bleu_h = evaluate.load("bleu")
    results = bleu_h.compute(predictions=predictions, references=references)
    print(results['bleu'])