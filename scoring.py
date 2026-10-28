# scoring.py — validated right-padded teacher-forced scorer returning three views of the same NLLs.
import numpy as np
import torch
import torch.nn.functional as F
import phase3_config as cfg


@torch.no_grad()
def batched_nll_stats(model, cid, sent_list, n_carrier, head_k=16):
    """Returns (mean, total, head) per sentence, after [carrier | sentence].
    mean  = mean nats/token over all tokens (legacy metric)
    total = summed nats (length-sensitive, tokenizer-robust within a family)
    head  = mean nats over the first `head_k` tokens (where a single injected vector acts;
            removes the 1/length dilution that confounds EN vs ZH)."""
    B, Lm = len(sent_list), max(len(s) for s in sent_list)
    ids = torch.full((B, n_carrier + Lm), cid, dtype=torch.long)
    att = torch.zeros_like(ids); att[:, :n_carrier] = 1
    tgt = torch.full((B, Lm), -100, dtype=torch.long)
    for b, s in enumerate(sent_list):
        t = torch.tensor(s)
        ids[b, n_carrier:n_carrier + len(s)] = t
        att[b, n_carrier:n_carrier + len(s)] = 1
        tgt[b, :len(s)] = t
    ids, att, tgt = ids.to(cfg.DEVICE), att.to(cfg.DEVICE), tgt.to(cfg.DEVICE)
    # logits predicting sentence token 0 come from the last carrier position (index n_carrier-1)
    logits = model(input_ids=ids, attention_mask=att).logits[:, n_carrier - 1:-1, :].float()
    nll = F.cross_entropy(logits.transpose(1, 2), tgt, ignore_index=-100, reduction="none")  # (B, Lm); 0 at pads
    n = (tgt != -100).sum(1)
    k = torch.clamp(n, max=head_k)
    pos = torch.arange(Lm, device=nll.device)[None, :]
    mean = nll.sum(1) / n
    total = nll.sum(1)
    head = (nll * (pos < k[:, None])).sum(1) / k
    return mean.cpu().numpy(), total.cpu().numpy(), head.cpu().numpy()