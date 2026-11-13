"""MTP speculative decoding on the device (#16): glm_vx/speculative.py's generate() over a ResidentGLM.

The MTP layer drafts with ResidentGLM.mtp_step and the model verifies the next token and the drafts
as one forward_many step, so each weight is read once for all of them. The rule is the CPU one,
with its functions: greedy (the model's greedy tokens whatever the drafts) or speculative sampling
(the model's distribution). Rolling back is moving the positions back: a cache row past a
position is never read, and the next step at that position writes it again. So the drafts' rows
are simply overwritten, and the MTP layer runs again on the accepted positions with the model's own
hidden states, as on the CPU.

  resident = ResidentGLM(model, max_tokens=..., mtp=True, fp8_kernels='x4', device_experts=True)
  tokens, stats = generate(resident, prompt, 64, draft_tokens=3)

draft_tokens is at most 3: forward_many verifies up to four tokens. With graph=True (and
device_experts) forward, forward_many and mtp_step each replay CUDA graphs, with the same tokens.
"""
import numpy as np

from glm_vx.speculative import Stats, _choose, accept_sampled, probabilities, sample

from .resident import MANY_TOKENS


def generate(resident, prompt, max_new_tokens, *, draft_tokens=3, eos=(), temperature=0.0, rng=None,
             choose_draft=None, index_share=None):
    """(tokens, stats): greedy at temperature 0, else sampled from `rng`; resident.reset() first.

    choose_draft(logits, position, step), greedy only, replaces the drafts (tests); index_share
    defaults to the config's index_share_for_mtp_iteration."""
    prompt = [int(t) for t in prompt]
    if not prompt or max_new_tokens < 1 or not 1 <= draft_tokens < MANY_TOKENS:
        raise ValueError(f'need a prompt, max_new_tokens >= 1 and 1 <= draft_tokens <= {MANY_TOKENS - 1}')
    sampled = temperature > 0
    if sampled and (rng is None or choose_draft is not None):
        raise ValueError('sampling needs an rng and draws its own drafts')
    if not resident.mtp:
        raise ValueError('generate needs ResidentGLM(..., mtp=True)')
    if index_share is None:
        index_share = bool(resident.config.get('index_share_for_mtp_iteration', False))
    choose_draft = choose_draft or (lambda logits, position, step: _choose(logits))
    stop = {int(t) for t in eos}
    stats = Stats()
    resident.reset()

    hiddens = []
    for token in prompt:
        logits, hidden = resident.forward_hidden(token)
        hiddens.append(hidden)
    for p in range(len(prompt) - 1):  # the MTP rows of the prompt: position p from (token p + 1, h_p)
        resident.mtp_step(prompt[p + 1], hiddens[p], p)
    out = [sample(probabilities(logits, temperature), rng) if sampled else _choose(logits)]
    hidden, position = hiddens[-1], len(prompt) - 1
    while len(out) < max_new_tokens and out[-1] not in stop:
        k = min(draft_tokens, max_new_tokens - len(out))
        drafts, draft_probs, token, h = [], [], out[-1], hidden
        for j in range(k):
            draft_logits, h = resident.mtp_step(token, h, position + j, reuse_selection=index_share and j > 0)
            if sampled:
                draft_probs.append(probabilities(draft_logits, temperature))
                token = sample(draft_probs[-1], rng)
            else:
                token = choose_draft(draft_logits, position + j + 2, j)
            drafts.append(token)
        start = resident.position
        all_logits, verified_hidden = resident.forward_many([out[-1]] + drafts)
        verified = [probabilities(l, temperature) if sampled else _choose(l) for l in all_logits]
        if sampled:
            accepted, last = accept_sampled(verified, draft_probs, drafts, rng)
        else:
            accepted = 0
            while accepted < k and drafts[accepted] == verified[accepted]:
                accepted += 1
            last = verified[accepted]
        stats.rounds += 1
        stats.drafted += k
        stats.accepted += accepted
        stats.accepted_per_round.append(accepted)
        resident.position = start + accepted + 1
        for i in range(accepted):  # the accepted drafts' MTP rows, from the model's hidden states
            resident.mtp_step(drafts[i], verified_hidden[i], position + 1 + i)
        for token in drafts[:accepted] + [last]:
            out.append(token)
            if token in stop or len(out) == max_new_tokens:
                break
        hidden, position = verified_hidden[accepted], start + accepted
    resident.position = min(resident.position, len(prompt) + len(out) - 1)  # as plain decoding leaves it
    return out, stats


def greedy(resident, prompt, max_new_tokens, *, eos=()):
    """Plain greedy decoding on the resident model, for comparison; resident.reset() first."""
    resident.reset()
    for token in prompt:
        logits = resident.forward(int(token))
    out = [_choose(logits)]
    stop = {int(t) for t in eos}
    while len(out) < max_new_tokens and out[-1] not in stop:
        out.append(_choose(resident.forward(out[-1])))
    return out
