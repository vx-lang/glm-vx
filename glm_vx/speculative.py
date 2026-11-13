"""MTP speculative decoding (#16): the checkpoint's MTP layer drafts, the model verifies, greedy.

GLM-5.3 has one MTP layer (num_nextn_predict_layers: 1), stored as model.layers.78 after the 78
decoder layers. As vLLM's Glm4MoeMultiTokenPredictorLayer wires it, it predicts the token after
next from the next token t and the model's hidden state h at the current position:

    e = enorm(embed(t))              (embed(t) is zero at position 0)
    x = eh_proj([e, hnorm(h)])       (hidden x 2*hidden; the embedding half first)
    x = decoder layer 78 (x)         (MLA with its own cache, a full indexer, MoE)
    g = shared_head.norm(x)          the draft's hidden state
    logits = lm_head(g)              (the model's own head and embeddings)

Drafting k tokens chains it: the next step takes the draft token and g in place of t and h, at the
next position. The MTP layer's cache at position p holds what the layer computed from
(token p+1, h_p), so after a round of drafts only the first entry is right: the later ones came
from drafts and draft hidden states. Verification runs the model on the next token and the k
drafts; the drafts are accepted up to the first one the model's greedy choice disagrees with, and
the model's choice after the last accepted token is emitted too. Then both caches roll back to the
accepted tokens, and the MTP layer runs again on the accepted positions with the model's own hidden
states, as vLLM's proposer does with the target's hidden states. So the output is the model's
greedy output whatever the drafts are, and the caches end as plain decoding leaves them.

With a temperature T > 0 the drafts are sampled from the MTP layer's softmax(logits / T) (q), and
accepted by speculative sampling (Leviathan et al. 2023, Chen et al. 2023): draft d with probability
min(1, p(d) / q(d)), where p is the model's softmax(logits / T); the first rejected one is replaced
by a sample of max(0, p - q), normalized; after k accepted drafts the model's token is sampled from
p. Each emitted token then has the model's own distribution, given the tokens before it: the
distribution glm_vx.scheduler samples, whatever the drafts are. T = 0 is greedy, as above.

Not here yet:
  - index_share_for_mtp_iteration: each draft step runs the MTP layer's indexer, where the
    official recipe reuses the first step's selection;
  - one batched verification step: the model runs the k + 1 tokens one at a time, which computes
    the same values. A batched step is what makes drafts pay on a GPU (gpu/resident.py).
"""
from dataclasses import dataclass, field

import numpy as np

from .model import LayerCache, RequestCache


class MTPDrafter:
    """The model's first MTP layer, model.layers.{num_hidden_layers}, with its own cache."""

    def __init__(self, model):
        if model.mtp_layers < 1:
            raise ValueError('the config has no MTP layer (num_nextn_predict_layers)')
        self.model = model
        self.layer = model.n_layers
        self.prefix = f'model.layers.{self.layer}'
        self.state = LayerCache()

    @property
    def position(self):
        return len(self.state.latents)

    def step(self, token, hidden, position):
        """The MTP layer at `position` from (token at position + 1, hidden at position): its logits
        for the token at position + 2 and its hidden state."""
        m = self.model
        if position != self.position:
            raise ValueError(f'MTP cache holds {self.position} positions, the step is at {position}')
        embedding = m._embedding(int(token))
        if position == 0:
            embedding = np.zeros_like(embedding)
        x = m._linear(self.prefix + '.eh_proj', np.concatenate((
            m._norm(self.prefix + '.enorm', embedding, m.eps),
            m._norm(self.prefix + '.hnorm', np.asarray(hidden, np.float32), m.eps))))
        try:
            x, _ = m.forward_layer(self.layer, x, position, self.state)
        except Exception:
            self.truncate(position)
            raise
        hidden = m._norm(self.prefix + '.shared_head.norm', x, m.eps)
        return m._linear(m.head_name, hidden), hidden

    def truncate(self, position):
        del self.state.latents[position:]
        del self.state.rope_keys[position:]
        del self.state.index_keys[position:]


@dataclass
class Stats:
    rounds: int = 0
    drafted: int = 0
    accepted: int = 0
    accepted_per_round: list = field(default_factory=list)

    @property
    def accept_length(self):
        """Tokens emitted per verification step: accepted drafts + the model's own token."""
        return (self.accepted + self.rounds) / self.rounds if self.rounds else 0.0


def _choose(logits):
    return int(np.argmax(logits))


def probabilities(logits, temperature):
    """softmax(logits / temperature) in float64, as glm_vx.scheduler samples."""
    scaled = (np.asarray(logits, np.float64) - float(np.max(logits))) / temperature
    probs = np.exp(scaled)
    return probs / probs.sum()


def sample(probs, rng):
    return int(rng.choice(len(probs), p=probs))


def accept_sampled(target, draft, drafts, rng):
    """Speculative sampling of one round: (accepted, token).

    target: the model's k + 1 distributions, at the next token and after each draft; draft: the
    k distributions the drafts were sampled from; drafts: the k tokens. The first `accepted` drafts
    are kept and `token` follows them."""
    for j, (d, p, q) in enumerate(zip(drafts, target, draft)):
        if rng.random() * q[d] < p[d]:  # u < min(1, p(d) / q(d)), with q(d) > 0 since d ~ q
            continue
        residual = np.maximum(p - q, 0.0)
        return j, sample(residual / residual.sum(), rng)
    return len(drafts), sample(target[len(drafts)], rng)


def generate(model, prompt, max_new_tokens, *, draft_tokens=3, drafter=None, eos=(), cache=None, choose_draft=None,
             temperature=0.0, rng=None):
    """Generation with MTP drafts: greedy (temperature 0), the same tokens as plain greedy decoding;
    or sampled at `temperature` from `rng` (numpy Generator), each token from the model's own
    distribution given the tokens before it.

    Returns (tokens, cache, drafter, stats). `cache` (the model's, new if None) ends holding the
    prompt and every emitted token but the last, as plain decoding leaves it. `choose_draft(logits,
    position, step)` picks the draft token for `position`, the round's step-th draft (default
    argmax); any choice gives the same output. Sampling draws the drafts from the MTP layer's own
    distribution, which speculative sampling needs, so it takes no `choose_draft`.
    """
    prompt = [int(t) for t in prompt]
    if not prompt or max_new_tokens < 1 or draft_tokens < 1:
        raise ValueError('need a prompt, max_new_tokens >= 1 and draft_tokens >= 1')
    if cache is None:
        cache = model.new_cache()
    if not isinstance(cache, RequestCache) or cache.position != 0:
        raise ValueError('generate starts from an empty cache')
    drafter = drafter or MTPDrafter(model)
    if drafter.position != 0:
        raise ValueError('the drafter must start empty')
    sampled = temperature > 0
    if sampled and (rng is None or choose_draft is not None):
        raise ValueError('sampling needs an rng and draws its own drafts')
    if not np.isfinite(temperature) or temperature < 0:
        raise ValueError('temperature must be finite and nonnegative')
    choose_draft = choose_draft or (lambda logits, position, step: _choose(logits))
    stop = {int(t) for t in eos}
    stats = Stats()

    def selections():
        return [None if s.selected_indices is None else s.selected_indices.copy() for s in cache.layers]

    hiddens = []
    for token in prompt:
        logits, hidden = model.forward_hidden(token, cache)
        hiddens.append(hidden)
    # The MTP cache for the prompt: position p from (token p + 1, h_p).
    for p in range(len(prompt) - 1):
        drafter.step(prompt[p + 1], hiddens[p], p)
    out = [sample(probabilities(logits, temperature), rng) if sampled else _choose(logits)]
    hidden, position = hiddens[-1], len(prompt) - 1  # the next draft step: (out[-1], h at position)
    while len(out) < max_new_tokens and out[-1] not in stop:
        k = min(draft_tokens, max_new_tokens - len(out))
        drafts, draft_probs, token, h = [], [], out[-1], hidden
        for j in range(k):
            draft_logits, h = drafter.step(token, h, position + j)
            if sampled:
                draft_probs.append(probabilities(draft_logits, temperature))
                token = sample(draft_probs[-1], rng)
            else:
                token = choose_draft(draft_logits, position + j + 2, j)
            drafts.append(token)
        # Verify: the model on out[-1] and the drafts, at positions position + 1 ...
        start = cache.position
        verified, verified_hidden, verified_selections = [], [], []
        for token in [out[-1]] + drafts:
            logits, h = model.forward_hidden(token, cache)
            verified.append(probabilities(logits, temperature) if sampled else _choose(logits))
            verified_hidden.append(h)
            verified_selections.append(selections())
        if sampled:
            accepted, last = accept_sampled(verified, draft_probs, drafts, rng)
        else:
            accepted = 0
            while accepted < k and drafts[accepted] == verified[accepted]:
                accepted += 1
            last = verified[accepted]
        emitted = drafts[:accepted] + [last]
        stats.rounds += 1
        stats.drafted += k
        stats.accepted += accepted
        stats.accepted_per_round.append(accepted)
        # Keep the model's cache up to the last accepted draft, and the MTP cache up to the first
        # draft step (built from out[-1] and the model's hidden), then rebuild the MTP entries of
        # the accepted drafts from the model's hidden states.
        model.truncate(cache, start + accepted + 1, verified_selections[accepted])
        drafter.truncate(position + 1)
        for i in range(accepted):
            drafter.step(drafts[i], verified_hidden[i], position + 1 + i)
        for token in emitted:
            out.append(token)
            if token in stop or len(out) == max_new_tokens:
                break
        hidden, position = verified_hidden[accepted], start + accepted
    # As plain decoding: every emitted token but the last has run. A stop inside the accepted
    # drafts leaves later accepted tokens in the cache; roll those back.
    end = len(prompt) + len(out) - 1
    if cache.position > end:
        keep = end - start - 1
        model.truncate(cache, end, verified_selections[keep])
        drafter.truncate(end - 1)
    return out, cache, drafter, stats


def greedy(model, prompt, max_new_tokens, *, eos=(), cache=None):
    """Plain greedy decoding, for comparison: (tokens, cache)."""
    cache = cache or model.new_cache()
    for token in prompt:
        logits = model.forward(int(token), cache)
    out = [_choose(logits)]
    stop = {int(t) for t in eos}
    while len(out) < max_new_tokens and out[-1] not in stop:
        out.append(_choose(model.forward(out[-1], cache)))
    return out, cache
