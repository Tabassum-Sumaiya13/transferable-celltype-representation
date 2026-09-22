"""
STAGE 2 - one token per marker, so that ANY panel fits.

The problem this replaces. The old pipeline used a fixed 19-marker feature table. Adding markers
to it HURT the marker-poor cohorts (ferguson -0.085 measured), because a cohort that does not
measure a marker had to be handed a zero, and a zero is a real value - "this cell is negative" -
not "nobody looked". Here a cell is a SET of marker tokens instead of a fixed-width row, so a
cohort with 17 markers presents 17 tokens and a cohort with 57 presents 57. Nothing is invented.

    token = value_embedding(u_coh) + identity_embedding(triple)

The value MLP is SHARED across all markers; only the identity embedding is marker-specific. That
split is what lets a panel the model has never seen still be encoded - the value pathway does not
need to have met the marker before. It is not a new idea here: models/marker_encoder.py already built
it this way for Gate 1 and its docstring says outright that this IS the Stage 2 design.

A MASKED token keeps its identity: `mask_emb + ident_emb`, never a bare mask. The model has to
know WHICH marker is hidden or the question is unanswerable. Stage 1 already proved this
necessary - MaskedMarkerProbe passes `query = self.enc.ident(mask_idx)` for exactly this reason.

THE ONE ARCHITECTURAL FORK - what to do with markers a cohort does not measure.

The plan mandates a learned `[ABSENT]` token: "absent markers get a learned [ABSENT] token, never
a zero." There is a strong argument against it. The set of markers a cohort measures is a
near-perfect COHORT FINGERPRINT: the union panel is 99 triples, per-cohort panels run 17-57, and
43 of the 99 are measured by exactly one cohort. Feeding [ABSENT] tokens therefore hands the model
panel identity = cohort identity, which is precisely what Stage 3's adversary then has to erase.
Absence carries no biology either - it is a property of the panel, not of the cell. A Sorin cell
would arrive as 17 measured and 82 absent tokens, 83% of the compute spent on saying "not looked
at". The counter-consideration is real too: the identity embeddings of the PRESENT markers already
leak the panel, and that cannot be avoided, because the model must use the markers it has. The
difference is that absent slots add ONLY panel information.

This is settled by measurement, not by opinion, so both arms are built from one module:

    Arm A (set)      slots = only the cohort's measured markers.  `present=None`.
    Arm B (absent)   slots = all 99;  unmeasured get `absent_emb + ident_emb` and attend normally.

Gate 2 check 4 races them on reconstruction R2 and on a cohort probe over the pooled embedding.
Declared in advance in panel/gate2_expect.csv: ship the better R2; if tied, ship Arm A.

Batching note. Cells are fed one cohort at a time (round-robin), so every cell in a batch shares
one panel and one slot layout. That is why `idx` and `present` are per-BATCH vectors rather than
per-cell: there is no ragged padding to carry. `key_padding_mask` is still accepted so a later
stage can mix cohorts in one batch if it ever needs to.
"""
import torch
import torch.nn as nn


class Block(nn.Module):
    """One pre-norm self-attention block.

    Pre-norm (normalise, then attend, then add) rather than post-norm: it trains without a warmup
    schedule, which matters here because the run budget is a CPU box and every re-run costs the
    whole gate.

    LayerNorm is per-token, so nothing in this block reads batch statistics. That is deliberate -
    the pipeline must be able to score a cohort whose batch composition is nothing like training.
    """

    def __init__(self, d, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x, key_padding_mask=None):
        h = self.n1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a
        return x + self.ff(self.n2(x))


class TokenModel(nn.Module):
    """Marker tokens -> set transformer -> a predicted value for every slot.

    Args
      n_vocab   size of the token vocabulary (99 triples on the current roster)
      d         model width
      blocks    number of attention blocks
      heads     attention heads per block

    forward(u, idx, hide, present=None, key_padding_mask=None)
      u        (B, M) float   the cohort-level ECDF value per slot, in [0, 1]
      idx      (M,)   long    global vocabulary index of each slot
      hide     (B, M) bool    slots whose value is hidden and must be predicted
      present  (M,)   bool    slots this cohort measures. None means "all of them" (Arm A)
      returns  pred (B, M) predicted value for every slot, z (B, d) pooled cell embedding

    `u` at a hidden or absent slot is never read - it is overwritten before the value MLP output
    is used - so the caller may pass anything there. Zeros are passed for clarity.
    """

    def __init__(self, n_vocab, d=64, blocks=2, heads=4, hidden=64):
        super().__init__()
        self.value = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.ident = nn.Embedding(n_vocab, d)
        self.mask_emb = nn.Parameter(torch.zeros(d))
        self.absent_emb = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(blocks)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)
        self.d, self.n_vocab = d, n_vocab

    def forward(self, u, idx, hide, present=None, key_padding_mask=None):
        v = self.value(u.unsqueeze(-1))                       # (B, M, d)

        # a hidden slot loses its value but KEEPS its identity - see the module docstring
        v = torch.where(hide.unsqueeze(-1), self.mask_emb.expand_as(v), v)

        # Arm B only. Arm A never builds an absent slot, so absent_emb stays unused and its
        # gradient stays zero - the two arms are the same code path with a different slot list.
        if present is not None:
            v = torch.where(present.view(1, -1, 1), v, self.absent_emb.expand_as(v))

        x = v + self.ident(idx).unsqueeze(0)
        for b in self.blocks:
            x = b(x, key_padding_mask=key_padding_mask)
        x = self.norm(x)

        # z is the arm's OWN representation of the cell: Arm A pools its measured slots, Arm B
        # pools all 99 including the absent ones. Pooling them differently would rig check 4 -
        # the whole question is whether carrying absent slots changes what the cell looks like.
        return self.head(x).squeeze(-1), x.mean(1)
