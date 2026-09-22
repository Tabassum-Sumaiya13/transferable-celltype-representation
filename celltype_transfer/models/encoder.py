"""
STAGE 3 - cell encoder + domain-adversarial training.

Purpose. Turn a cell's marker tokens into `z_cell`, a vector that carries BIOLOGY but not BATCH.
The old pipeline measured what happens when it fails: image-level features let the model recognise
which dataset it was looking at, and that cost -0.032 mean macro-F1, -0.125 on Keren.

The mechanism is a gradient reversal layer (GRL). It is the identity going forward and multiplies
the gradient by -lambda coming back, so the head on top of it learns to predict the domain while
the encoder underneath learns to make that prediction impossible.

WHY THE DOMAIN IS SLIDE, NOT COHORT (decision D-16 - do not re-litigate).
A 6-way cohort adversary is easy to drive to chance, and that is exactly the problem: cohort is
confounded with tissue on this roster (colorectal, head & neck, breast, lung, skin x2), so pushing
cohort accuracy to 1/6 means the embedding can no longer tell colon from lung - and colon and lung
tumour cells genuinely differ. Slide-to-slide variation INSIDE one cohort is near-pure batch
effect: same tissue, same machine, same disease. Erasing that is safe. Achievability is not the
goal; not destroying biology is.

The cohort head is therefore kept at a LOW weight and read as an INVERTED GUARD: if cohort
accuracy falls toward 1/6, that is a warning that tissue identity is being erased, not a success.

THE METRIC WAS REWRITTEN BECAUSE THE ORIGINAL WAS DEFECTIVE (D-15).
"The discriminator's accuracy must fall toward chance" is unusable on 1,137 slides - chance is
0.09%, no run reaches it, and no threshold was ever given, so the gate could have failed an
adversary that was working perfectly. Two replacements:

  retained_bits = log2(N_slides) - CE_bits          # 0 at chance, scale-free, comparable across
                                                     # lambda. Reported absolutely and as a
                                                     # fraction of the lambda=0 value.
  ... scored by a FRESH probe on HELD-OUT SLIDES. Adversarial training can HIDE information from
  the co-trained discriminator without removing it: the discriminator settles into a bad optimum
  and the encoder exploits it. So the honest measurement freezes the encoder, trains a NEW
  discriminator from scratch, and scores it on slides it has never seen. Reporting the co-trained
  and fresh numbers side by side makes that gap visible - it is invisible with either alone.

Two claims from an earlier review that do NOT hold, recorded so they are not raised again:
  - "class imbalance lets the discriminator cheat" - mean slide share is 0.09%, and even a slide
    3x the mean reaches ~0.4%. Guessing the largest slide buys almost nothing.
  - "a probe scoring high means it is fitting noise" - slide identity is genuinely present in the
    marker values. That is real signal, which is the whole reason it has to be removed.
"""
import torch
import torch.nn as nn

from .tokens import Block


class GradReverse(torch.autograd.Function):
    """Identity forwards, gradient scaled by -lambda backwards.

    Deliberately a no-op at lambda=0, so the lambda=0 arm of the sweep is EXACTLY the plain
    encoder with no adversary - not "an adversary configured to do nothing". That matters because
    lambda=0 is the fallback the gate ships if no lambda beats it, and it must be a clean baseline.
    """

    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def grad_reverse(x, lam):
    return GradReverse.apply(x, lam)


class CellEncoder(nn.Module):
    """Marker tokens -> z_cell.

    The token layer repeats Stage 2's design rather than importing its module, on purpose: Stage
    2's `TokenModel` is already trained and cached in work/ckpt/s2_*.pt, and refactoring it would
    invalidate every one of those checkpoints. The shapes are identical, so `load_stage2()` below
    can still warm-start from them.

    `d_tok` stays at Stage 2's 64 while `d_z` is the plan's 128. Widening the ATTENTION is what
    costs - it scales with the square of the token count - whereas widening only the pooled output
    is a single linear layer. So the expensive dimension is kept small and the cheap one is not.

    Arms A and B are the same call with a different slot list, exactly as in Stage 2:
        Arm A (set)     idx = the cohort's own triples,  present = None
        Arm B (absent)  idx = arange(n_vocab),           present = the cohort's measured mask
    """

    def __init__(self, n_vocab, d_tok=64, d_z=128, blocks=2, heads=4, hidden=64):
        super().__init__()
        self.value = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, d_tok))
        self.ident = nn.Embedding(n_vocab, d_tok)
        self.mask_emb = nn.Parameter(torch.zeros(d_tok))
        self.absent_emb = nn.Parameter(torch.zeros(d_tok))
        self.blocks = nn.ModuleList([Block(d_tok, heads) for _ in range(blocks)])
        self.norm = nn.LayerNorm(d_tok)
        self.proj = nn.Sequential(nn.Linear(d_tok, d_z), nn.GELU(), nn.Linear(d_z, d_z))
        self.d_tok, self.d_z, self.n_vocab = d_tok, d_z, n_vocab

    def forward(self, u, idx, present=None, hide=None, key_padding_mask=None,
                return_tokens=False):
        """`return_tokens` hands back the PER-TOKEN features as well as the pooled z_cell.

        Stage 6 needs them: its masked-marker auxiliary loss predicts a hidden marker's value from
        its own token, which the pooled vector has already averaged away. Additive and off by
        default, so Stage 3's checkpoints and numbers are untouched.
        """
        v = self.value(u.unsqueeze(-1))
        if hide is not None:                       # Stage 6 reuses the masked-marker objective
            v = torch.where(hide.unsqueeze(-1),    # as an auxiliary loss
                            self.mask_emb.expand_as(v), v)
        if present is not None:                    # Arm B: unmeasured slots get [ABSENT]
            v = torch.where(present.view(1, -1, 1), v, self.absent_emb.expand_as(v))
        x = v + self.ident(idx).unsqueeze(0)
        for b in self.blocks:
            x = b(x, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        z = self.proj(x.mean(1))
        return (z, x) if return_tokens else z

    def load_stage2(self, state):
        """Warm-start the token layer and attention blocks from a Stage 2 checkpoint.

        Stage 2's masked-marker task is a pretraining objective for exactly these weights, so this
        is free signal. `proj` has no Stage 2 counterpart and stays randomly initialised. Reported
        as a Gate 3 arm rather than assumed to help - a warm start can also lock in whatever the
        pretext task over-fitted.
        """
        keep = {k: v for k, v in state.items()
                if not k.startswith('head.') and k in self.state_dict()
                and self.state_dict()[k].shape == v.shape}
        missing = self.load_state_dict(keep, strict=False)
        return len(keep), missing


class Heads(nn.Module):
    """One task head plus two adversarial heads behind the gradient reversal layer.

    Cell type is a PLAIN LINEAR head with class-balanced sampling, not the prototype loss. That is
    deliberate scope control: Gate 3 asks one question - does the adversary help cross-cohort
    transfer - and the simplest head that answers it fairly is the one that does not introduce a
    second moving part. Prototypes, the collapse guard and Kendall sigma weighting are Stage 6.

    Descendant-tolerant cross-entropy is NOT used here either, and that is not an oversight:
    section 7 gap 2 records that Stage 1b's nesting layer missed all three cases declared for it,
    so the parent/child graph is not yet trustworthy enough to put inside a loss.

    The two adversarial heads are deliberately asymmetric in weight. Slide carries the batch
    effect and gets the full lambda; cohort is confounded with tissue and gets a fraction of it,
    and is read as a guard rather than an objective - see the module docstring.

    THE SLIDE HEAD'S CAPACITY IS NOT COSMETIC (D-37). Gate 3 ran with `slide` as a single Linear
    while the SlideProbe that judged it had two hidden layers. That let the encoder win by making
    slide identity merely NON-LINEARLY separable - cheap, and it removes nothing. Measured: the
    co-trained head fell to -3.29 bits, below chance, while the fresh probe still recovered 89%.
    `deep_slide=True` matches the head to the probe, so the encoder has to actually remove the
    information to beat it. Default stays False so the Gate 3 result reproduces exactly.
    """

    def __init__(self, d_z, n_class, n_slide, n_cohort, cohort_frac=0.1, deep_slide=False,
                 hidden=256):
        super().__init__()
        self.cls = nn.Linear(d_z, n_class)
        self.slide = (nn.Sequential(nn.Linear(d_z, hidden), nn.GELU(),
                                    nn.Linear(hidden, hidden), nn.GELU(),
                                    nn.Linear(hidden, n_slide))
                      if deep_slide else nn.Linear(d_z, n_slide))
        self.cohort = nn.Linear(d_z, n_cohort)
        self.cohort_frac = cohort_frac
        self.deep_slide = deep_slide

    def forward(self, z, lam=0.0):
        return (self.cls(z),
                self.slide(grad_reverse(z, lam)),
                self.cohort(grad_reverse(z, lam * self.cohort_frac)))


class SlideProbe(nn.Module):
    """A FRESH discriminator, trained from scratch on a frozen encoder.

    This is the honest measurement the original plan lacked. A co-trained discriminator can be
    beaten without the information being gone - the encoder only has to find a direction that
    particular discriminator is not looking in. Training a new one from scratch, and scoring it on
    slides it has never seen, asks whether the information is actually absent.

    Two hidden layers rather than a linear probe, so a low score cannot be dismissed as the probe
    being too weak to find what is there.
    """

    def __init__(self, d_z, n_slide, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_z, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, n_slide))

    def forward(self, z):
        return self.net(z)
