"""
STAGE 6 - losses, prototypes, and the guards that make them safe.

Three losses exist on this roster, not the four the design planned. The neighbourhood-context
loss predicts a cell's spatial surroundings and needs `z_neigh` from Stage 4, which is deferred
(D-18), so it has no input and is not written. The declared "2 vs 4" ablation is therefore a
2 vs 3, and gate6_expect.csv says so rather than renumbering it quietly.

  cell type       prototype loss, weight PINNED at 1.0
  masked marker   Stage 2's objective carried forward as an auxiliary
  VICReg          variance / invariance / covariance regularisation

WHAT IS DELIBERATELY ABSENT
  Descendant-tolerant cross-entropy. It needs the Stage 1b nesting graph, which missed all three
  cases declared for it (section 7 gap 2). A loss built on an untrusted parent/child graph would
  reward wrong predictions silently, and the failure would read as a modelling result rather than
  a graph defect. Deferred, not rejected - re-test nesting and it drops in unchanged.

  The adversary. Gate 3 measured 5 lambdas x 5 folds and none beat lambda=0 (D-36).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Prototypes(nn.Module):
    """One learnable vector per Stage 1b cluster, in z_cell space.

    Learnable ON PURPOSE: the protein data is allowed to correct Stage 1b's clustering during
    training. Where a prototype travels a long way, the model is disagreeing with the label space
    - that is a finding worth reporting, which is why `drift_report` exists.

    Initialisation is NOT random. work/prototypes.npy holds each cluster's mean marker signature
    over the canonical 99-triple vocabulary, so a signature is encoded as a synthetic cell through
    the frozen encoder and the result becomes that cluster's starting point. A random start would
    throw away the one thing Stage 1b actually knows.
    """

    def __init__(self, n_class, d_z, init=None, temp=0.1):
        super().__init__()
        self.p = nn.Parameter(torch.randn(n_class, d_z) * 0.02 if init is None
                              else init.clone().float())
        self.temp = temp
        self.register_buffer('p0', self.p.detach().clone())
        self.register_buffer('frozen', torch.zeros(n_class, n_class, dtype=torch.bool))

    def logits(self, z):
        """Cosine similarity to each prototype, temperature-scaled.

        Cosine rather than Euclidean so the loss cannot be minimised by inflating ||z||, which is
        the cheap degenerate solution for a distance-based classifier.
        """
        return (F.normalize(z, dim=-1) @ F.normalize(self.p, dim=-1).t()) / self.temp

    def pairwise(self, detach=True):
        """Pairwise prototype distances. Detached by default - the guard only ever MEASURES, and
        letting its gradient reach the prototypes would turn a diagnostic into a silent loss."""
        p = self.p.detach() if detach else self.p
        n = F.normalize(p, dim=-1)
        d = torch.cdist(n.unsqueeze(0), n.unsqueeze(0)).squeeze(0)
        return d + torch.eye(len(d), device=d.device) * 1e9   # ignore the diagonal

    def min_distance(self):
        return float(self.pairwise().min())

    def drift(self):
        """How far each prototype has moved from its Stage 1b initialisation, cosine distance."""
        a, b = F.normalize(self.p, dim=-1), F.normalize(self.p0, dim=-1)
        return (1.0 - (a * b).sum(-1)).detach().cpu().numpy()


class CollapseGuard:
    """Watch the minimum pairwise prototype distance and freeze pairs that breach the floor.

    Without this a class can vanish - two prototypes merge, one label stops being predicted, and
    macro-F1 over classes present in the truth can still look reasonable because the surviving
    class absorbs both. The guard makes the merge visible and stops it getting worse.

    A breach FREEZES the offending pair rather than aborting: two clusters merging may be a true
    statement about the label space, and killing the run would discard the evidence.
    """

    def __init__(self, protos, floor_frac=0.5):
        self.protos = protos
        self.d0 = protos.min_distance()
        self.floor = self.d0 * floor_frac
        self.history, self.events = [], []

    def step(self, epoch):
        d = self.protos.pairwise()
        m = float(d.min())
        self.history.append(m)
        if m < self.floor:
            i, j = np.unravel_index(int(d.argmin().cpu()), d.shape)
            i, j = int(i), int(j)
            if not bool(self.protos.frozen[i, j]):
                self.protos.frozen[i, j] = self.protos.frozen[j, i] = True
                self.events.append(dict(epoch=epoch, a=i, b=j, distance=round(m, 4)))
        return m

    def passed(self):
        """PASS means the minimum never fell below half its starting value."""
        return bool(self.history) and min(self.history) > self.floor


class KendallSigma(nn.Module):
    """Learnable uncertainty weighting over the AUXILIARY losses only.

    Each auxiliary term is weighted 1/(2 sigma^2) with a +log(sigma) penalty, so the model can
    down-weight a task it finds unhelpful.

    THE CELL-TYPE SIGMA IS NOT LEARNED, and that is the whole design point. Kendall weighting is
    free to drive a loss toward zero when its labels look noisy - and cross-cohort cell-type
    labels look extremely noisy, because they come from six different annotation schemes. Left
    learnable, the one task that matters is the one most likely to be switched off. It is pinned
    at weight 1.0.

    Gate 6 check 1 caps how far any auxiliary log-sigma may travel, because a runaway sigma
    deletes its term while the training curve still looks perfectly healthy.
    """

    def __init__(self, names):
        super().__init__()
        self.names = list(names)
        self.log_sigma = nn.Parameter(torch.zeros(len(self.names)))

    def forward(self, losses):
        total, parts = 0.0, {}
        for i, n in enumerate(self.names):
            if n not in losses or losses[n] is None:
                continue
            s = self.log_sigma[i]
            w = torch.exp(-2.0 * s)
            total = total + w * losses[n] + s
            parts[n] = float(w.detach())
        return total, parts

    def snapshot(self):
        return {n: float(v) for n, v in zip(self.names, self.log_sigma.detach().cpu())}


def vicreg(z, gamma=1.0, eps=1e-4):
    """Variance and covariance terms of VICReg, as a representation regulariser.

    The invariance term needs two augmented views of the same cell and there is no declared
    augmentation for spatial proteomics values, so only variance and covariance are used - the
    two that act on a single view. Variance keeps every dimension of z alive; covariance pushes
    dimensions to carry different information rather than duplicating one another.

    This is the term Gate 6 check 3 decides on. If it does not improve validation macro-F1 it is
    dropped, and that is reported.
    """
    z = z - z.mean(0)
    std = torch.sqrt(z.var(0) + eps)
    v = F.relu(gamma - std).mean()
    n, d = z.shape
    if n < 2:
        return v
    cov = (z.t() @ z) / (n - 1)
    off = cov - torch.diag(torch.diag(cov))
    return v + (off.pow(2).sum() / d)


def masked_marker_loss(model_out, target, hide):
    """Stage 2's objective carried forward: predict a hidden marker's u_coh from the others.

    Label-free, so it works on every cell of every cohort regardless of annotation scheme. It is
    what keeps the representation general rather than collapsing onto whatever the 25 clusters
    happen to separate.
    """
    if not bool(hide.any()):
        return model_out.sum() * 0.0
    return F.mse_loss(model_out[hide], target[hide])


def confidence_weighted_ce(logits, y, w=None):
    """Cross-entropy, optionally weighted per cell by label confidence.

    1-OF-5 COVERAGE and it must be reported that way: UPMC ships kNN.prob spanning 0.14 to 1.0,
    while CRC, Keren, Phillips and Sorin all report a constant 1.0. On four cohorts this is
    exactly equal to plain cross-entropy.
    """
    ce = F.cross_entropy(logits, y, reduction='none')
    return ce.mean() if w is None else (ce * w).sum() / w.sum().clamp_min(1e-8)
