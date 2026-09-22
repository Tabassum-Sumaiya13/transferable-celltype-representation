"""
Marker value encoder + the two FiLM variants raced in GATE 1.

A cell arrives as, per marker, one or two VALUE CHANNELS. Each channel is an empirical-CDF
position in [0,1]; which group the ECDF was taken inside is the thing Gate 1 measures:

    u_img   ECDF inside (image, marker)    strongest batch removal, destroys slide prevalence
    u_coh   ECDF inside (cohort, marker)   keeps prevalence, leaves slide drift in

A NOTE ON WHAT WAS PLANNED AND MEASURED WRONG. The plan called for a second channel holding
"the value relative to a cohort-level reference" - tanh of a robust z-score - to stop ranking
from destroying prevalence. Building it showed two problems. It saturates: 5.5-17.5% of cells
land at |lvl| > 0.99, worst on Sorin, which arrives uint8 so most markers have a median of 0
and a near-zero IQR. And more fundamentally, ANY per-cell function of the raw value computed
from cohort statistics is a monotone transform of that value, so it carries exactly the
information `u_coh` already carries (measured correlation 0.89-0.97). It is not a second
source of information; it is `u_coh` with a lossy squash on top. Worse, it would have broken
the bake-off: an arm holding {u_img, lvl} strictly contains an arm holding {u_coh, lvl}, so
V1 could not have lost. The second channel is therefore the OTHER GROUPING itself, which is
genuinely independent information, and the arms are built as a ladder so each comparison
isolates one decision.

FiLM (feature-wise linear modulation) is the learned slide-drift correction. It reads
per-(slide, marker) summary statistics and emits a scale and a shift. Three deliberate
constraints, all aimed at one failure - a FiLM that memorises WHICH COHORT it is looking at
would re-inject exactly the information the Stage 3 adversary exists to remove:

  1. The statistics fed in are standardised WITHIN COHORT by harmonise_values.py, so every slide
     lands in one common distribution instead of six separated clouds. Cohort offsets are
     already gone by then (channel 1 is a per-cohort ECDF), so nothing is lost.
  2. The OUTPUT is bounded, not the weights: gamma = 1 + eps*tanh(.), beta = eps*tanh(.).
     Weight decay shrinks parameters but does not bound what the function can do. This does.
     Both heads are zero-initialised, so training starts at exactly the identity and the
     worst case degrades to "FiLM does nothing", which is the V3 arm.
  3. Capacity is chosen by measurement, not assumed - hence two arms:
        V2a  FiLMScalar  4 parameters per marker, reading only that marker's own 2 statistics
        V2b  FiLMMLP     an MLP over the WHOLE slide-statistics vector
     If V2b does not clearly beat V2a, V2a ships. A tie is positive evidence that the extra
     capacity is unnecessary, and the small model has far less to overfit.

Both arms emit a SCALAR gamma/beta per marker applied to channel 1, which keeps |gamma-1| and
|beta| directly readable as "how much slide correction this marker got" - what Gate 1 check 5
reports.
"""
import torch
import torch.nn as nn

EPS_FILM = 0.3          # FiLM may rescale by at most +/-30% and shift by at most +/-0.3


class FiLMScalar(nn.Module):
    """V2a - the low-capacity arm.

    Per marker, gamma and beta are read off that marker's OWN two slide statistics only
    (median and IQR of its cohort-level rank inside the slide). 4 parameters per marker, and
    no path by which one marker's statistics can influence another marker's correction.
    """

    def __init__(self, n_markers, eps=EPS_FILM):
        super().__init__()
        self.wg = nn.Parameter(torch.zeros(n_markers, 2))
        self.wb = nn.Parameter(torch.zeros(n_markers, 2))
        self.eps = eps

    def forward(self, u, s):
        # u: (B, M)   s: (B, M, 2) slide statistics, already standardised within cohort
        g = 1.0 + self.eps * torch.tanh((s * self.wg).sum(-1))
        b = self.eps * torch.tanh((s * self.wb).sum(-1))
        return g * u + b, g, b


class FiLMMLP(nn.Module):
    """V2b - the high-capacity arm.

    An MLP sees the whole slide-statistics vector at once, so one marker's correction may
    depend on how the rest of the slide behaved. Same bounded output and the same
    zero-initialised identity start, so the ONLY difference from V2a is capacity.
    """

    def __init__(self, n_markers, hidden=32, eps=EPS_FILM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * n_markers, hidden), nn.GELU(),
            nn.Linear(hidden, 2 * n_markers),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.n_markers, self.eps = n_markers, eps

    def forward(self, u, s):
        h = self.net(s.flatten(1)).view(-1, self.n_markers, 2)
        g = 1.0 + self.eps * torch.tanh(h[:, :, 0])
        b = self.eps * torch.tanh(h[:, :, 1])
        return g * u + b, g, b


FILMS = {'scalar': FiLMScalar, 'mlp': FiLMMLP}


class MarkerEncoder(nn.Module):
    """value channels -> a d-dim token per marker, plus a learned marker-identity embedding.

    The value MLP is SHARED across markers and identity is added separately, rather than one
    private MLP per marker. That is already the Stage 2 token design: a panel the model has
    never seen still gets encoded, because only the identity embedding is marker-specific.
    """

    def __init__(self, n_markers, n_ch=1, d=32, hidden=64):
        super().__init__()
        self.value = nn.Sequential(nn.Linear(n_ch, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.ident = nn.Embedding(n_markers, d)
        self.d, self.n_markers, self.n_ch = d, n_markers, n_ch

    def forward(self, ch):
        # ch: (B, M, n_ch)
        return self.value(ch) + self.ident.weight


class MaskedMarkerProbe(nn.Module):
    """The GATE 1 arm-comparison model: hide one marker, predict its value from the rest.

    Why this task. Gate 1 must be scored leave-one-cohort-out, because fitting to slide
    statistics is invisible on held-out CELLS (same slides, same statistics) and only shows up
    on a held-out COHORT. But a cross-cohort classifier needs a shared label space, and that
    does not exist until Stage 1b. Masked-marker regression needs no labels at all, is
    cross-cohort by construction, and is the exact objective Stage 2 trains on - so the arm
    that wins here is the arm that helps downstream.

    The prediction target is ALWAYS the cohort-level ECDF value, identical for every arm.
    Only the INPUT representation differs, so the R2 numbers are comparable.

    Tokens are pooled by CONCATENATION over the marker axis, not by averaging. Averaging was
    tried first and underfits badly - it lands below a plain linear regression on the same
    task, because the mean of `value_mlp(v) + identity` mixes every marker's value into one
    vector and the head can no longer tell which value came from which marker. Concatenating
    is fine here precisely because Gate 1 races normalisation arms on a FIXED shared core, so
    M is the same in every cohort and every fold. Handling a panel that varies is Stage 2's
    job, and that is what the set transformer and the [ABSENT] token are for - this probe is
    deliberately the simplest thing that measures the arms fairly.
    """

    def __init__(self, n_markers, film=None, n_ch=1, d=32, hidden=128):
        super().__init__()
        self.enc = MarkerEncoder(n_markers, n_ch=n_ch, d=d, hidden=64)
        self.mask_token = nn.Parameter(torch.zeros(d))
        self.head = nn.Sequential(
            nn.Linear(n_markers * d + d, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.film = FILMS[film](n_markers) if film else None

    def forward(self, ch, s, mask_idx):
        # ch: (B, M, n_ch)   s: (B, M, 2)   mask_idx: (B,)
        gb = None
        if self.film is not None:
            u, g, b = self.film(ch[..., 0], s)          # FiLM corrects channel 1 only
            ch = torch.cat([u.unsqueeze(-1), ch[..., 1:]], dim=-1)
            gb = (g, b)

        tok = self.enc(ch)
        B, M, d = tok.shape
        hit = torch.zeros(B, M, dtype=torch.bool, device=tok.device)
        hit[torch.arange(B, device=tok.device), mask_idx] = True

        # blank the hidden marker so its value cannot leak, then keep the marker axis intact
        tok = torch.where(hit.unsqueeze(-1), self.mask_token.expand(B, M, d), tok)
        query = self.enc.ident(mask_idx)                # (B, d) "which one is hidden"
        return self.head(torch.cat([tok.flatten(1), query], -1)).squeeze(-1), gb
