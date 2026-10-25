"""Image observations for the torus models (IMAGE_TORUS_PREREG.md). Wraps the existing classes; edits none of them.

The token stream stays GridWorld's (actions 0-3, classes 4-19, blank 20): the loss, the targets (class tokens), the
revisit mask and every floor are the token task's at the same seed. `use_image_obs` replaces a model's `token_emb`
by `ImageObsEmbedding`, which keeps the learned token rows for ACTIONS and, at every observation slot, feeds a small
CNN an IMAGE instead:
  - class token k  -> a fresh random instance of class k, drawn at every forward from the ACTIVE split's bank:
    split None (default) = "train" in training mode, "test" in eval mode; set_split(m, "train") lets an eval
    readout use train images; an explicit "test" in training mode raises;
  - blank token    -> the all-zero image (constant, like the token task's single blank token);
  the encoder output is multiplied by the fixed gain ENC_GAIN (Amendment 1) so that image and token observation
  embeddings start at the same norm;
so recognition must be invariant to the instance. The banks live on the model's device as non-persistent uint8
buffers (no loader, no checkpoint bloat); the instance draw uses a private torch.Generator, so the global RNG
stream (dropout, init) is the token twin's exactly.

Pairing: the encoder is initialised from its own seed inside fork_rng, so at a seed every arm (and the token twin)
has the same base weights and the same encoder; eval reseeds the instance generator per walk, so every arm sees the
same images on the same walk.

Observation slots are the ODD positions of the interleaved stream ([a1, o1, a2, o2, ...], environment.py
invariant). The CNN runs on every odd slot (blank slots get the zero image), so shapes are static and training adds
no host-device sync. A position whose token is not an observation keeps its token row; `strict` (checked outside
training) refuses observation tokens at even positions and action tokens at odd ones.

Arms (`build`): MapWM = train_variant's Vanilla_r4; ActOnly / NormStep = model_codes' LEAK remedies (step from
action tokens only / step reads LayerNorm(embedding)); RoPE = train_variant's RoPE (index rotary; --n-layers 4 is
the depth reference). obs = "image" or "token" (the token twin: the unmodified model).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_ACT, N_CLS = 4, 16
BLANK = N_ACT + N_CLS          # 20, GridWorld(n_obs_types=16).unified_blank
VOCAB = BLANK + 1              # 21
ENC_SEED_MUL, ENC_SEED_ADD = 7919, 17
# Amendment 1 (M2): FIXED output gain of the encoder, a constant buffer (not a parameter: the optimiser sees the same
# parameterisation in every arm). Rule: G = sqrt(d_model) / (mean over the encoder inits of run seeds 0-15 of the mean
# output norm on the first 64 train-split images of each class), so that at init an image observation embedding has
# the norm of a token row (nn.Embedding N(0, 1) rows: E|row| ~ sqrt(128) = 11.31). Seed-independent (the batch seeds
# 3100-3107 are not used); `calibrate_gain()` recomputes it (test_image_torus checks the constant). Measured
# 2026-10-25: mean init norm 0.3725 (sd 0.035 over the 16 inits) -> G = 30.3729.
ENC_GAIN = 30.3729
SAMPLE_SEED_MUL, SAMPLE_SEED_ADD = 1_000_003, 7


class ImageEncoder(nn.Module):
    """28x28 grey -> d_model. Three stride-2 convs (14, 7, 4 px), ReLU, no normalisation layer (batch statistics
    would couple sequence positions), linear readout. 79,968 parameters at d_model 128 (the transformer is ~204k)."""

    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 4 * 4, d_model))

    def forward(self, x):                      # x: (N, 1, 28, 28) float in [0, 1]
        return self.net(x)


class ImageObsEmbedding(nn.Module):
    def __init__(self, tok_emb, d_model, banks, sample_seed, enc_seed):
        super().__init__()
        self.tok = tok_emb                       # the model's own nn.Embedding; only action rows are used
        with torch.random.fork_rng(devices=[]):  # the encoder init leaves the global RNG untouched
            torch.manual_seed(enc_seed)
            self.encoder = ImageEncoder(d_model)
        self.register_buffer("enc_gain", torch.tensor(float(ENC_GAIN)))   # fixed scale (Amendment 1, M2)
        for split in ("train", "test"):
            b = banks[split]
            assert b["images"].dtype == torch.uint8 and tuple(b["images"].shape[1:]) == (28, 28)
            assert len(b["cnt"]) == N_CLS and int(b["cnt"].min()) > 0
            self.register_buffer(f"imgs_{split}", b["images"], persistent=False)
            self.register_buffer(f"off_{split}", b["off"].long(), persistent=False)
            self.register_buffer(f"cnt_{split}", b["cnt"].long(), persistent=False)
        self.split = None                        # None: follow the mode (train -> "train", eval -> "test")
        self.strict = True
        self.sample_seed = int(sample_seed)
        self._gen = None
        self.draws = {"train": 0, "test": 0}     # observation slots served per split, counted host-side (no sync)
        self.last_idx = None                     # (B, n) bank indices of the last forward (tests / readouts)

    def reseed(self, seed):
        self.sample_seed = int(seed); self._gen = None

    def _generator(self, dev):
        if self._gen is None or self._gen.device != dev:
            self._gen = torch.Generator(device=dev).manual_seed(self.sample_seed)
        return self._gen

    def _check(self, tokens):
        ev, od = tokens[:, 0::2], tokens[:, 1::2]
        if bool((ev >= N_ACT).any()) or bool((od < N_ACT).any()):
            raise ValueError("ImageObsEmbedding: stream is not interleaved [action, observation, ...]")

    def active_split(self):
        return self.split or ("train" if self.training else "test")

    def forward(self, tokens):
        if self.training and self.split not in (None, "train"):
            raise RuntimeError("ImageObsEmbedding: test-split images requested in training mode")
        split = self.active_split()
        if self.strict and not self.training:
            self._check(tokens)
        x = self.tok(tokens)
        ot = tokens[:, 1::2]
        B, n = ot.shape
        if n == 0:
            return x
        dev = tokens.device
        is_obj = (ot >= N_ACT) & (ot < BLANK)
        cls = (ot - N_ACT).clamp(0, N_CLS - 1)
        off, cnt = getattr(self, f"off_{split}"), getattr(self, f"cnt_{split}")
        imgs = getattr(self, f"imgs_{split}")
        u = torch.rand(B, n, generator=self._generator(dev), device=dev)
        c = cnt[cls]
        idx = off[cls] + torch.minimum((u * c).long(), c - 1)
        self.last_idx = idx
        im = imgs[idx]                                             # (B, n, 28, 28) uint8
        im = torch.where(is_obj[..., None, None], im, torch.zeros_like(im))   # blank -> the zero image
        e = (self.encoder(im.reshape(B * n, 1, 28, 28).to(x.dtype) / 255.0) * self.enc_gain).view(B, n, -1)
        slot = torch.where((is_obj | (ot == BLANK))[..., None], e, x[:, 1::2])
        out = x.clone()
        out[:, 1::2] = slot
        self.draws[split] += B * n
        return out


def calibrate_gain(d_model=128, seeds=range(16), n_per_class=64, bank=None):
    """The ENC_GAIN rule (Amendment 1): sqrt(d_model) / mean init output norm over the encoder inits of run seeds 0-15,
    on the first n_per_class train-split images of each class. Returns (gain, per-seed mean norms)."""
    import math
    if bank is None:
        from mapformer import image_bank
        bank = image_bank.load_split("train")
    idx = torch.cat([bank["off"][k] + torch.arange(n_per_class) for k in range(N_CLS)])
    x = bank["images"][idx].float()[:, None] / 255.0
    norms = []
    for s in seeds:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed((s * ENC_SEED_MUL + ENC_SEED_ADD) % (2 ** 31))
            e = ImageEncoder(d_model)
        with torch.no_grad():
            norms.append(float(e(x).norm(dim=1).mean()))
    return math.sqrt(d_model) / (sum(norms) / len(norms)), norms


def use_image_obs(model, banks, seed):
    d = model.token_emb.embedding_dim
    model.token_emb = ImageObsEmbedding(
        model.token_emb, d, banks,
        sample_seed=(seed * SAMPLE_SEED_MUL + SAMPLE_SEED_ADD) % (2 ** 31),
        enc_seed=(seed * ENC_SEED_MUL + ENC_SEED_ADD) % (2 ** 31))
    return model


def set_split(model, split):
    assert split in (None, "train", "test"), split
    model.token_emb.split = split


def image_emb(model):
    e = model.token_emb
    return e if isinstance(e, ImageObsEmbedding) else None


# ------------------------------------------------------------------ arms
def _classes():
    from mapformer.train_variant import VARIANT_MAP
    from mapformer.model_codes import MapWM_ActOnly, MapWM_NormStep
    return {"MapWM": VARIANT_MAP["Vanilla_r4"], "ActOnly": MapWM_ActOnly, "NormStep": MapWM_NormStep,
            "RoPE": VARIANT_MAP["RoPE"]}


ARMS = ("MapWM", "ActOnly", "NormStep", "RoPE")
PATH_ARMS = ("MapWM", "ActOnly", "NormStep")


def build(arm, obs, seed, n_layers=1, banks=None, d_model=128, n_heads=2, grid_size=64, seed_rng=True):
    """The model of one cell. seed_rng=True seeds torch as train_variant does (torch.manual_seed(seed) right before
    the model is built; the environment draws no torch numbers), so the token twin of MapWM is bitwise
    train_variant's Vanilla_r4 at the same seed."""
    assert arm in ARMS and obs in ("image", "token"), (arm, obs)
    if seed_rng:
        torch.manual_seed(seed)
    m = _classes()[arm](vocab_size=VOCAB, d_model=d_model, n_heads=n_heads, n_layers=n_layers, grid_size=grid_size)
    assert len(m.layers) == n_layers, (len(m.layers), n_layers)        # rule 17
    if obs == "image":
        assert banks is not None
        use_image_obs(m, banks, seed)
    return m


def step_of(model, tokens):
    """The step map's output (B, L, H, n_blocks) for this arm on these tokens (image arms: draws images).
    None for arms without a step (RoPE)."""
    if not hasattr(model, "action_to_lie"):
        return None
    x = model.token_emb(tokens)
    if hasattr(model, "step"):                   # model_codes._StepOverride arms
        return model.step(tokens, x)
    return model.action_to_lie(x)


def cell_name(arm, obs, n_layers):
    return f"{'I' if obs == 'image' else 'T'}_{arm}{'' if n_layers == 1 else f'_L{n_layers}'}"
