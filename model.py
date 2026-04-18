import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from timm.layers import DropPath
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


# ──────────────────────────────────────────────────────────────
#  Batched SSM core layer
# ──────────────────────────────────────────────────────────────

class BatchedMambaCore(nn.Module):
    """Batched multi-direction Mamba: packs K scan directions into a single selective_scan call.

    K=2: bidirectional (forward + backward)
    K=4: four directions (forward + backward + even-odd + odd-even)
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, K=4,
                 seq_len=None, dropout=0.0, merge_mode="add"):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        self.K = K
        self.merge_mode = merge_mode
        self.capture_states = False
        self._cached_states = None

        # ── Shared parameters ──
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding='same',
            groups=self.d_inner,
        )
        self.act = nn.SiLU()
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # ── Per-direction parameters (K groups, stored together) ──
        # x_proj: projects d_inner to (dt_rank + 2*d_state), independent per direction
        self.x_proj_weight = nn.Parameter(
            torch.randn(K, self.dt_rank + d_state * 2, self.d_inner)
            * (self.d_inner ** -0.5)
        )

        # dt_proj: projects dt_rank to d_inner, independent per direction
        dt_init_std = self.dt_rank ** -0.5
        self.dt_projs_weight = nn.Parameter(
            torch.empty(K, self.d_inner, self.dt_rank).uniform_(-dt_init_std, dt_init_std)
        )
        # dt bias: initialized so softplus(bias) falls in [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(K, self.d_inner) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_projs_bias = nn.Parameter(inv_dt)

        # A: S4D initialization, stored in log form
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            "n -> d n", d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_logs = nn.Parameter(
            repeat(A_log, "d n -> r d n", r=K).flatten(0, 1).contiguous()
        )  # (K*d_inner, d_state)
        self.A_logs._no_weight_decay = True

        # D: skip connection
        self.Ds = nn.Parameter(
            repeat(torch.ones(self.d_inner), "d -> r d", r=K).flatten(0, 1).contiguous()
        )  # (K*d_inner,)
        self.Ds._no_weight_decay = True

        # ── Even-odd / odd-even permutation indices (used when K=4) ──
        if K == 4 and seq_len is not None:
            eo_perm, eo_inv = self._build_interleave_indices(seq_len, even_first=True)
            oe_perm, oe_inv = self._build_interleave_indices(seq_len, even_first=False)
            self.register_buffer('eo_perm', eo_perm)
            self.register_buffer('eo_inv', eo_inv)
            self.register_buffer('oe_perm', oe_perm)
            self.register_buffer('oe_inv', oe_inv)

        # ── Projection layer for concat mode ──
        if merge_mode == "concat":
            self.merge_proj = nn.Linear(self.d_inner * K, self.d_inner, bias=False)

    @staticmethod
    def _build_interleave_indices(L, even_first=True):
        if even_first:
            perm = list(range(0, L, 2)) + list(range(1, L, 2))
        else:
            perm = list(range(1, L, 2)) + list(range(0, L, 2))
        inv_perm = [0] * L
        for i, p in enumerate(perm):
            inv_perm[p] = i
        return torch.tensor(perm, dtype=torch.long), torch.tensor(inv_perm, dtype=torch.long)

    def _cross_scan(self, x):
        """Expand input into K 1D sequences along different directions.
        x: (B, D, L) → xs: (B, K, D, L)
        """
        B, D, L = x.shape
        if self.K == 1:
            return x.unsqueeze(1)                    # (B, 1, D, L)
        elif self.K == 2:
            xs = x.new_empty(B, 2, D, L)
            xs[:, 0] = x
            xs[:, 1] = x.flip(-1)
        else:  # K == 4
            xs = x.new_empty(B, 4, D, L)
            xs[:, 0] = x                           # forward
            xs[:, 1] = x.flip(-1)                   # backward
            xs[:, 2] = x[:, :, self.eo_perm]        # even-odd
            xs[:, 3] = x[:, :, self.oe_perm]        # odd-even
        return xs

    def _cross_merge(self, ys):
        """Merge K scan results back to the original order.
        ys: (B, K, D, L) → y: (B, D, L)
        """
        # Align each direction back to the original order
        if self.K == 1:
            return ys[:, 0]                          # (B, D, L) returned as-is
        elif self.K == 2:
            aligned = [ys[:, 0], ys[:, 1].flip(-1)]
        else:  # K == 4
            aligned = [ys[:, 0],
                       ys[:, 1].flip(-1),
                       ys[:, 2][:, :, self.eo_inv],
                       ys[:, 3][:, :, self.oe_inv]]

        if self.merge_mode == "concat":
            # (B, K*D, L) → transpose → (B, L, K*D) → proj → (B, L, D) → transpose
            cat = torch.cat(aligned, dim=1)              # (B, K*D, L)
            y = self.merge_proj(cat.transpose(1, 2))     # (B, L, D)
            return y.transpose(1, 2)                     # (B, D, L)
        else:  # add
            if self.K == 2:
                return (aligned[0] + aligned[1]) / 2
            else:
                return aligned[0] + aligned[1] + aligned[2] + aligned[3]

    @torch.no_grad()
    def _compute_ssm_states(self, u, delta, A, B, C, delta_bias):
        """Reference pure-Python implementation to compute intermediate SSM states, used for visualization only.

        Returns dict with:
            delta: (B, K*d_inner, L) — time steps after softplus
            hidden_states: (B, K*d_inner, N, L) — hidden state at each step
            B: (B, K, N, L), C: (B, K, N, L) — input/output modulation matrices
        """
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        delta = F.softplus(delta)

        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))

        # B is 4D: (B, K, N, L); expand to (B, K*d_inner, N, L)
        B_exp = B.float()
        B_exp = B_exp.repeat_interleave(self.d_inner, dim=1)  # (B, K*d_inner, N, L)
        deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B_exp, u)

        x = A.new_zeros((batch, dim, dstate))
        hidden_states = []
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            hidden_states.append(x.clone())

        return {
            'delta': delta.detach().cpu(),                                  # (B, K*D, L)
            'hidden_states': torch.stack(hidden_states, dim=-1).detach().cpu(),  # (B, K*D, N, L)
            'B': B.detach().cpu(),                                          # (B, K, N, L)
            'C': C.detach().cpu(),                                          # (B, K, N, L)
        }

    def forward(self, x):
        # x: (B, L, D)  D=d_model
        B, L, D = x.shape

        # ── Input projection + gate split ──
        xz = self.in_proj(x)                               # (B, L, 2*d_inner)
        x_ssm, z = xz.chunk(2, dim=-1)                     # each (B, L, d_inner)
        z = self.act(z)                                     # SiLU gate

        # ── DWConv local feature extraction ──
        x_ssm = x_ssm.transpose(1, 2)                      # (B, d_inner, L)
        x_ssm = self.act(self.conv1d(x_ssm))                  # DWConv + SiLU

        # ── CrossScan: expand into K directions ──
        xs = self._cross_scan(x_ssm)                        # (B, K, d_inner, L)

        # ── Project to SSM parameters dt, B, C (independent per direction) ──
        K = self.K
        R, N = self.dt_rank, self.d_state

        x_dbl = torch.einsum(
            "b k d l, k c d -> b k c l", xs, self.x_proj_weight
        )                                                    # (B, K, R+2N, L)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)

        # dt: rank → d_inner
        dts = torch.einsum(
            "b k r l, k d r -> b k d l", dts, self.dt_projs_weight
        )                                                    # (B, K, d_inner, L)

        # ── Flatten K into D dimension, single selective_scan call ──
        xs_flat = xs.contiguous().view(B, -1, L)             # (B, K*d_inner, L)
        dts_flat = dts.contiguous().view(B, -1, L)           # (B, K*d_inner, L)
        As = -torch.exp(self.A_logs.float())                 # (K*d_inner, N)
        delta_bias = self.dt_projs_bias.view(-1).float()     # (K*d_inner,)

        ys = selective_scan_fn(
            xs_flat, dts_flat, As,
            Bs.contiguous(), Cs.contiguous(),
            self.Ds.float(), z=None,
            delta_bias=delta_bias,
            delta_softplus=True,
            return_last_state=False,
        )                                                    # (B, K*d_inner, L)

        if self.capture_states:
            self._cached_states = self._compute_ssm_states(
                xs_flat, dts_flat, As, Bs, Cs, delta_bias)

        # ── CrossMerge: merge K scan results ──
        ys = ys.view(B, K, self.d_inner, L)
        y = self._cross_merge(ys)                            # (B, d_inner, L)

        # ── Output normalization + gating + projection ──
        y = y.transpose(1, 2)                                # (B, L, d_inner)
        y = self.out_norm(y)
        y = y * z                                            # gate
        out = self.dropout(self.out_proj(y))                 # (B, L, d_model)
        return out




class ChannelDropout(nn.Module):
    """During training, randomly drop entire input channels to prevent the model from relying on any single channel."""
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        # x: [B, L, C]
        if self.training and self.p > 0:
            mask = torch.rand(x.size(0), 1, x.size(2), device=x.device) > self.p
            x = x * mask / (1 - self.p)
        return x


# ──────────────────────────────────────────────────────────────
#  Basic components
# ──────────────────────────────────────────────────────────────

class GatedFFN(nn.Module):
    """SwiGLU gated FFN: gate(x) * up(x) → down"""

    def __init__(self, d_model, expand=4, dropout=0.1):
        super().__init__()
        hidden = d_model * expand
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.w_down(self.act(self.w_gate(x)) * self.w_up(x)))


class AttentionPooling(nn.Module):
    """Attention-weighted pooling."""

    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1),
        )

    def forward(self, x):
        # x: [B, N, D]
        weights = self.attn(x).softmax(dim=1)  # [B, N, 1]
        return (x * weights).sum(dim=1)         # [B, D]


# ──────────────────────────────────────────────────────────────
#  CNN patch embedding (compresses [B, L, C] into [B, L', D])
# ──────────────────────────────────────────────────────────────

class Multi_Resolution_Data(nn.Module):
    """Multi-resolution patch embedding: downsample with Conv1d using different kernel_size/stride values.

    Input:  [B, seq_len, enc_in]
    Output: list of [B, L_i, d_model]  (one per resolution)
    """

    def __init__(self, enc_in, d_model, resolution_list, stride_list):
        super(Multi_Resolution_Data, self).__init__()
        self.multi_res = nn.ModuleList([
            nn.Conv1d(
                in_channels=enc_in,
                out_channels=d_model,
                kernel_size=res,
                stride=s,
                padding=0)
            for res, s in zip(resolution_list, stride_list)
        ])
        self.norms = nn.ModuleList([nn.BatchNorm1d(d_model) for _ in resolution_list])
        self.act = nn.GELU()

    def forward(self, x):
        # x: [B, L, C] → [B, C, L]
        x = x.permute(0, 2, 1)
        x_list = []
        for l in range(len(self.multi_res)):
            out = self.act(self.norms[l](self.multi_res[l](x)))
            x_list.append(out.permute(0, 2, 1))  # [B, L_i, d_model]
        return x_list
# ──────────────────────────────────────────────────────────────
#  1D bidirectional Mamba block (operates on [B, L', D], K=2)
# ──────────────────────────────────────────────────────────────

class BiMambaBlock(nn.Module):
    """Mamba block: K=2 bidirectional (forward + backward), K=1 unidirectional (forward only)."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2,
                 seq_len=125, dropout=0.1, drop_path=0.0, merge_mode="concat",
                 bidirectional=True):
        super().__init__()
        K = 2 if bidirectional else 1
        self.core = BatchedMambaCore(
            d_model, d_state, d_conv, expand, K=K,
            seq_len=seq_len, dropout=dropout, merge_mode=merge_mode)
        self.norm = nn.LayerNorm(d_model)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        # x: [B, L', D]
        return self.drop_path(self.core(self.norm(x)))


# ──────────────────────────────────────────────────────────────
#  Multi-scale CNN patchify + bidirectional Mamba classifier
# ──────────────────────────────────────────────────────────────

class ChannelMixing(nn.Module):
    """Lightweight channel interaction layer: cross-channel feature mixing over the EEG channel dimension.

    Uses 1x1 Conv (i.e. pointwise) to learn spatial relationships between channels,
    e.g. differential patterns like "frontal vs. temporoparietal".

    Input:  [B, L, C]
    Output: [B, L, C]
    """

    def __init__(self, n_channels, expand=2, dropout=0.1):
        super().__init__()
        hidden = n_channels * expand
        self.net = nn.Sequential(
            nn.Linear(n_channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_channels),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(n_channels)

    def forward(self, x):
        # x: [B, L, C] — residual connection
        return x + self.net(self.norm(x))


class MultiScalePatchMambaClassifier(nn.Module):
    """Multi-scale CNN stem + bidirectional Mamba (K=2) classifier.

    Multi-branch CNN stem downsamples with different strides; each branch is
    scanned by an independent BiMamba, and features from all scales are fused
    for classification.

    Input:  [B, seq_len, input_dim]  e.g. [B, 250, 12]
    Output: [B, num_classes]
    """

    def __init__(self, input_dim=12, d_model=128, d_state=16, d_conv=4,
                 expand=2, ffn_expand=4, n_layers=2, num_classes=5, dropout=0.1,
                 seq_len=250, drop_path_rate=0.1,
                 patch_strides=(2, 5, 10), channel_drop=0.1,
                 merge_mode="concat", use_channel_mix=False,
                 bidirectional=True, **kwargs):
        super().__init__()
        self.use_channel_mix = use_channel_mix
        self.channel_dropout = ChannelDropout(p=channel_drop)
        self.n_scales = len(patch_strides)
        actual_strides = patch_strides

        # DropPath schedule (shared across scales)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]

        # ── Channel interaction layer ──
        if use_channel_mix:
            cm_in = input_dim
            self.channel_mix = ChannelMixing(cm_in, expand=2, dropout=dropout)

        # Multi-resolution patch embedding
        embed_in = input_dim

        self.multi_res_embed = Multi_Resolution_Data(
            embed_in, d_model,
            resolution_list=list(patch_strides),
            stride_list=list(actual_strides))

        # Per-scale: pos embed + MH-BiMamba blocks + norm + pool
        self.pos_embeds = nn.ParameterList()
        self.scale_layers = nn.ModuleList()
        self.final_norms = nn.ModuleList()
        self.pools = nn.ModuleList()

        for kernel, stride in zip(patch_strides, actual_strides):
            patched_len = (seq_len - kernel) // stride + 1

            self.pos_embeds.append(
                nn.Parameter(torch.randn(1, patched_len, d_model) * 0.02))

            blocks = nn.ModuleList()
            for i in range(n_layers):
                block_modules = [
                    BiMambaBlock(d_model, d_state, d_conv, expand,
                                 seq_len=patched_len,
                                 dropout=dropout, drop_path=dpr[i],
                                 merge_mode=merge_mode,
                                 bidirectional=bidirectional),
                    nn.LayerNorm(d_model),
                    GatedFFN(d_model, ffn_expand, dropout=dropout),
                    DropPath(dpr[i]) if dpr[i] > 0.0 else nn.Identity(),
                ]
                blocks.append(nn.ModuleList(block_modules))
            self.scale_layers.append(blocks)
            self.final_norms.append(nn.LayerNorm(d_model))
            self.pools.append(AttentionPooling(d_model))

        self.dropout_layer = nn.Dropout(dropout)

        # Fusion: concat pooled outputs from each scale → project back to d_model
        fusion_in = d_model * self.n_scales
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_in),
            nn.Linear(fusion_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def enable_visualization(self):
        """Enable SSM state capture (used for visualization only; slows down forward pass)."""
        for blocks in self.scale_layers:
            for mamba_block, *_ in blocks:
                mamba_block.core.capture_states = True

    def disable_visualization(self):
        """Disable SSM state capture and release cached states."""
        for blocks in self.scale_layers:
            for mamba_block, *_ in blocks:
                mamba_block.core.capture_states = False
                mamba_block.core._cached_states = None

    def get_ssm_states(self):
        """Collect cached SSM states from all layers.

        Returns: dict[scale_idx][layer_idx] → {delta, hidden_states, B, C}
        """
        states = {}
        for s_idx, blocks in enumerate(self.scale_layers):
            states[s_idx] = {}
            for l_idx, (mamba_block, *_) in enumerate(blocks):
                if mamba_block.core._cached_states is not None:
                    states[s_idx][l_idx] = mamba_block.core._cached_states
        return states

    def forward(self, x):
        # x: [B, L, C] = [B, 256, 19]
        x = self.channel_dropout(x)
        # ── Channel interaction ──
        if self.use_channel_mix:
            x = self.channel_mix(x)                       # [B, L, 2C]

        # ── Temporal branch: multi-scale BiMamba ──
        multi_res_list = self.multi_res_embed(x)          # list of [B, Li, D]

        scale_feats = []
        for i in range(self.n_scales):
            h = multi_res_list[i]                         # [B, Li, D]
            h = h + self.pos_embeds[i]
            h = self.dropout_layer(h)

            for block in self.scale_layers[i]:
                mamba, norm2, gffn, dp2 = block[0], block[1], block[2], block[3]
                h = h + mamba(h)
                h = h + dp2(gffn(norm2(h)))
            h = self.final_norms[i](h)

            h = self.pools[i](h)                          # [B, D]
            scale_feats.append(h)

        # ── Fusion ──
        fused = self.fusion(torch.cat(scale_feats, dim=-1))                         # [B, D]
        logits = self.head(fused)                          # [B, num_classes]
        return logits, fused
