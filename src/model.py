"""LACE-VQA: Lightweight Answerability-aware Cross-modal Encoder on top of a frozen CLIP backbone,
plus the CLIP-global-feature MLP baseline (re-implementation in the spirit of Deuser et al., 2022).
"""
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LACEConfig:
    dim_vision: int = 768      # CLIP vision hidden size
    dim_text: int = 512        # CLIP text hidden size
    dim_proj: int = 512        # CLIP joint embedding size
    n_img_tokens: int = 49
    max_q_len: int = 32
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 2
    ffn_mult: int = 4
    dropout: float = 0.1
    bidirectional: bool = True   # also update image tokens with question context
    use_patch: bool = True       # use patch tokens (False -> global features only)
    use_global: bool = True      # CLIP global-embedding shortcut branch
    n_answers: int = 3000
    n_types: int = 4
    # residual fusion (v2): a wide, strongly regularised global MLP (as in the CLIP-MLP baseline) gives the
    # base representation; the patch-token path adds a zero-initialised residual correction to it.
    residual_fusion: bool = False
    global_hidden: int = 1024
    global_dropout: float = 0.5
    global_inputs: int = 4       # 2: [g_v; g_q]   4: [g_v; g_q; g_v*g_q; |g_v-g_q|]

    def to_dict(self):
        return asdict(self)


class FFN(nn.Module):
    def __init__(self, d, mult, p):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d * mult), nn.GELU(), nn.Dropout(p), nn.Linear(d * mult, d))

    def forward(self, x):
        return self.net(x)


class FusionLayer(nn.Module):
    """Question self-attention -> question-to-image cross-attention -> FFN;
    optionally image-to-question cross-attention -> FFN (co-attention)."""

    def __init__(self, c: LACEConfig):
        super().__init__()
        d, h, p = c.d_model, c.n_heads, c.dropout
        self.ln_q1, self.ln_q2, self.ln_q3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.cross_q2v = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.ffn_q = FFN(d, c.ffn_mult, p)
        self.drop = nn.Dropout(p)
        self.bidirectional = c.bidirectional
        if c.bidirectional:
            self.ln_v1, self.ln_v2, self.ln_qkv = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
            self.cross_v2q = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
            self.ffn_v = FFN(d, c.ffn_mult, p)

    def forward(self, q, v, q_pad):
        x = self.ln_q1(q)
        q = q + self.drop(self.self_attn(x, x, x, key_padding_mask=q_pad, need_weights=False)[0])
        kv = self.ln_kv(v)
        q = q + self.drop(self.cross_q2v(self.ln_q2(q), kv, kv, need_weights=False)[0])
        q = q + self.drop(self.ffn_q(self.ln_q3(q)))
        if self.bidirectional:
            qk = self.ln_qkv(q)
            v = v + self.drop(self.cross_v2q(self.ln_v1(v), qk, qk, key_padding_mask=q_pad,
                                             need_weights=False)[0])
            v = v + self.drop(self.ffn_v(self.ln_v2(v)))
        return q, v


class AttnPool(nn.Module):
    """Attention pooling over tokens, optionally conditioned on a guide vector."""

    def __init__(self, d, guided=False):
        super().__init__()
        self.guided = guided
        self.score = nn.Sequential(nn.Linear(d * (2 if guided else 1), d), nn.Tanh(), nn.Linear(d, 1))

    def forward(self, x, pad_mask=None, guide=None):
        h = x if not self.guided else torch.cat([x, guide.unsqueeze(1).expand_as(x)], dim=-1)
        s = self.score(h).squeeze(-1)
        if pad_mask is not None:
            s = s.masked_fill(pad_mask, float("-inf"))
        w = torch.softmax(s, dim=-1)
        return torch.bmm(w.unsqueeze(1), x).squeeze(1), w


class LACEHead(nn.Module):
    def __init__(self, c: LACEConfig):
        super().__init__()
        self.c = c
        d = c.d_model
        parts = 0
        if c.use_patch:
            self.img_in = nn.Sequential(nn.LayerNorm(c.dim_vision), nn.Linear(c.dim_vision, d))
            self.txt_in = nn.Sequential(nn.LayerNorm(c.dim_text), nn.Linear(c.dim_text, d))
            self.img_pos = nn.Parameter(torch.zeros(1, c.n_img_tokens, d))
            self.type_emb = nn.Parameter(torch.zeros(2, d))
            nn.init.trunc_normal_(self.img_pos, std=0.02)
            nn.init.trunc_normal_(self.type_emb, std=0.02)
            self.layers = nn.ModuleList([FusionLayer(c) for _ in range(c.n_layers)])
            self.ln_q, self.ln_v = nn.LayerNorm(d), nn.LayerNorm(d)
            self.pool_q = AttnPool(d)
            self.pool_v = AttnPool(d, guided=True)
            parts += 2
        if c.residual_fusion:
            assert c.use_global, "residual fusion needs the global branch"
            g_in = c.global_inputs * c.dim_proj
            hid = c.global_hidden
            self.glob = nn.Sequential(nn.LayerNorm(g_in), nn.Linear(g_in, hid), nn.GELU(), nn.Dropout(c.global_dropout))
            if c.use_patch:
                self.tok = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, hid), nn.GELU(), nn.Dropout(c.dropout),
                                         nn.Linear(hid, hid))
                nn.init.zeros_(self.tok[-1].weight)
                nn.init.zeros_(self.tok[-1].bias)
        else:
            if c.use_global:
                # [img, txt, img*txt, |img-txt|] of the L2-normalised CLIP embeddings
                self.glob = nn.Sequential(nn.LayerNorm(4 * c.dim_proj), nn.Linear(4 * c.dim_proj, d), nn.GELU(),
                                          nn.Dropout(c.dropout))
                parts += 1
            assert parts > 0, "enable use_patch and/or use_global"
            hid = 2 * d
            self.trunk = nn.Sequential(nn.LayerNorm(parts * d), nn.Linear(parts * d, hid), nn.GELU(),
                                       nn.Dropout(c.dropout))
        self.answer_head = nn.Linear(hid, c.n_answers)
        self.answerable_head = nn.Sequential(nn.Linear(hid, d), nn.GELU(), nn.Linear(d, 1))
        self.type_head = nn.Linear(hid, c.n_types)

    def forward(self, patch, img_global, txt_tokens, txt_mask, txt_global):
        feats = []
        if self.c.use_patch:
            q_pad = txt_mask == 0
            q = self.txt_in(txt_tokens) + self.type_emb[0]
            v = self.img_in(patch) + self.img_pos + self.type_emb[1]
            for layer in self.layers:
                q, v = layer(q, v, q_pad)
            q, v = self.ln_q(q), self.ln_v(v)
            q_vec, _ = self.pool_q(q, q_pad)
            v_vec, _ = self.pool_v(v, guide=q_vec)
            feats += [q_vec, v_vec]
        if self.c.use_global:
            g = torch.cat([img_global, txt_global, img_global * txt_global, (img_global - txt_global).abs()], -1)
            if self.c.residual_fusion and self.c.global_inputs == 2:
                g = g[:, : 2 * self.c.dim_proj]
            if not self.c.residual_fusion:
                feats.append(self.glob(g))
        if self.c.residual_fusion:
            h = self.glob(g)
            if self.c.use_patch:
                h = h + self.tok(torch.cat(feats, dim=-1))
        else:
            h = self.trunk(torch.cat(feats, dim=-1))
        return {
            "answer_logits": self.answer_head(h),
            "answerable_logit": self.answerable_head(h).squeeze(-1),
            "type_logits": self.type_head(h),
        }


class ClipMLPBaseline(nn.Module):
    """Global CLIP image+text embeddings -> MLP (answer, answerability and type heads).
    Same inputs as the global branch of LACE; no patch tokens, no cross-attention."""

    def __init__(self, c: LACEConfig, hidden: int = 1024):
        super().__init__()
        self.c = c
        self.net = nn.Sequential(nn.LayerNorm(2 * c.dim_proj), nn.Linear(2 * c.dim_proj, hidden), nn.GELU(),
                                 nn.Dropout(0.5))
        self.answer_head = nn.Linear(hidden, c.n_answers)
        self.answerable_head = nn.Linear(hidden, 1)
        self.type_head = nn.Linear(hidden, c.n_types)

    def forward(self, patch, img_global, txt_tokens, txt_mask, txt_global):
        h = self.net(torch.cat([img_global, txt_global], -1))
        return {"answer_logits": self.answer_head(h),
                "answerable_logit": self.answerable_head(h).squeeze(-1),
                "type_logits": self.type_head(h)}


def build_head(kind: str, c: LACEConfig) -> nn.Module:
    if kind == "lace":
        return LACEHead(c)
    if kind == "clip_mlp":
        return ClipMLPBaseline(c)
    raise ValueError(kind)


def vqa_loss(out, batch, w_answerable=0.5, w_type=0.2):
    """Soft-target BCE over the answer vocabulary (scaled by vocab size, standard for VQA)
    + masked BCE for answerability + masked CE for answer type."""
    tgt = batch["target"]
    l_ans = F.binary_cross_entropy_with_logits(out["answer_logits"], tgt, reduction="mean") * tgt.size(1)
    loss = l_ans
    logs = {"l_ans": l_ans.item()}
    ans = batch["answerable"]
    m = ans >= 0
    if w_answerable > 0 and m.any():
        l_a = F.binary_cross_entropy_with_logits(out["answerable_logit"][m], ans[m])
        loss = loss + w_answerable * l_a
        logs["l_answerable"] = l_a.item()
    if w_type > 0 and (batch["atype"] >= 0).any():
        l_t = F.cross_entropy(out["type_logits"], batch["atype"], ignore_index=-100)
        loss = loss + w_type * l_t
        logs["l_type"] = l_t.item()
    return loss, logs


class EndToEnd(nn.Module):
    """Full deployable pipeline (raw pixels + token ids -> logits). Used for latency/int8/robustness."""

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, pixel_values, input_ids, attention_mask):
        patch, img_g = self.backbone.encode_image(pixel_values)
        txt, txt_g = self.backbone.encode_text(input_ids, attention_mask)
        return self.head(patch.float(), img_g.float(), txt.float(), attention_mask, txt_g.float())


def count_params(m: nn.Module, trainable_only=False):
    return sum(p.numel() for p in m.parameters() if (p.requires_grad or not trainable_only))
