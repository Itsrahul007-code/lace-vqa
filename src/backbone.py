"""Frozen CLIP backbone: image patch tokens + global embeddings, question tokens + global embedding."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_transform(image_size: int):
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(CLIP_MEAN, CLIP_STD),
    ])


def load_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    img.draft("RGB", (1024, 1024))  # fast JPEG decode for very large phone photos
    return img.convert("RGB")


class ClipBackbone(nn.Module):
    """Wraps a Hugging Face CLIPModel. All weights are frozen.

    encode_image -> (patch tokens [B, G*G, Dv] from the penultimate layer, pooled to a GxG grid,
                     L2-normalised global image embedding [B, P])
    encode_text  -> (token states [B, L, Dt], L2-normalised global text embedding [B, P])
    """

    def __init__(self, name_or_path: str = "openai/clip-vit-base-patch16", pool_grid: int = 7,
                 feature_layer: int = -2):
        super().__init__()
        from transformers import AutoTokenizer, CLIPModel
        self.clip = CLIPModel.from_pretrained(name_or_path)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        except Exception as e:  # some CLIP variants (e.g. TinyCLIP) share OpenAI's BPE vocabulary
            print(f"[backbone] tokenizer from {name_or_path} failed ({e}); using OpenAI CLIP tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.name = name_or_path
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad_(False)
        vc = self.clip.config.vision_config
        self.image_size = vc.image_size
        self.grid = vc.image_size // vc.patch_size
        self.pool_grid = min(pool_grid, self.grid) if pool_grid > 0 else self.grid
        self.feature_layer = feature_layer
        self.dim_vision = vc.hidden_size
        self.dim_text = self.clip.config.text_config.hidden_size
        self.dim_proj = self.clip.config.projection_dim
        self.transform = build_transform(self.image_size)

    @property
    def n_img_tokens(self):
        return self.pool_grid * self.pool_grid

    @torch.no_grad()
    def encode_image(self, pixel_values: torch.Tensor):
        vm = self.clip.vision_model
        out = vm(pixel_values=pixel_values, output_hidden_states=True)
        tokens = out.hidden_states[self.feature_layer][:, 1:, :]  # drop CLS
        B, N, D = tokens.shape
        if self.pool_grid != self.grid:
            g = tokens.transpose(1, 2).reshape(B, D, self.grid, self.grid)
            g = F.adaptive_avg_pool2d(g, self.pool_grid)
            tokens = g.flatten(2).transpose(1, 2)
        glob = self.clip.visual_projection(out.pooler_output)
        return tokens, F.normalize(glob, dim=-1)

    @torch.no_grad()
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
        glob = self.clip.text_projection(out.pooler_output)
        return out.last_hidden_state, F.normalize(glob, dim=-1)

    def tokenize(self, questions, max_len: int = 32, device=None):
        enc = self.tokenizer(list(questions), padding="max_length", truncation=True,
                             max_length=max_len, return_tensors="pt")
        if device is not None:
            enc = {k: v.to(device) for k, v in enc.items()}
        return enc
