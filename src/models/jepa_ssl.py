import torch
import torch.nn as nn
import torch.nn.functional as F

def random_block_mask(batch_size, num_tokens, mask_ratio, device):
    """
    Lightweight 1D contiguous block mask over token sequence.
    Returns: (B, num_tokens) bool
    """
    mask = torch.zeros((batch_size, num_tokens), dtype=torch.bool, device=device)
    n_mask = max(1, int(num_tokens * mask_ratio))
    for b in range(batch_size):
        start = torch.randint(0, max(1, num_tokens - n_mask + 1), (1,), device=device).item()
        mask[b, start:start + n_mask] = True
    return mask

class JEPAContinuedPretrain(nn.Module):
    """
    Lightweight JEPA-like objective:
    - teacher: EMA of student
    - student sees masked tokens via bool_masked_pos (shape: B x num_patches)
    - teacher sees full image
    - predictor maps student masked tokens -> teacher tokens
    """
    def __init__(self, backbone, embed_dim: int, ema_momentum: float = 0.996):
        super().__init__()
        self.student = backbone
        import copy
        self.teacher = copy.deepcopy(backbone)
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.m = ema_momentum

    @torch.no_grad()
    def _ema_update(self):
        ms = self.m
        for ps, pt in zip(self.student.model.parameters(), self.teacher.model.parameters()):
            pt.data.mul_(ms).add_(ps.data, alpha=(1.0 - ms))

    def forward(self, pixel_values, mask_ratio=0.6):
        device = pixel_values.device
        B = pixel_values.shape[0]

        # Teacher: full tokens
        with torch.no_grad():
            t_tokens = self.teacher.forward_tokens(pixel_values)  # (B, T, D)

        # IMPORTANT: I-JEPA HF uses T == num_patches (no CLS), e.g. 256 for ViT-H/14@224
        num_tokens = t_tokens.shape[1]

        # bool_masked_pos must be (B, num_patches=num_tokens) :contentReference[oaicite:1]{index=1}
        mask_tokens = random_block_mask(B, num_tokens, mask_ratio, device)

        # Student: masked tokens
        s_tokens = self.student.forward_tokens(pixel_values, bool_masked_pos=mask_tokens)

        # Select masked token vectors
        s_masked = s_tokens[mask_tokens]   # (B*n_mask, D)
        t_masked = t_tokens[mask_tokens]   # (B*n_mask, D)

        pred = self.predictor(s_masked)

        pred = F.normalize(pred, dim=-1)
        t_masked = F.normalize(t_masked, dim=-1)
        loss = 2 - 2 * (pred * t_masked).sum(dim=-1).mean()

        self._ema_update()
        return loss
