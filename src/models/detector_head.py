#Cross-Attention

import torch
import torch.nn as nn

class OOCDetectorHead(nn.Module):
    """
    Sử dụng Cross-Attention để tìm sự tương quan giữa Object (1D) và Context (Sequence 2D).
    Output: Logit cho phân loại OOC.
    """
    def __init__(self, dim: int, num_heads: int = 8, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        
        # Mạng Cross-Attention: batch_first=True rất quan trọng vì input là [Batch, Seq, Dim]
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

        # Mạng MLP quyết định (Classification Head)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden), # Nối Object gốc và Object đã "hiểu" ngữ cảnh
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, obj_emb, ctx_emb):
        """
        obj_emb: [Batch, Dim] (Từ hàm encode_object)
        ctx_emb: [Batch, Seq_Len, Dim] (Từ hàm encode_context)
        """
        # 1. Ép kiểu Object thành dạng Sequence có độ dài = 1
        query = obj_emb.unsqueeze(1)  # Shape: [Batch, 1, Dim]
        key_value = ctx_emb           # Shape: [Batch, Seq_Len, Dim]
        
        # 2. Object "nhìn" vào Context
        attn_out, _ = self.cross_attn(query=query, key=key_value, value=key_value)
        
        # 3. Residual connection & LayerNorm (Chuẩn hóa)
        obj_context_aware = self.norm(query + attn_out).squeeze(1) # Trở lại [Batch, Dim]
        
        # 4. Nối đặc trưng vật thể nguyên bản và đặc trưng đã đối chiếu với nền
        x = torch.cat([obj_emb, obj_context_aware], dim=-1)
        
        # 5. Phân loại
        return self.mlp(x).squeeze(-1)