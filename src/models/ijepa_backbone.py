#bỏ @torch.no_grad() 
#đổi tên hàm: encode_object và encode_context 

import torch
from transformers import AutoImageProcessor, AutoModel, AutoConfig

class IJepaBackbone(torch.nn.Module):
    def __init__(self, model_id: str, attn_implementation: str = "sdpa",
                 gradient_checkpointing: bool = False,
                 use_mask_token: bool = False):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_id)

        # Kích hoạt mask token nếu sử dụng bool_masked_pos
        config = AutoConfig.from_pretrained(model_id)
        
        self.model = AutoModel.from_pretrained(
            model_id,
            config=config,
            attn_implementation=attn_implementation,
            use_mask_token=use_mask_token,
        )

        if gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

    def encode_object(self, pixel_values: torch.Tensor):
        """
        Dành cho ảnh Đối tượng (Object Crop).
        Sử dụng Global Average Pooling (GAP) để tạo ra vector đại diện duy nhất.
        
        Input: pixel_values shape (B, 3, H, W)
        Output: (B, D) - Vector 1D
        """
        out = self.model(pixel_values=pixel_values)
        # Lấy trung bình cộng của tất cả patch tokens (dim=1)
        z = out.last_hidden_state.mean(dim=1)
        return z

    def encode_context(self, pixel_values: torch.Tensor, bool_masked_pos=None):
        """
        Dành cho ảnh Bối cảnh (Context/Scene).
        Trả về trực tiếp chuỗi Patch Tokens để giữ lại thông tin không gian chi tiết,
        phục vụ cho việc tính toán Cross-Attention.
        
        Input: pixel_values shape (B, 3, H, W)
        Output: (B, Seq_Len, D) - Chuỗi vector 2D
        """
        out = self.model(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos)
        # Không pooling, trả về toàn bộ seq_len
        return out.last_hidden_state

    # -------------------------------------------------------------------
    # Các hàm dưới đây được giữ lại để đảm bảo tính tương thích ngược (Backward Compatibility)
    # với các file code cũ chưa kịp cập nhật tên hàm mới.
    # -------------------------------------------------------------------
    #def encode(self, pixel_values: torch.Tensor):
    #    return self.encode_object(pixel_values)

    #def forward_tokens(self, pixel_values: torch.Tensor, bool_masked_pos=None):
    #    return self.encode_context(pixel_values, bool_masked_pos)