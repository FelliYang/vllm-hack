# 只是一个工具函数，帮助保存attn_scores
import torch
import os
from vllm.logger import init_logger

logger = init_logger(__name__)

class AttnScoreHelper:
    """单例模式的注意力分数记录助手"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AttnScoreHelper, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        # 确保只初始化一次
        if not AttnScoreHelper._initialized:
            self.global_slot = []
            AttnScoreHelper._initialized = True
            logger.info("Attnention Score Recorder 注册成功")
            self.SAVE_DIR = "/tmp/AttnScores"
    
    def inject_reqid_slot(self, req_id):
        """在全局变量中注入一个req_id slot"""
        self.global_slot.append(req_id)
    
    def clean_reqid_slot(self):
        """清空req_id slot"""
        self.global_slot = []

    def remove_reqid_slot(self, req_id):
        if req_id in self.global_slot:
            self.global_slot.remove(req_id)
        else:
            logger.error("remove slot 不在列表最开头，不符合预期")



    def _get_physical_gpu_id(self, tensor):
        device = tensor.device
        if device.type != "cuda":
            return None  # CPU tensors have no GPU ID

        logical_id = device.index   # e.g., 0 or 1

        visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if visible is None:
            # No remapping → logical id is physical id
            return logical_id

        # CUDA_VISIBLE_DEVICES="3,5" --> ["3", "5"]
        mapping = visible.split(",")
        return int(mapping[logical_id])
    
    def record_attn_score_for_GQA(self, layer_idx, q, k, v, q_heads_num, kv_heads_num, scaling):
        # ------------------- 这里开始插入你的 hack -------------------
        if  q.dim() != 2:
            logger.warning(f"Unexpected Q.size {q.shape}; 当前样本的Attn_Score记录失效")
            return

        seq_len, hidden_size = q.shape

        # ---- shapes ----
        # q: [seq_len, hidden_size]
        # k: [seq_len, hidden_size]
        if seq_len > 1: # prefill
            num_heads = q_heads_num
            num_kv_heads = kv_heads_num
            head_dim = hidden_size // num_heads
            group_size = num_heads // num_kv_heads
            scale = scaling   # <-- 加上注意力缩放

            # dp_rank = getattr(self.parallel_config, "data_parallel_rank", 0)

            # ---- 1. last Q of shape [num_heads, head_dim] ----
            q_last_heads = q[-1].view(num_heads, head_dim)  # [num_heads, head_dim]

            # ---- 2. reshape K: [seq_len, num_kv_heads, head_dim] ----
            K_all = k.view(seq_len, num_kv_heads, head_dim)

            # ---- 3. gather kv-head per q-head ----
            kv_indices = torch.arange(num_heads, device=q.device) // group_size
            K_gathered = K_all[:, kv_indices, :]   # [seq_len, num_heads, head_dim]

            # ---- 4. vectorized q*K (batch matmul) with scaling (bf16 -> fp32) ----
            q32 = q_last_heads.float()                # 上升精度 FP32
            K32 = K_gathered.float()                  # 上升精度 FP32

            # FP32 计算 logits
            logits = (q32.unsqueeze(0) * K32).sum(dim=-1) * scale

            # logits: [seq_len, num_heads]

            # ---- 5. softmax over seq dimension ----
            probs = torch.softmax(logits, dim=0)       # [seq_len, num_heads]

            # ---- 6. final shape: [num_heads, seq_len]
            attn_scores = probs.transpose(0, 1).contiguous()
            attn_scores = attn_scores.to(torch.bfloat16).cpu()

            # ---- 7. save file ----
            cuda_rank = self._get_physical_gpu_id(q)
            save_path = os.path.join(self.SAVE_DIR,  f"CUDA_{cuda_rank}", f"layer_{layer_idx}.pt")

            status = self.record_to_file(attn_scores, save_path)
            
            if status:
                logger.info(
                    f"[Layer {layer_idx}] appended one sample"
                )
            else:
                logger.error("req_id slot 不存在，无法写入attention scores")

    def record_to_file(self, attn_scores, save_file ):
        """记录注意力分数到文件"""
        # 确保目录存在
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        
        if self.global_slot:
            if os.path.exists(save_file):
                obj = torch.load(save_file)
            else:
                obj = []
            obj.append([self.global_slot[0], attn_scores]) # 取slots中的第一个元素，代表当前正在计算的req-id
            torch.save(obj, save_file)
            return 1
        else:
            return 0


# 使用示例
if __name__ == "__main__":
    # 创建实例
    helper1 = AttnScoreHelper()
    helper1.inject_reqid_slot("req_001")
    
    # 再次创建，获得的是同一个实例
    helper2 = AttnScoreHelper()
    print(helper2.global_slot)  # 输出: req_001
    
    # 验证是同一个实例
    print(helper1 is helper2)  # 输出: True