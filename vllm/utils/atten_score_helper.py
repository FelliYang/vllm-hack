# 只是一个工具函数，帮助保存attn_scores
import torch
import os
from vllm.logger import init_logger
import json

logger = init_logger(__name__)

from dataclasses import dataclass, fields
from typing import Optional

@dataclass
class AttnScoreConfig:
    save_dir: str = "/tmp/AttnScores"
    sparse_save: bool = True
    sparse_top_k: int = 4096
    attn_tail_len: Optional[int] = None  # None = 不保存

    def __post_init__(self):
        self.sparse_save = self._parse_bool(self.sparse_save)
        self.sparse_top_k = self._parse_int(self.sparse_top_k, 4096)
        self.attn_tail_len = self._parse_int(self.attn_tail_len, None)

    @staticmethod
    def _parse_bool(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "on")
        return bool(v)

    @staticmethod
    def _parse_int(v, default):
        if v is None:
            return default
        try:
            return int(v)
        except Exception:
            return default

    @classmethod
    def from_file(cls, path: str) -> "AttnScoreConfig":
        if not os.path.exists(path):
            logger.error(f"attentino config file {path} not exist, failed to load config {path}")
            return cls()
        try:
            with open(path, "r") as f:
                data = json.load(f)
            valid_fields = {f.name for f in fields(cls)}
            filtered = {k: v for k, v in data.items() if k in valid_fields}
            return cls(**filtered)
        except Exception as e:
            logger.error(f"failed to load config {path}: {e}")
            return cls()


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
        if AttnScoreHelper._initialized:
            return

        self.global_slot = []

        # config 文件路径（仍然可用 env 控制）
        self.CONFIG_FILE = os.getenv(
            "ATTN_CONFIG_FILE", "/tmp/attn_config.json"
        )

        # 加载配置对象
        self.config = AttnScoreConfig.from_file(self.CONFIG_FILE)

        AttnScoreHelper._initialized = True
        logger.info("Attention Score Recorder 注册成功")


    def inject_reqid_slot(self, req_id):
        """在全局变量中注入一个req_id slot"""
        if req_id not in self.global_slot:
            self.global_slot.append(req_id)
            return 1 #  inject success
        else:
            return 0
    
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
        # config = self.read_config()
        # 每次都会重新读attn score file
        self.config = AttnScoreConfig.from_file(self.CONFIG_FILE)
        
        if not self.config.attn_tail_len or self.config.attn_tail_len < 1:
            return 

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

            # ---- 0. 确保 attn_tail_len 不超过 seq_len ----
            actual_tail_len = min(self.config.attn_tail_len, seq_len)

            # ---- 1. last actual_tail_len Q tokens of shape [actual_tail_len, num_heads, head_dim] ----
            q_last_heads = q[-actual_tail_len:].view(actual_tail_len, num_heads, head_dim)  
            # [actual_tail_len, num_heads, head_dim]

            # ---- 2. reshape K: [seq_len, num_kv_heads, head_dim] ----
            K_all = k.view(seq_len, num_kv_heads, head_dim)

            # ---- 3. gather kv-head per q-head ----
            kv_indices = torch.arange(num_heads, device=q.device) // group_size
            K_gathered = K_all[:, kv_indices, :]   # [seq_len, num_heads, head_dim]

            # ---- 4. vectorized q*K (batch matmul) with scaling (bf16 -> fp32) ----
            q32 = q_last_heads.float()                # [actual_tail_len, num_heads, head_dim]
            K32 = K_gathered.float()                  # [seq_len, num_heads, head_dim]

            # FP32 计算 logits
            logits = torch.einsum('thd,shd->tsh', q32, K32) * scale
            # logits: [actual_tail_len, seq_len, num_heads]

            # ---- 4.5. apply causal mask ----
            # q 的位置是 [seq_len - actual_tail_len, ..., seq_len - 1]
            q_positions = torch.arange(seq_len - actual_tail_len, seq_len, device=q.device)  # [actual_tail_len]
            k_positions = torch.arange(seq_len, device=q.device)  # [seq_len]

            # 创建 causal mask: q_pos >= k_pos 的位置才能看到
            causal_mask = q_positions.unsqueeze(1) >= k_positions.unsqueeze(0)  # [actual_tail_len, seq_len]

            # 将不能看到的位置设为 -inf
            logits = logits.masked_fill(~causal_mask.unsqueeze(-1), float('-inf'))  # broadcast to [actual_tail_len, seq_len, num_heads]

            # ---- 5. softmax over seq dimension ----
            probs = torch.softmax(logits, dim=1)       # [actual_tail_len, seq_len, num_heads]

            # ---- 6. final shape: [actual_tail_len, num_heads, seq_len]
            attn_scores = probs.transpose(1, 2).contiguous()  # [actual_tail_len, num_heads, seq_len]

            # ---- 7. save file ----
            cuda_rank = self._get_physical_gpu_id(q)
            save_path = os.path.join(self.config.save_dir,  f"CUDA_{cuda_rank}", f"layer_{layer_idx}.pt")

            status = self.record_to_file(attn_scores, save_path)
            
            if status:
                logger.info(
                    f"[Layer {layer_idx}] appended one sample to {save_path}, seq len is {seq_len}"
                )
            else:
                logger.error("req_id slot 不存在，无法写入attention scores")

    def sparsify_attn_scores(self, attn_scores, top_k):
        """
        将注意力分数稀疏化，只保留每个 head 在 seq 维度上的 top-k 值
        
        Args:
            attn_scores: [actual_tail_len, num_heads, seq_len] 或 [num_heads, seq_len]
            top_k: 保留的 top-k 个值
        
        Returns:
            sparse_dict: {
                'shape': original shape,
                'values': [actual_tail_len, num_heads, top_k] 的值,
                'indices': [actual_tail_len, num_heads, top_k] 的索引
            }
        """
        original_shape = attn_scores.shape
        
        # 处理两种可能的输入形状
        if len(original_shape) == 2:  # [num_heads, seq_len]
            attn_scores = attn_scores.unsqueeze(0)  # 变成 [1, num_heads, seq_len]
        
        actual_tail_len, num_heads, seq_len = attn_scores.shape
        effective_k = min(top_k, seq_len)  # 确保 k 不超过 seq_len
        
        # 在 seq 维度上取 top-k
        # topk 返回 (values, indices)
        top_values, top_indices = torch.topk(attn_scores, k=effective_k, dim=-1, largest=True, sorted=True)
        # top_values: [actual_tail_len, num_heads, effective_k]
        # top_indices: [actual_tail_len, num_heads, effective_k]
        
        # 计算压缩后的权重保留率（在 seq 维度上求和）
        sparse_row_sums = top_values.sum(dim=-1)  # [actual_tail_len, num_heads]
        weight_loss = 1.0 - sparse_row_sums  # 与 1.0 的距离

        # 统计最大丢失率及其位置
        max_weight_loss = weight_loss.max().item()
        mean_weight_loss = weight_loss.mean().item()
        
        # 找到最大丢失率的位置 [token_idx, head_idx]
        max_loss_flat_idx = weight_loss.view(-1).argmax().item()
        max_loss_token_idx = max_loss_flat_idx // num_heads
        max_loss_head_idx = max_loss_flat_idx % num_heads
        
        logger.info(
            f"[Sparsify] top_k={effective_k}/{seq_len} | "
            f"Max weight loss: {max_weight_loss:.4f} ({max_weight_loss*100:.2f}%) "
            f"at token_idx={max_loss_token_idx}, head_idx={max_loss_head_idx} | "
            f"Mean weight loss: {mean_weight_loss:.4f} ({mean_weight_loss*100:.2f}%)"
        )
        
        sparse_dict = {
            'shape': original_shape,
            'values': top_values.to(torch.bfloat16).cpu(),
            'indices': top_indices.to(torch.int32).cpu(),  # 用 int32 节省空间
            'top_k': effective_k
        }
        
        return sparse_dict


    def densify_attn_scores(self, sparse_dict):
        """
        将稀疏化的注意力分数还原为稠密格式
        
        Args:
            sparse_dict: sparsify_attn_scores 返回的字典
        
        Returns:
            attn_scores: 还原后的稠密张量，shape 为 sparse_dict['shape']
        """
        original_shape = sparse_dict['shape']
        top_values = sparse_dict['values']
        top_indices = sparse_dict['indices']
        
        # 处理两种可能的原始形状
        if len(original_shape) == 2:  # [num_heads, seq_len]
            num_heads, seq_len = original_shape
            actual_tail_len = 1
        else:  # [actual_tail_len, num_heads, seq_len]
            actual_tail_len, num_heads, seq_len = original_shape
        
        # 创建全零张量
        attn_scores = torch.zeros(actual_tail_len, num_heads, seq_len, dtype=torch.bfloat16)
        
        # 使用 scatter_ 将 top-k 值放回原位置
        # attn_scores[t, h, indices[t, h, k]] = values[t, h, k]
        attn_scores.scatter_(
            dim=-1,  # 在 seq_len 维度上 scatter
            index=top_indices.long(),  # 需要 long 类型
            src=top_values
        )
        
        # 恢复原始形状
        if len(original_shape) == 2:
            attn_scores = attn_scores.squeeze(0)  # [num_heads, seq_len]
        
        return attn_scores

    def record_to_file(self, attn_scores, save_file ):
        """记录注意力分数到文件"""
        # 确保目录存在=
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        if self.global_slot:
            if os.path.exists(save_file):
                obj = torch.load(save_file)
            else:
                obj = []
            # 修改attn_scores save逻辑
            if self.config.sparse_save:
                # 稀疏化保存
                sparse_data = self.sparsify_attn_scores(attn_scores, self.config.sparse_top_k)
                # 取slots中的第一个元素，代表当前正在计算的req-id
                obj.append([self.global_slot[0], sparse_data])
            else:  # 全量save
                attn_scores = attn_scores.to(torch.bfloat16).cpu()
                obj.append([self.global_slot[0], attn_scores])
                
            torch.save(obj, save_file)
            return 1
        else:
            return 0



def test_one_instance():
    # 创建实例
    helper1 = AttnScoreHelper()
    helper1.inject_reqid_slot("req_001")
    
    # 再次创建，获得的是同一个实例
    helper2 = AttnScoreHelper()
    print(helper2.global_slot)  # 输出: req_001
    
    # 验证是同一个实例
    print(helper1 is helper2)  # 输出: True

def test_sparse_dense_alignment():
    """测试从文件读取的 dense 和 sparse 格式 attention scores 的对齐性"""
    print("\n=== 测试 Dense 和 Sparse 格式对齐性 ===")
    
    helper = AttnScoreHelper()
    
    # 文件路径配置
    dense_file_path = "/tmp/debug/dense/CUDA_0/layer_0.pt"
    sparse_file_path = "/tmp/debug/sparse/CUDA_0/layer_0.pt"
    # dense_file_path = "/tmp/dense/CUDA_1/layer_0.pt"
    # sparse_file_path = "/tmp/sparse/CUDA_0/layer_0.pt"
    # all_tensors = torch.load(dense_file_path)
    # 读取数据
    # print("\n[1] 读取数据...")
    # for i in range(8):
    #     dense_file_path = f"/tmp/dense/CUDA_{i}/layer_0.pt"
    #     all_tensors = torch.load(dense_file_path)
    #     print(f"  Dense shape: {all_tensors[0][1].shape}")

    attn_scores_dense = torch.load(dense_file_path)[0][1]
    sparse_dict = torch.load(sparse_file_path)[0][1]
    
    print(f"  Dense shape: {attn_scores_dense.shape}")
    print(f"  Sparse shape: {sparse_dict['shape']}")
    # print(f"  Sparse indices shape: {sparse_dict['indices'].shape}")
    
    # 验证形状
    # assert attn_scores_dense.shape == tuple(sparse_dict['shape']), \
    #     f"形状不匹配: {attn_scores_dense.shape} vs {sparse_dict['shape']}"
    
    actual_tail_len, num_heads, seq_len = attn_scores_dense.shape
    actual_tail_len = 1
    top_k = sparse_dict['values'].shape[-1]
    
    # 验证 sparse indices 位置的对齐性
    print("\n[2] 验证 Sparse Indices 对齐性...")
    
    max_diff = 0.0
    total_diff = 0.0
    total_points = 0
    
    for t in range(actual_tail_len):
        for h in range(num_heads):
            # 获取 sparse 的 indices 和 values
            sparse_indices = sparse_dict['indices'][t, h]
            sparse_values = sparse_dict['values'][t, h]
            
            # 从 dense 中提取对应位置的值
            dense_values = attn_scores_dense[t, h].gather(0, sparse_indices)
            
            # 计算差异
            diff = (dense_values - sparse_values).abs()
            max_diff = max(max_diff, diff.max().item())
            total_diff += diff.sum().item()
            total_points += top_k
    
    mean_diff = total_diff / total_points
    
    print(f"  比较点数: {total_points:,}")
    print(f"  最大差异: {max_diff:.6e}")
    print(f"  平均差异: {mean_diff:.6e}")
    
    # 最终验证
    print("\n[3] 验证结果:")
    tolerance = 1e-3
    
    if max_diff < tolerance:
        print(f"  ✅ 对齐验证通过 (max_diff={max_diff:.6e} < {tolerance})")
    else:
        print(f"  ❌ 对齐验证失败 (max_diff={max_diff:.6e} >= {tolerance})")
        assert False, f"对齐差异超过容忍度: {max_diff:.6e}"
    
    print("\n✅ 测试通过!")

def test_sparse_save():
    """测试稀疏保存和还原功能 - 3D large scale"""
    print("\n=== 测试稀疏保存和还原 (3D Large Scale) ===")
    
    helper = AttnScoreHelper()
    
    # 创建大规模测试数据
    actual_tail_len = 4
    num_heads = 2
    seq_len = 10000
    top_k = 4096
    
    print(f"测试参数:")
    print(f"  Shape: [{actual_tail_len}, {num_heads}, {seq_len}]")
    print(f"  Top-K: {top_k}")
    
    # 生成随机 logits 并用 softmax 归一化为注意力分数
    print("\n[0] 生成注意力分数...")
    logits = torch.randn(actual_tail_len, num_heads, seq_len, dtype=torch.float32)
    attn_scores_original = torch.softmax(logits, dim=-1).to(torch.bfloat16)
    
    # 验证 softmax 归一化
    row_sums = attn_scores_original.sum(dim=-1)
    print(f"  Softmax 归一化检查 (每行和应为1.0):")
    print(f"    最小值: {row_sums.min().item():.6f}")
    print(f"    最大值: {row_sums.max().item():.6f}")
    print(f"    平均值: {row_sums.mean().item():.6f}")
    
    # 稀疏化
    print("\n[1] 稀疏化...")
    sparse_dict = helper.sparsify_attn_scores(attn_scores_original, top_k)
    print(f"  原始形状: {attn_scores_original.shape}")
    print(f"  稀疏后 values 形状: {sparse_dict['values'].shape}")
    print(f"  稀疏后 indices 形状: {sparse_dict['indices'].shape}")
    
    # 还原
    print("\n[2] 还原...")
    attn_scores_restored = helper.densify_attn_scores(sparse_dict)
    print(f"  还原后形状: {attn_scores_restored.shape}")
    
    # 检查还原后的归一化情况
    restored_row_sums = attn_scores_restored.sum(dim=-1)
    print(f"  还原后每行和 (不再为1.0，因为丢弃了部分权重):")
    print(f"    最小值: {restored_row_sums.min().item():.6f}")
    print(f"    最大值: {restored_row_sums.max().item():.6f}")
    print(f"    平均值: {restored_row_sums.mean().item():.6f}")
    
    # 统计数值差异
    print("\n[3] 数值差异统计:")
    
    # 计算保留的 top-k 位置的差异
    total_preserved = 0
    max_diff_preserved = 0.0
    mean_diff_preserved = 0.0
    
    for t in range(actual_tail_len):
        for h in range(num_heads):
            # 获取原始 top-k 的索引和值
            original_topk_values, original_topk_indices = torch.topk(
                attn_scores_original[t, h], k=top_k, largest=True
            )
            # 从还原tensor中取出对应位置的值
            restored_topk_values = attn_scores_restored[t, h].gather(0, original_topk_indices)
            
            # 计算差异
            diff = (original_topk_values - restored_topk_values).abs()
            max_diff_preserved = max(max_diff_preserved, diff.max().item())
            mean_diff_preserved += diff.sum().item()
            total_preserved += top_k
    
    mean_diff_preserved /= total_preserved
    
    # 计算非 top-k 位置的值（应该全为0，但原始值不为0）
    mask = torch.ones_like(attn_scores_original, dtype=torch.bool)
    for t in range(actual_tail_len):
        for h in range(num_heads):
            _, top_indices = torch.topk(attn_scores_original[t, h], k=top_k)
            mask[t, h, top_indices] = False
    
    non_topk_original = attn_scores_original[mask]
    non_topk_restored = attn_scores_restored[mask]
    
    print(f"  Top-{top_k} 位置的差异:")
    print(f"    最大差异: {max_diff_preserved:.6e}")
    print(f"    平均差异: {mean_diff_preserved:.6e}")
    
    print(f"  非 Top-{top_k} 位置:")
    print(f"    原始值统计 (被丢弃的权重):")
    print(f"      最大值: {non_topk_original.max().item():.6e}")
    print(f"      平均值: {non_topk_original.mean().item():.6e}")
    print(f"      总和: {non_topk_original.sum().item():.6f}")
    print(f"    还原后值 (应为0):")
    print(f"      最大值: {non_topk_restored.abs().max().item():.6e}")
    print(f"      平均值: {non_topk_restored.abs().mean().item():.6e}")
    
    # 整体差异
    total_diff = (attn_scores_original - attn_scores_restored).abs()
    overall_max_diff = total_diff.max().item()
    overall_mean_diff = total_diff.mean().item()
    
    print(f"  整体差异:")
    print(f"    最大差异: {overall_max_diff:.6e}")
    print(f"    平均差异: {overall_mean_diff:.6e}")
    
    # 权重保留率
    total_weight_original = attn_scores_original.sum().item()
    total_weight_restored = attn_scores_restored.sum().item()
    weight_retention = (total_weight_restored / total_weight_original) * 100
    
    print(f"  权重保留统计:")
    print(f"    原始权重总和: {total_weight_original:.2f}")
    print(f"    还原权重总和: {total_weight_restored:.2f}")
    print(f"    权重保留率: {weight_retention:.2f}%")
    
    # 压缩率统计
    print("\n[4] 存储压缩统计:")
    original_size = attn_scores_original.numel() * 2  # bfloat16 = 2 bytes
    sparse_size = (sparse_dict['values'].numel() * 2 +  # values: bfloat16
                   sparse_dict['indices'].numel() * 4)  # indices: int32
    compression_ratio = original_size / sparse_size
    saved_percent = (1 - sparse_size / original_size) * 100
    
    print(f"  原始大小: {original_size / 1024 / 1024:.2f} MB")
    print(f"  稀疏后大小: {sparse_size / 1024 / 1024:.2f} MB")
    print(f"  压缩率: {compression_ratio:.2f}x")
    print(f"  节省空间: {saved_percent:.2f}%")
    
    # 验证
    print("\n[5] 验证:")
    assert max_diff_preserved < 1e-3, f"❌ Top-k 还原误差过大: {max_diff_preserved}"
    assert non_topk_restored.abs().max() < 1e-6, f"❌ 非 Top-k 位置应为0"
    print("  ✅ Top-k 位置还原正确")
    print("  ✅ 非 Top-k 位置正确置零")
    print(f"  ✅ 保留了 {weight_retention:.2f}% 的注意力权重")
    print("\n✅ 所有测试通过!")


if __name__ == "__main__":
    test_sparse_dense_alignment()

    

    