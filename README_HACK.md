# vLLM Attention Score Recorder
A lightweight hack tool for recording attention scores during inference in vLLM.

## Hack Code Summary
```bash
git diff v0.11.2 HEAD --stat
 README_HACK.md                            |  27 +++++
 vllm/model_executor/models/ernie45_moe.py |   7 ++
 vllm/model_executor/models/qwen3_moe.py   |   9 ++
 vllm/utils/atten_score_helper.py          | 534 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 vllm/v1/worker/gpu_model_runner.py        |  24 +++-
 5 files changed, 600 insertions(+), 1 deletion(-)
```

## Usage & Limitation
This tool supports recording attention scores in the **prefill** stage. If you want to record attention score in decoded stage, try to transform it into a prefill task by repeating the infer process twice. In the aspect of infer mode, this tool supports single node **DP** and **PP** infer mode currently. TP is currently not supported due to splited attention heads. 

**Note**: 
-  you must apply all these flags `--enforce-eager` | `--max-num-seqs 1` | `--max-num-batched-tokens xxx` | `--no-enable-prefix-caching` to exactly control vllm to execute the HACK code.
- This code only hacks `Qwen3_MOE` and `Ernie45_MOE` forward to save attention scores. It is very easy to hack other models in several lines of code. For example:
```bash
@@ -295,6 +295,7 @@ class Qwen3MoeAttention(nn.Module):
 
         self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
         self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
+        self.layer_idx = extract_layer_index(prefix)
 
     def forward(
         self,
@@ -312,6 +313,14 @@ class Qwen3MoeAttention(nn.Module):
         k_by_head = self.k_norm(k_by_head)
         k = k_by_head.view(k.shape)
         q, k = self.rotary_emb(positions, q, k)
+
+        import os
+        if os.getenv("RECORD_ATTN_SCORE", "False") == "True":
+            # logger.info("记录attention score")
+            from vllm.utils.atten_score_helper import AttnScoreHelper
+            helper = AttnScoreHelper()
+            helper.record_attn_score_for_GQA(self.layer_idx, q, k, v, self.num_heads, self.num_kv_heads, self.scaling)
+
         attn_output = self.attn(q, k, v)
```



### How to Control
You can control the behavior by global environment variable in SHELL. Specifically,
- `RECORD_ATTN_SCORE` is an overall control whether to record attn scores
- `ATTN_CONFIG_FILE` can be used to configure the recorder. For Example:

**Example config file** (`/tmp/attn_config.json`):
```json
{
  "sparse_save": true,
  "sparse_top_k": 4096,
  "save_dir": "/tmp/AttnScores",
  "attn_tail_len": 0
}
```

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `sparse_save` | boolean | Whether to save using sparse tensor format | `true` |
| `sparse_top_k` | integer | Number of top-k values to keep (only when `sparse_save=true`) | `4096` |
| `save_dir` | string | Directory path to save attention scores | `"/tmp/AttnScores"` |
| `attn_tail_len` | integer | Number of tail tokens to record (0=none, 1=last token only) | `0` |

### Example of Ernie (DP)
```bash
ATTN_CONFIG_FILE="/tmp/attn_config.json" RECORD_ATTN_SCORE=True  uv run vllm serve /dev/shm/ERNIE-4.5-21B-A3B/ --max-num-batched-tokens 131072 --max-num-seqs 1 --no-enable-prefix-caching --max-model-len 131072 --enforce-eager  --data-parallel-size 8
```
```bash
# outputs look like
/tmp/AttnScores
├── CUDA_0 # attn scores of the 15 examples dispatched by VLLM to [DP0]
│   ├── layer_0.pt
│   ├── layer_1.pt
│   ├── layer_2.pt
│   ├── ...
├── CUDA_1  # attn scores of another 14 examples dispatched by VLLM to [DP1]
│   ├── layer_0.pt
│   ├── layer_1.pt
│   ├── layer_2.pt
│   ├── ...
├── CUDA_2
├── CUDA_3
├── CUDA_4
├── CUDA_5
├── CUDA_6
├── CUDA_7
```

### Example of Qwen (DP+PP)
```bash
ATTN_CONFIG_FILE="/tmp/attn_config.json" RECORD_ATTN_SCORE=True uv run vllm serve /dev/shm/Qwen3-30B-A3B-Instruct-2507/ --max-num-batched-tokens 131072 --max-num-seqs 1 --no-enable-prefix-caching --max-model-len 131072 --enforce-eager --data-parallel-size 4 --pipeline-parallel-size 2
```

```bash
# outputs look like
/tmp/AttnScores
├── CUDA_0 # attn scores of the 14 examples dispatched by VLLM to [DP0_PP0]
│   ├── layer_0.pt
│   ├── layer_1.pt
│   ├── layer_2.pt
│   ├── ...
│   ├── layer_23.pt
├── CUDA_1 # attn scores of the same 14 examples dispatched by VLLM to [DP0_PP1]
│   ├── layer_24.pt
│   ├── layer_25.pt
│   ├── layer_26.pt
│   ├── ...
├── CUDA_2
├── CUDA_3
├── CUDA_4
├── CUDA_5
├── CUDA_6
├── CUDA_7
```