## Usage & Limitation
This tool supports to record attention scores in the **prefill** stage. If you want to record attention score in decoded stage, try to transform it into a prefill task by repeating the infer process twice. In the aspect of infer mode, this tool supports single node **DP** and **PP** infer mode currently. TP is currently not supported due to splited attention heads. 

**Note**: you must apply all these flags `--enforce-eager` | `--max-num-seqs 1` | `--max-num-batched-tokens xxx` | `--no-enable-prefix-caching` to exactly control vllm to execute the HACK code.

### How to Control
You can control the behavior by global environment variable in SHELL. Specifially,
- `RECORD_ATTN_SCORE` is an overall control where to record attn scores
- `ATTN_CONFIG_FILE` can be used to config the hacker. For Example:
```bash
  {
    "sparse_save": true, # where to save by sparse tensor to save memory
    "sparse_top_k": 4096, # sparse strategy, only valid when `sparse_save` is true
    "save_dir": "/tmp/sparse", # Attn score file save_dir
    "attn_tail_len": 0, # How num Tail Tokens' Attn Score to save. 0 refs to no tokens, 1 refs to last token, >1 refs to multiple tokens.
  }
```
Remind to delete these notes when passing the config into json file.

### Example of Ernie
```bash
ATTN_CONFIG_FILE="/tmp/attn_config.json" RECORD_ATTN_SCORE=True  uv run vllm serve /dev/shm/ERNIE-4.5-21B-A3B-MidTrain-exp6/ --max-num-batched-tokens 131072 --max-num-seqs 1 --no-enable-prefix-caching --max-model-len 131072 --enforce-eager  --data-parallel-size 8
```
### Example of Qwen
```bash
ATTN_CONFIG_FILE="/tmp/attn_config.json" RECORD_ATTN_SCORE=True uv run vllm serve /dev/shm/Qwen3-30B-A3B-Instruct-2507/ --max-num-batched-tokens 131072 --max-num-seqs 1 --no-enable-prefix-caching --max-model-len 131072 --enforce-eager --data-parallel-size 4 --pipeline-parallel-size 2
```