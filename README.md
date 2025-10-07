# fintun_1

This repository contains a reference script for fine-tuning causal language models with LoRA adapters on a JSONL dataset. The dataset is expected to provide `text` as the prompt input and `target` as the desired output response.

## Usage

```
python lora_finetune.py \
  --model-name-or-path meta-llama/Llama-2-7b-hf \
  --train-file path/to/train.jsonl \
  --validation-file path/to/valid.jsonl \
  --output-dir lora-llama2 \
  --max-seq-length 1024 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --num-train-epochs 3 \
  --learning-rate 2e-4 \
  --template "### Input\n{text}\n\n### Response\n{target}"
```

The script supports optional gradient checkpointing and 4-bit quantization-aware training (via bitsandbytes) to reduce memory usage.
