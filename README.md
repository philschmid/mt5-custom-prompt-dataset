# Example for fine-tuning MT5 on a custom prompt dataset

This is an example on how to fine-tune MT5 on a custom prompt dataset. The dataset was created using [BigScience PromptSource library](https://github.com/bigscience-workshop/promptsource).

# Experiments

```bash
python3 scripts/train.py \
  --model_id "google/mt5-small" \
  --dataset_id "philschmid/prompted-germanquad" \
  --epochs 7 \
  --train_batch_size 8 \
  --eval_batch_size 8 \
  --hub_model_id  "mt5-small-prompted-germanquad-1"
```