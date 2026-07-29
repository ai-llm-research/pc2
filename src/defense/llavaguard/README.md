# LlavaGuard v1.2-7B

## Model
- **HuggingFace**: `AIML-TUDA/LlavaGuard-v1.2-7B-OV-hf`
- **Architecture**: LLaVA-OneVision Qwen2-7B (text generation)
- **Output**: Structured JSON verdict `{"rating": "Safe"|"Unsafe", "category": ..., "rationale": ...}`
- **Policies**: `default` (O1–O10), `default_psc` (O1–O11 combined)

## Usage

```bash
python infer.py \
  --manifest <manifest.json (refer to example manifest.json)> \
  --policy [default|default_psc] \
  --out-csv results/output.csv \
  --out-json results/output.json
```