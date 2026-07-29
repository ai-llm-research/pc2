# Relevant Language-based Defense (Prototype)

## Usage

```bash
export OPENAI_API_KEY=<your key>
python src/defense/relevant_language/infer.py \
  -i <input.json (with "final_prompt" as target prompt)> \
  -o <output.json (with "final_prompt" as result prompt)>
```