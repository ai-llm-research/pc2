# PC<sup>2</sup>: Politically Controversial Content Generation via Jailbreaking Attacks on GPT-based Text-to-Image Models

🎉 This paper has been accepted to the **33rd ACM Conference on Computer and Communications Security (CCS 2026).** 🎉

This is the official implementation for paper: [PC<sup>2</sup>: Politically Controversial Content Generation via Jailbreaking Attacks on GPT-based Text-to-Image Models](https://arxiv.org/abs/2601.05150)

> ⚠️ **Content Warning.** This repository studies attacks that generate politically controversial content. As a result, it contains and produces material — prompts, references to political figures and events, and other examples — that some readers may find offensive, disturbing, or otherwise uncomfortable. This content is included **strictly for research and safety-evaluation purposes** and does not reflect the views of the authors. Please proceed with discretion.

## Prerequisites

- Python 3.10.18 (recommended)
- A virtual environment (optional but recommended)
- Required Python packages installed (see `requirements.txt`)
- An OpenAI API key exported as `OPENAI_API_KEY` (used by `src/prepare_embeddings.py`, `src/main.py`, and the defense scripts):

  ```bash
  export OPENAI_API_KEY=<your key>
  ```

- (For defense) install lid.176.bin from fasttext (`https://fasttext.cc/docs/en/language-identification.html`)

## Pre-computing Embeddings

Before running the main script, you **must** pre-calculate the embeddings used by the attack pipeline.

Run:

```bash
python src/prepare_embeddings.py
```

This step computes and stores embedding representations that are later consumed by `src/main.py`. Skipping this step will result in runtime errors or incomplete results.


## Running the Main Script

The main entry point for PC<sup>2</sup> is `src/main.py`.

### Command Format

```bash
python src/main.py -p "<target prompt>" -o <output_file>.json
```

### Arguments

- `-p, --prompt`  
  The target prompt to be attacked. The prompt MUST contain a prefix to draw an image (e.g., "Create a realistic image for").
- `-o, --output`  
  Path to the output JSON file where results will be saved.

### Example

```bash
python src/main.py -p "Create a realistic image for <POLITICAL_FIGURE> holding <POLITICAL_OBJECT>" -o test_output.json
```

This command runs the jailbreak-based generation pipeline using the provided target prompt and writes the results to `test_output.json`. You may configure the language selection percentile with `--percentile` argument (default = 50).

### Output Format

The output JSON file contains the final multilingual adversarial prompt and the languages selected for each component:

```json
{
    "final_prompt": "<MULTILINGUAL_PROMPT>",
    "selected_languages": [
        "Latvian",
        "Dutch",
        "Croatian"
    ]
}
```

- `final_prompt` — the generated multilingual prompt to be sent to the target model.
- `selected_languages` — the languages chosen for the base prompt and each detected PCC component during final prompt selection.

> **Note on API cost.** Both scripts make paid OpenAI API calls. With the default models (`gpt-4o-2024-05-13`, `text-embedding-3-large`), one measured
> run cost about **$0.29** for the one-time `prepare_embeddings.py` setup and about **$1.16** per target prompt.

## Running the Defense Script

Please refer to the `README.md` files in `defense/relevant_language` and `defense/llavaguard` for instructions on running the defense scripts.

## Disclaimer

This project is intended solely to **analyze and stress-test** the safety mechanisms of generative models. It is not designed to produce, promote, or disseminate harmful, misleading, abusive, or otherwise unsafe content.

Materials that may be sensitive, unsafe, or otherwise unsuitable for public release have been omitted from the public release. Specifically, the omitted materials include the complete set of sensitive prompts, generated images, and generated adversarial prompts used in the experiments. In particular, `data/dataset.txt` ships only as a small **format template**: it contains an illustrative subset of the prompts (20 of the 240 used in the experiments), and within that subset the names of the real public figures targeted are replaced with the `<POLITICAL_FIGURE>` placeholder. The file therefore documents the input format without releasing the full prompt set or naming individuals. Verified researchers may request access to these materials by emailing [ai.llm.researcher@gmail.com] with the subject line "[PC2] Material Access Request" and including their name, affiliation, research purpose, and intended use of the requested materials. Requests will be considered on a case-by-case basis and are subject to appropriate safeguards and intended research use.

The authors assume no responsibility for misuse of this code.

## Citation

If you use this repository in your research, please cite our paper.

```
@article{choi2026pc2,
  title={PC2: Politically Controversial Content Generation via Jailbreaking Attacks on GPT-based Text-to-Image Models},
  author={Choi, Wonwoo and Seo, Minjae and Song, Minkyoo and Heo, Hwanjo and Shin, Seungwon and You, Myoungsung},
  journal={arXiv preprint arXiv:2601.05150},
  year={2026}
}
```
