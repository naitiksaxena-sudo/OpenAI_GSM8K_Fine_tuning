# GSM8K: QLoRA Fine-Tuning with a Verifier and Best-of-N

A practice project on grade-school math (GSM8K): fine-tune a small language model, train a second model to judge its answers, and test whether picking the best of several sampled answers beats a single greedy answer.

## What was built

1. **Generator.** Phi-3.5-mini-instruct fine-tuned with QLoRA (4-bit, LoRA rank 16, alpha 16, all seven attention and MLP projection modules) on the GSM8K training set. One epoch, batch size 8, learning rate 2e-4.
2. **Candidate sampling.** For 100 shuffled GSM8K training questions, 4 answers were sampled (temperature 0.7, top-p 0.9, up to 256 tokens) from each of two checkpoints, checkpoint-935 and checkpoint-500. That gives 800 records. Each answer was labeled correct if its final `#### N` answer matched the gold answer.
3. **Verifier.** Llama-3.2-1B-Instruct fine-tuned with LoRA (rank 16, alpha 32) as a binary classifier of whether a candidate solution is correct. 4-bit, 3 epochs, learning rate 2e-4, inputs cut to 512 tokens, stratified 90/10 split, best epoch chosen by F1.
4. **Calculator check.** Every `<<expression=result>>` step in a candidate is recomputed with a safe arithmetic evaluator. Candidates with a wrong step are dropped before the verifier scores the rest. If none survive, all candidates are kept.
5. **Best-of-N.** Sample 4 answers, apply the calculator check, and let the verifier pick the highest-scoring one. Compared against a single greedy answer.
6. **RFT and DPO.** Rejection-sampling fine-tuning (full GSM8K training set plus the model's own correct answers from step 2) and DPO (correct vs incorrect answer pairs from step 2, beta 0.1, learning rate 5e-6) were trained on top of the generator.

## Results

Evaluated on 50 GSM8K test questions (the test split shuffled with seed 42, first 50 taken). Generator: [fill in: base, RFT or DPO adapter].

| Method | Accuracy |
|---|---|
| Greedy baseline | 72% (36/50) |
| Best-of-4 + calculator + verifier | 72% (36/50) |

Best-of-N with the verifier gave no improvement over greedy.

Earlier numbers from this project (44% greedy rising to 58% with best-of-4, calculator, RFT and DPO) were measured with a partly loaded adapter and were discarded. The RFT and DPO evaluations were affected, so no valid result is reported for them.

The verifier reached an F1 of about 0.89 to 0.92 on its own validation split. That split comes from the same 100 questions it was trained on, so it does not show that the verifier helps at selection time.

## Models and data

- Generator adapter: https://huggingface.co/Naitik-sudo123/phi35-mini-gsm8k-lora
- RFT adapter: https://huggingface.co/Naitik-sudo123/phi35-mini-gsm8k-rft-lora
- DPO adapter: https://huggingface.co/Naitik-sudo123/phi35-mini-gsm8k-dpo-lora
- Verifier adapter: https://huggingface.co/Naitik-sudo123/llama32-1b-gsm8k-verifier-lora
- Verifier training data: https://huggingface.co/datasets/Naitik-sudo123/gsm8k-verifier-training-data
- Static showcase of pre-computed pipeline runs: [add Space link]

## Repository contents

```
notebook-fine-tuning.ipynb    All the code, in the order it was run
README.md                     This file
```

Only code is stored here. Weights, checkpoints and datasets are on Hugging Face and Kaggle.

## Known limitations

- **Prompt mismatch.** The generator was trained with a chat template, but sampling and evaluation use a plain `Question: ... Answer:` prompt.
- **Noisy labels.** Verifier labels used a strict `####` answer match. A correct answer written in prose was labeled incorrect. The evaluation extractor is more lenient (it accepts the last number in a complete sentence), so labeling and evaluation do not use the same rule.
- **Calculator check is partial.** It only checks `<<expression=result>>` steps. A candidate with no such steps passes automatically.
- **Small scale.** The verifier data covers 100 questions and the evaluation 50, so a difference of a few points is within noise.
- **Notebook state.** It is a working notebook: some cells repeat, paths point to Kaggle datasets, and the evaluation outputs were not saved in it.

## How this was built

I built this with the help of an AI assistant (Claude), which wrote most of the code while I ran the experiments and learned how each stage works. I am now studying the underlying architecture and writing my own code, starting with a transformer built from scratch.

## Tools

Python, PyTorch, Hugging Face (transformers, PEFT, TRL, datasets), Unsloth, bitsandbytes, scikit-learn, Kaggle notebooks
