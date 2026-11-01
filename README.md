# GSM8K: QLoRA Fine-Tuning with a Verifier and Best-of-N

A practice project on grade-school math (GSM8K): fine-tune a small language model, train a second model to judge its answers, and test whether picking the best of several sampled answers beats a single greedy answer.

## What was built

1. **Generator.** Phi-3.5-mini-instruct fine-tuned with QLoRA (4-bit, LoRA rank 16, alpha 16, all seven attention and MLP projection modules) on the GSM8K training set. One epoch, batch size 8, learning rate 2e-4.
2. **Candidate sampling.** For 100 shuffled GSM8K training questions, 4 answers were sampled (temperature 0.7, top-p 0.9, up to 256 tokens) from each of two checkpoints, checkpoint-935 and checkpoint-500. That gives 800 records (100 × 4 × 2). Each answer was labeled correct if its final `#### N` answer matched the gold answer.
3. **Verifier.** Llama-3.2-1B-Instruct fine-tuned with LoRA (rank 16, alpha 32) as a binary classifier of whether a candidate solution is correct. 4-bit, 3 epochs, learning rate 2e-4, inputs cut to 512 tokens, stratified 90/10 split, best epoch chosen by F1.
4. **Calculator check.** Every `<<expression=result>>` step in a candidate is recomputed with a safe arithmetic evaluator. Candidates with a wrong step are dropped before the verifier scores the rest. If none survive, all candidates are kept.
5. **Best-of-N.** Sample 4 answers, apply the calculator check, and let the verifier pick the highest-scoring one. Compared against a single greedy answer.
6. **RFT and DPO attempts.** Rejection-sampling fine-tuning (full GSM8K training set plus the model's own correct answers from step 2) and DPO (correct vs incorrect answer pairs from step 2, beta 0.1, learning rate 5e-6) were attempted on top of the generator. No valid evaluation exists for them.

## Results

Evaluated on 50 GSM8K test questions (the test split shuffled with seed 42, first 50 taken). Generator: the QLoRA-fine-tuned adapter with all seven LoRA modules loaded.

| Method | Accuracy |
|---|---|
| Greedy baseline | 72% (36/50, 95% interval roughly 58–83%) |
| Best-of-4 + calculator + verifier | 72% (36/50) |

Best-of-N with the verifier gave no measurable improvement over greedy. With 50 questions, a small gain could not be detected either way.

**Discarded results.** Earlier numbers from this project (44% greedy rising to 58% with best-of-4, calculator, RFT and DPO) were measured with only two of the adapter's seven LoRA modules loaded. They are not valid and are not used here.

The verifier reached an F1 of about 0.89 to 0.92 on its own validation split. That split comes from the same 100 questions it was trained on, so it does not show that the verifier helps at selection time.

## Unconfirmed points and possible confusions

Read these before relying on anything above.

- **Source of the 72% figures.** The notebook has no saved output for the evaluation cells. The 72% / 72% numbers come from the author's records and the linked Space, and cannot be checked from this file.
- **Which adapter the notebook loads.** The notebook's final generator cells load the DPO adapter path. The 72% is attributed to the QLoRA generator with all seven modules loaded. The notebook does not show which adapter produced that figure, and this was not re-verified.
- **RFT and DPO training.** The RFT cell lists all seven LoRA modules. The author's recollection is that RFT and DPO were not run on the full seven-module adapter. This is unconfirmed. Either way, no valid evaluation exists for them, so no result is claimed.
- **Verifier F1.** The 0.89–0.92 range is from the author's records. The notebook does not save the training output.
- **Run order.** The notebook is a working notebook. Cells were edited and rerun over time, some repeat, and it may not match the exact order or version in which each result was produced.
- **Dataset file name.** The sampling cell writes `verifier_training_data_v2.jsonl`. The dataset published on Hugging Face is named `verifier_training_data.jsonl`.
- **Stale "58%" label in the code.** The upload-script cell contains the commit message "Upload DPO-tuned generator adapter (final generator, 58% best-of-4)" and describes the DPO adapter as the "final" generator. These were written before the partial-loading problem was found. The 58% figure was discarded (see Discarded results), and the DPO adapter has no valid evaluation. The notebook was deliberately left unedited. If the script was run as written, the commit history of the DPO adapter repository on Hugging Face may carry the same message. Treat it as outdated.
- **Other comments in the code.** Comments or names that call a model "final" or "best", and the evaluation cell comment "swap in — was phi35_mini_gsm8k_lora before", record how the notebook was edited over time. They are not evidence about which adapter produced a reported result.
- **Printed output.** Any printed accuracy in the notebook's saved output predates the discovery of the partial-loading problem. Only the results in this README are valid.

## Known limitations

- **Prompt mismatch.** The generator was trained with a chat template, but sampling and evaluation use a plain `Question: ... Answer:` prompt.
- **Noisy labels.** Verifier labels used a strict `####` answer match. A correct answer written in prose was labeled incorrect. The evaluation extractor is more lenient (it accepts the last number in a complete sentence), so labeling and evaluation do not use the same rule.
- **Calculator check is partial.** It only checks `<<expression=result>>` steps. A candidate with no such steps passes automatically.
- **Small scale.** The verifier data covers 100 questions and the evaluation 50, so a difference of a few points is within noise.
- **Not rerun or edited.** None of these issues was fixed by rerunning, and the notebook was not edited afterwards. It is kept as it was run, so its contents show what actually happened.

## Models and data

- Generator adapter: https://huggingface.co/Naitik-sudo123/phi35-mini-gsm8k-lora
- RFT adapter: https://huggingface.co/Naitik-sudo123/phi35-mini-gsm8k-rft-lora
- DPO adapter: https://huggingface.co/Naitik-sudo123/phi35-mini-gsm8k-dpo-lora
- Verifier adapter: https://huggingface.co/Naitik-sudo123/llama32-1b-gsm8k-verifier-lora
- Verifier training data: https://huggingface.co/datasets/Naitik-sudo123/gsm8k-verifier-training-data
- Static showcase of pre-computed pipeline runs: https://huggingface.co/spaces/Naitik-sudo123/gsm8k-generator-verifier

## Repository contents

```
notebook-fine-tuning.ipynb    All the code from the project (a working notebook, not a clean run)
README.md                     This file
```

Only code is stored here. Weights, checkpoints and datasets are on Hugging Face and Kaggle. The notebook is kept exactly as it was during the project and has not been cleaned up. Where its code, comments or output disagree with this README, this README is correct.

## How this was built

I built this with the help of an AI assistant (Claude), which wrote most of the code while I ran the experiments and learned how each stage works. I am now studying the underlying architecture and writing my own code, starting with a transformer built from scratch.

## Tools

Python, PyTorch, Hugging Face (transformers, PEFT, TRL, datasets), Unsloth, bitsandbytes, scikit-learn, Kaggle notebooks
