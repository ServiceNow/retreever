# Third-Party Notices

ReTreever is a training stack. It does **not** bundle any datasets or model
weights — these are downloaded at runtime from their original sources under
their own licenses. A small amount of third-party source code is adapted into
this repository; see [Adapted source code](#adapted--vendored-source-code).

License entries below reflect each project's publicly published terms.

---

## Datasets

Fetched at runtime by `scripts/data_prep/*`; none are redistributed here.

| Dataset | Key / source | How obtained | Typical license / terms |
|---|---|---|---|
| HotpotQA | HF `hotpot_qa` (config `distractor`) | `load_dataset` from HF | CC BY-SA 4.0 |
| TopiOCQA | HF `McGill-NLP/TopiOCQA` (JSONL) | direct download from HF | CC BY-SA 4.0 (per dataset card) |
| RepliQA | HF `ServiceNow/repliqa` | `load_dataset` from HF | CC BY 4.0 (ServiceNow's own dataset) |
| ImageNet-1K | HF `ILSVRC/imagenet-1k` (gated) | `load_dataset` after accepting license + `hf login` | Custom ImageNet terms — non-commercial research only; must accept gate |
| VoxCeleb2 | Oxford VGG (`robots.ox.ac.uk/.../vox2.html`) — registration required | manual download of `vox2_dev_aac.zip` / `vox2_test_aac.zip`, then local flatten script | VoxCeleb usage terms — research use; registration/agreement required (annotations CC BY) |

---

## Pretrained model weights

Downloaded on demand (Hugging Face / torch hub) and selected via
`model.encoder_type` in `retreever/models/encoders.py`. The default is `bge`.
None are bundled.

| Encoder key(s) | Model | License |
|---|---|---|
| `bge`, `bgem3` | BAAI/bge-large-en-v1.5, bge-m3 | MIT |
| `bert` | bert-base-uncased | Apache-2.0 |
| `dpr` | facebook/dpr-*-encoder-single-nq-base | CC-BY-NC-4.0 (non-commercial) |
| `contriever_msmarco` | facebook/contriever-msmarco | CC-BY-NC-4.0 (non-commercial; per facebookresearch/contriever) |
| `distilbert_msmarco` | sentence-transformers/msmarco-distilbert-cos-v5 | Apache-2.0 |
| `simcse` | princeton-nlp/sup-simcse-bert-base-uncased | MIT |
| `llm` (optional) | meta-llama/Llama-3.2-1B-Instruct | Llama 3.2 Community License (custom) |
| `dinov2-*` | facebook/dinov2-{small,base,large,giant} | Apache-2.0 |
| `resnet18..152` | torchvision ImageNet-pretrained weights | BSD-3-Clause |
| `clip-vit-*` | openai/clip-vit-* | MIT |
| `wav2vec2-*` | facebook/wav2vec2-* | Apache-2.0 |
| `hubert-*` | facebook/hubert-* | Apache-2.0 |
| `wavlm-*` | microsoft/wavlm-* | MIT |
| `ast` | MIT/ast-finetuned-audioset-... | BSD-3-Clause |
| `clap-fused`, `clap-unfused` | laion/clap-htsat-* | Apache-2.0 |
| `panns-*` (optional) | PANNs Cnn6/10/14 (panns-inference) | Apache-2.0 |
| `beats` (optional, `trust_remote_code=True`) | microsoft/unilm BEATs | MIT |

---

## Adapted / vendored source code

Third-party code adapted into this repository, with attribution.

| Location | Adapted / taken from | License |
|---|---|---|
| `retreever/utils/neural.py` (`AllGather`) | `Lightning-AI/lightning-bolts` | Apache-2.0 |
| `retreever/utils/neural.py` (header utilities) | `bigcode-project/bigcode-encoder` | Apache-2.0 |
| `scripts/train.py` (cache workaround) | HuggingFace `datasets` GitHub issue snippet | Apache-2.0 |

---

## Dependencies (installed, not bundled)

Declared in `requirements.txt` / `pyproject.toml`.

| Package | License |
|---|---|
| torch, torchvision, torchaudio | BSD-3-Clause |
| transformers | Apache-2.0 |
| datasets | Apache-2.0 |
| accelerate | Apache-2.0 |
| deepspeed | Apache-2.0 |
| hydra-core | MIT |
| omegaconf | BSD-3-Clause |
| huggingface_hub | Apache-2.0 |
| numpy, scipy | BSD-3-Clause |
| pandas | BSD-3-Clause |
| tqdm | MPL-2.0 / MIT |
| evaluate | Apache-2.0 |
| Pillow | HPND (MIT-CMU) |
| sentencepiece | Apache-2.0 |
| annoy | Apache-2.0 |
| faiss-cpu | MIT |
| wandb | MIT (client); metrics upload to a hosted service unless `WANDB_MODE=disabled` |
| peft (optional, LoRA) | Apache-2.0 |
| setuptools (build) | MIT |
| panns-inference (optional) | Apache-2.0 |
