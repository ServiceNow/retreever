"""Prepare HotpotQA for ReTreever training.

Downloads ``hotpot_qa`` (config ``distractor``) from Hugging Face and converts
it to the same canonical ReTreever schema used by ``topiocqa_all_history`` and
``repliqa_0to3_4``:

  - ``context``         (str) -- concatenation of supporting paragraphs
                                  (kept around so the dataset also works with
                                  collators configured with
                                  ``context_field="context"``)
  - ``contexts_list``   (list[str]) -- all 10 distractor paragraphs
  - ``titles_list``     (list[str]) -- title for each paragraph
  - ``useful_contexts`` (list[int 0/1]) -- 1 for supporting paragraphs
  - ``question``        (str)
  - ``answer``          (str)
  - ``sample_idx``      (int)
  - ``dataset``         ("hotpotqa")
  - ``context_uid``     (int) -- unique per distinct gold-context concatenation;
                                  shared across train/val splits

Output ``DatasetDict``: ``train``, ``val``, ``cuid2text``.

Note: there is no original ``create_hotpotqa.py`` script in the research repo
that we are mirroring. This recipe matches the schema observed in the canonical
``hotpotqa`` snapshot used during training.

USAGE
-----
::

    python -m scripts.data_prep.hotpotqa --out-dir /path/to/hotpotqa
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm

DATASET_NAME = "hotpotqa"


def _paragraph_text(title: str, sentences: list[str]) -> str:
    return title + ". " + " ".join(sentences)


def _convert_examples(
    raw_examples,
    split_name: str,
    context_to_uid: dict[str, int],
) -> list[dict]:
    converted: list[dict] = []
    print(f"Converting {split_name}...")
    for example in tqdm(raw_examples, desc=split_name):
        titles = example["context"]["title"]
        sentences_per_title = example["context"]["sentences"]
        contexts_list = [
            _paragraph_text(t, s) for t, s in zip(titles, sentences_per_title)
        ]

        supporting_titles = set(example["supporting_facts"]["title"])
        useful_contexts = [1 if t in supporting_titles else 0 for t in titles]

        # Concatenated gold context: matches the collator's concatenate_ctxs.
        gold_concat = " ".join(
            ctx for ctx, flag in zip(contexts_list, useful_contexts) if flag == 1
        )

        if gold_concat not in context_to_uid:
            context_to_uid[gold_concat] = len(context_to_uid)
        context_uid = context_to_uid[gold_concat]

        converted.append(
            {
                "context": gold_concat,
                "contexts_list": contexts_list,
                "titles_list": list(titles),
                "useful_contexts": useful_contexts,
                "question": example["question"],
                "answer": example["answer"],
                "sample_idx": len(converted),
                "dataset": DATASET_NAME,
                "context_uid": context_uid,
            }
        )
    print(f"  - {len(converted)} examples converted")
    return converted


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--hf-repo", default="hotpot_qa")
    parser.add_argument("--hf-config", default="distractor")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    print(f"Loading {args.hf_repo} (config={args.hf_config})...")
    raw = load_dataset(args.hf_repo, args.hf_config, cache_dir=args.cache_dir)
    print(f"  Available splits: {list(raw.keys())}")

    context_to_uid: dict[str, int] = {}

    splits: dict[str, list[dict]] = {}
    splits["train"] = _convert_examples(raw["train"], "train", context_to_uid)
    val_key = "validation" if "validation" in raw else ("val" if "val" in raw else None)
    if val_key is None:
        raise SystemExit(f"No validation split found. Available: {list(raw.keys())}")
    splits["val"] = _convert_examples(raw[val_key], "val", context_to_uid)

    print(f"\nUnique gold-context concatenations across all splits: {len(context_to_uid)}")
    cuid2text = [
        {"context_uid": uid, "context": text} for text, uid in context_to_uid.items()
    ]
    cuid2text.sort(key=lambda x: x["context_uid"])

    retreever_data = DatasetDict(
        {
            "train": Dataset.from_list(splits["train"]),
            "val": Dataset.from_list(splits["val"]),
            "cuid2text": Dataset.from_list(cuid2text),
        }
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    retreever_data.save_to_disk(str(out_dir))

    print("\nDone. Summary:")
    for name, ds in retreever_data.items():
        print(f"  {name}: {len(ds)} rows")
    print(f"\nKNOWN_DATASETS['hotpotqa'] = {out_dir.resolve()}")


if __name__ == "__main__":
    main()
