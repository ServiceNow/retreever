"""Prepare ``topiocqa`` for ReTreever training.

Faithful port of the original ``create_topiocqa_allhistory.py`` recipe used
to produce the ``topiocqa`` dataset:

1. Download the raw TopiOCQA JSONL files directly from Hugging Face:
   ``McGill-NLP/TopiOCQA/data/topiocqa_{train,valid}.jsonl``.
2. Hold out 200 conversations (seed=42) from the original train split as the
   new val split. Use the remaining conversations as the new train split.
3. Promote the original valid split to the test split.
4. For every example, build the "all history" question as
   ``' [SEP] '.join(Context + [Question])`` and use ``Gold_passage.text`` as
   the gold context. Skip UNANSWERABLE examples (``Gold_passage.text`` empty).
5. Assign a single integer ``context_uid`` per distinct context string,
   globally across all three splits.

The resulting ``DatasetDict`` has splits: ``train``, ``val``, ``test``.

USAGE
-----
::

    python -m scripts.data_prep.topiocqa \
        --out-dir /path/to/topiocqa
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict

DATASET_NAME = "topiocqa"
TRAIN_URL = "https://huggingface.co/datasets/McGill-NLP/TopiOCQA/resolve/main/data/topiocqa_train.jsonl"
VALID_URL = "https://huggingface.co/datasets/McGill-NLP/TopiOCQA/resolve/main/data/topiocqa_valid.jsonl"

# Held-out conversations from train form the new val split. Matches the
# original recipe (np.random.seed(42), then choose 200 without replacement).
N_VAL_CONVERSATIONS = 200
SEED = 42


def _convert_examples(
    raw_examples,
    split_name: str,
    context_to_uid: dict[str, int],
) -> list[dict]:
    """Convert a list of raw TopiOCQA dicts to the ReTreever schema.

    ``context_to_uid`` is mutated to assign global, shared-across-splits ids.
    """
    converted: list[dict] = []
    skipped = 0
    print(f"Converting {split_name}...")
    for example in raw_examples:
        gold_passage = example["Gold_passage"]
        if "text" not in gold_passage or not gold_passage["text"]:
            skipped += 1
            continue

        # Concatenate prior turns + current turn with [SEP].
        context_parts = example["Context"] if example["Context"] else []
        full_question = " [SEP] ".join(list(context_parts) + [example["Question"]])

        context_text = gold_passage["text"]
        context_title = gold_passage["title"]

        if context_text not in context_to_uid:
            context_to_uid[context_text] = len(context_to_uid)
        context_uid = context_to_uid[context_text]

        converted.append(
            {
                "context": context_text,
                "contexts_list": [context_text],
                "titles_list": [context_title],
                "useful_contexts": [1],
                "question": full_question,
                "answer": example["Answer"],
                "sample_idx": len(converted),
                "dataset": DATASET_NAME,
                "context_uid": context_uid,
                # TopiOCQA-specific fields retained for downstream analysis.
                "is_nq": example["is_nq"],
                "topic": example["Topic"],
                "topic_section": example["Topic_section"],
                "rationale": example["Rationale"],
                "conversation_no": example["Conversation_no"],
                "turn_no": example["Turn_no"],
            }
        )

    print(f"  - {len(converted)} examples converted")
    print(f"  - {skipped} UNANSWERABLE examples skipped")
    return converted


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-url", default=TRAIN_URL)
    parser.add_argument("--valid-url", default=VALID_URL)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--n-val-conversations",
        type=int,
        default=N_VAL_CONVERSATIONS,
        help="Number of conversations to hold out from train for the new val split.",
    )
    args = parser.parse_args()

    print("Downloading and reading JSONL files...")
    df_train = pd.read_json(args.train_url, lines=True)
    df_valid = pd.read_json(args.valid_url, lines=True)

    print(f"Original train: {len(df_train)} examples; valid: {len(df_valid)} examples")

    print("\nSplitting train into train/val by conversation...")
    unique_convs = df_train["Conversation_no"].unique()
    print(f"  Unique conversations in train: {len(unique_convs)}")

    # Use legacy RandomState (not default_rng) so byte-exact reproduction of
    # the original on-disk val split is possible.
    rng = np.random.RandomState(args.seed)
    val_conv_ids = rng.choice(unique_convs, size=args.n_val_conversations, replace=False)
    val_conv_set = set(val_conv_ids.tolist())

    df_val = df_train[df_train["Conversation_no"].isin(val_conv_set)]
    df_train_new = df_train[~df_train["Conversation_no"].isin(val_conv_set)]
    print(f"  New train: {len(df_train_new)}  |  new val: {len(df_val)}  |  test: {len(df_valid)}")

    # Global uid map shared across all splits.
    context_to_uid: dict[str, int] = {}
    train_converted = _convert_examples(df_train_new.to_dict("records"), "train", context_to_uid)
    val_converted = _convert_examples(df_val.to_dict("records"), "val", context_to_uid)
    test_converted = _convert_examples(df_valid.to_dict("records"), "test", context_to_uid)

    print(f"\nUnique contexts across all splits: {len(context_to_uid)}")

    retreever_data = DatasetDict(
        {
            "train": Dataset.from_list(train_converted),
            "val": Dataset.from_list(val_converted),
            "test": Dataset.from_list(test_converted),
        }
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    retreever_data.save_to_disk(str(out_dir))

    print("\nDone. Summary:")
    for name, ds in retreever_data.items():
        print(f"  {name}: {len(ds)} rows")
    print(f"\nKNOWN_DATASETS['topiocqa'] = {out_dir.resolve()}")


if __name__ == "__main__":
    main()
