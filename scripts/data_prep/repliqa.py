"""Prepare ``repliqa`` for ReTreever training.

Faithful port of the original notebook recipe used to produce the
``repliqa`` dataset:

1. Load ``ServiceNow/repliqa`` from Hugging Face. The dataset is published as
   five non-overlapping splits ``repliqa_0`` ... ``repliqa_4``.
2. From the unique ``document_id``s in ``repliqa_3``, pick 400 at random
   (seed=42) and hold them out as the val split. The remaining ``repliqa_3``
   examples join ``repliqa_0`` + ``repliqa_1`` + ``repliqa_2`` to form train.
3. Use ``repliqa_4`` as the test split.
4. For every example, use ``long_answer`` as the gold context.
5. Assign a single integer ``context_uid`` per distinct context string,
   globally across all three splits.

Output ``DatasetDict`` splits: ``train``, ``val``, ``test``.

USAGE
-----
::

    python -m scripts.data_prep.repliqa --out-dir /path/to/repliqa
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset

DATASET_NAME = "repliqa"
HF_REPO = "ServiceNow/repliqa"
SEED = 42
N_VAL_DOC_IDS = 400


def _convert_examples(
    raw_examples,
    split_name: str,
    context_to_uid: dict[str, int],
) -> list[dict]:
    converted: list[dict] = []
    print(f"Converting {split_name}...")
    for example in raw_examples:
        context_text = example["long_answer"]
        if context_text not in context_to_uid:
            context_to_uid[context_text] = len(context_to_uid)
        context_uid = context_to_uid[context_text]

        converted.append(
            {
                "context": context_text,
                "contexts_list": [context_text],
                "titles_list": [example["document_topic"]],
                "useful_contexts": [1],
                "question": example["question"],
                "answer": example["answer"],
                "sample_idx": len(converted),
                "dataset": DATASET_NAME,
                "context_uid": context_uid,
                # RepliQA-specific fields retained for downstream analysis.
                "document_id": example["document_id"],
                "document_topic": example["document_topic"],
                "question_id": example["question_id"],
            }
        )
    print(f"  - {len(converted)} examples converted")
    return converted


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--hf-repo", default=HF_REPO)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--n-val-doc-ids",
        type=int,
        default=N_VAL_DOC_IDS,
        help="Number of document_ids from repliqa_3 to hold out as validation.",
    )
    args = parser.parse_args()

    print(f"Loading {args.hf_repo}...")
    raw = load_dataset(args.hf_repo, cache_dir=args.cache_dir)
    expected = {"repliqa_0", "repliqa_1", "repliqa_2", "repliqa_3", "repliqa_4"}
    missing = expected - set(raw.keys())
    if missing:
        raise SystemExit(
            f"Missing splits in {args.hf_repo}: {missing}. Available: {list(raw.keys())}"
        )

    print("\nSelecting validation document_ids from repliqa_3...")
    repliqa_3_doc_ids = list({ex["document_id"] for ex in raw["repliqa_3"]})
    print(f"  Unique document_ids in repliqa_3: {len(repliqa_3_doc_ids)}")

    # Use legacy RandomState (not default_rng) so byte-exact reproduction of
    # the original on-disk val split is possible.
    rng = np.random.RandomState(args.seed)
    val_doc_ids = set(rng.choice(repliqa_3_doc_ids, size=args.n_val_doc_ids, replace=False).tolist())
    print(f"  Selected {len(val_doc_ids)} document_ids for validation.")

    train_examples_raw: list[dict] = []
    val_examples_raw: list[dict] = []
    for split_name in ["repliqa_0", "repliqa_1", "repliqa_2", "repliqa_3"]:
        print(f"  Processing {split_name}...")
        for example in raw[split_name]:
            # Only repliqa_3 examples with selected doc_ids go to val.
            if split_name == "repliqa_3" and example["document_id"] in val_doc_ids:
                val_examples_raw.append(example)
            else:
                train_examples_raw.append(example)

    print(f"  Train (raw): {len(train_examples_raw)}  |  Val (raw): {len(val_examples_raw)}")

    test_examples_raw = list(raw["repliqa_4"])
    print(f"  Test (raw, = repliqa_4): {len(test_examples_raw)}")

    # Global uid map shared across all splits.
    context_to_uid: dict[str, int] = {}
    train_converted = _convert_examples(train_examples_raw, "train", context_to_uid)
    val_converted = _convert_examples(val_examples_raw, "val", context_to_uid)
    test_converted = _convert_examples(test_examples_raw, "test", context_to_uid)

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
    print(f"\nKNOWN_DATASETS['repliqa'] = {out_dir.resolve()}")


if __name__ == "__main__":
    main()
