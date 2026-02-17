"""Evaluation script for ReTreever models.

Evaluates retrieval performance on test sets using Hit@K, NDCG@K, Recall@K, mAP@K metrics.
"""

import argparse
import logging
import sys
import os
import torch
import datasets
from pathlib import Path
from omegaconf import OmegaConf

from retreever.models.retreever import load_from_ckpt as load_retreever
from retreever.models.mrl import load_from_ckpt as load_mrl
from retreever.evaluation.evaluator import RetEvaluator
from retreever.data.collators import (
    SupervisedCollator,
    ImageSupervisedCollator,
    AudioSupervisedCollator,
    TextImageSupervisedCollator,
)
from retreever.data.imagenet_dataset import ImageNetRetrievalDataset
from retreever.data.voxceleb_dataset import VoxCeleb2RetrievalDataset
from retreever.data.text_image_dataset import TextImageRetrievalDataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def create_parser():
    """Create argument parser for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate ReTreever model on retrieval tasks"
    )

    # Model arguments
    parser.add_argument(
        "--model_ckpt",
        type=str,
        required=True,
        help="Path to model checkpoint (.bin file)",
    )
    parser.add_argument(
        "--model_cfg",
        type=str,
        required=True,
        help="Path to model configuration (.yaml file)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="retreever",
        choices=["retreever", "mrl"],
        help="Model type",
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name: nq, hotpotqa, repliqa, topiocqa, imagenet1k, voxceleb2, coco, flickr30k",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=None,
        help="Number of samples to evaluate (None = full dataset)",
    )

    # Evaluation settings
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[1, 3, 10, 50, 100],
        help="K values for metrics (e.g., Hit@K, NDCG@K)",
    )
    parser.add_argument(
        "--num_distractors",
        type=int,
        default=0,
        help="Number of distractor contexts to add to index",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on",
    )

    # Output
    parser.add_argument(
        "--save_path",
        type=str,
        default="eval_results",
        help="Path to save evaluation results",
    )

    return parser


def load_model(model_type, model_ckpt, model_cfg, device="cuda"):
    """Load model from checkpoint."""
    logger.info(f"Loading {model_type} model from {model_ckpt}")
    
    if model_type == "retreever":
        model, cfg = load_retreever(model_ckpt, model_cfg)
    elif model_type == "mrl":
        model, cfg = load_mrl(model_ckpt, model_cfg)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully on {device}")
    return model, cfg


def load_eval_dataset(dataset_name, data_dir, split="test", subset_size=None):
    """Load evaluation dataset."""
    logger.info(f"Loading {dataset_name} dataset from {data_dir}, split={split}")
    
    # Hack from: https://github.com/huggingface/datasets/issues/1785
    datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True
    
    if dataset_name == "imagenet1k":
        eval_data = ImageNetRetrievalDataset(
            data_dir=data_dir,
            split=split,
            subset=subset_size,
            for_eval=True,
        )
    elif dataset_name == "voxceleb2":
        eval_data = VoxCeleb2RetrievalDataset(
            data_dir=data_dir,
            split=split,
            for_eval=True,
            sample_rate=16000,
            audio_ext='m4a',
            subset=subset_size,
        )
    elif dataset_name in ["flickr30k", "coco"]:
        data_path = Path(data_dir) / "dataset"
        dataset_dict = datasets.load_from_disk(data_path)
        
        eval_data = TextImageRetrievalDataset(
            data=dataset_dict[split],
            images_base_dir=data_dir,
            subset_size=subset_size,
        )
    else:
        # Text datasets
        data = datasets.load_from_disk(data_dir)
        eval_data = data[split]
        
        if subset_size is not None:
            eval_data = eval_data.select(range(min(subset_size, len(eval_data))))
    
    logger.info(f"Loaded {len(eval_data)} examples")
    return eval_data


def create_collator(dataset_name, model):
    """Create appropriate collator for dataset."""
    if dataset_name == "imagenet1k":
        return ImageSupervisedCollator(
            query_processor=model.query_encoder.processor,
            ctx_processor=model.context_encoder.processor,
        )
    elif dataset_name == "voxceleb2":
        return AudioSupervisedCollator(
            model.query_encoder.processor,
            model.context_encoder.processor,
        )
    elif dataset_name in ["flickr30k", "coco"]:
        return TextImageSupervisedCollator(
            query_tokenizer=model.query_encoder.tokenizer,
            ctx_processor=model.context_encoder.processor,
        )
    else:
        # Text datasets
        return SupervisedCollator(
            model.query_encoder.tokenizer,
            model.context_encoder.tokenizer,
            model.query_encoder.prefix,
            model.context_encoder.prefix,
            context_field="contexts_list" if dataset_name == "hotpotqa" else "context",
        )


def main():
    """Main evaluation function."""
    parser = create_parser()
    args = parser.parse_args()

    # Load model
    model, cfg = load_model(
        args.model_type,
        args.model_ckpt,
        args.model_cfg,
        args.device
    )

    # Load dataset
    eval_data = load_eval_dataset(
        args.dataset,
        args.data_dir,
        args.split,
        args.subset_size
    )

    # Create collator
    collator = create_collator(args.dataset, model)

    # Create data loader
    eval_loader = torch.utils.data.DataLoader(
        eval_data,
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
    )

    # Create evaluator
    evaluator = RetEvaluator(
        ks=args.k_values,
        additional_ctxs_per_device=args.num_distractors,
    )

    # Run evaluation
    logger.info("Starting evaluation...")
    with torch.no_grad():
        metrics = evaluator(model, eval_loader)

    # Print results
    logger.info("\n" + "="*50)
    logger.info("Evaluation Results:")
    logger.info("="*50)
    for metric_name, value in sorted(metrics.items()):
        logger.info(f"{metric_name:30s}: {value:.4f}")
    logger.info("="*50)

    # Save results
    os.makedirs(args.save_path, exist_ok=True)
    results_file = os.path.join(
        args.save_path,
        f"{args.dataset}_{args.model_type}_{args.split}_results.txt"
    )
    
    with open(results_file, 'w') as f:
        f.write("="*50 + "\n")
        f.write("Evaluation Results\n")
        f.write("="*50 + "\n")
        f.write(f"Model: {args.model_ckpt}\n")
        f.write(f"Dataset: {args.dataset} ({args.split})\n")
        f.write(f"Samples: {len(eval_data)}\n")
        f.write("="*50 + "\n\n")
        for metric_name, value in sorted(metrics.items()):
            f.write(f"{metric_name:30s}: {value:.4f}\n")
    
    logger.info(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
