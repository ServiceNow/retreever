#!/usr/bin/env python3
"""
Convert COCO dataset to ReTreever format for text-image retrieval.

This script:
1. Loads COCO captions from annotations
2. Saves images to disk with unique IDs
3. Creates multiple text queries per image (one per caption)
4. Organizes into ReTreever format with 'train', 'val', 'test', and 'cuid2text' splits
"""

import os
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import shutil
import numpy as np
from datasets import Dataset, DatasetDict

def load_coco_annotations(ann_file):
    """Load COCO annotations file."""
    with open(ann_file, 'r') as f:
        data = json.load(f)
    return data

def convert_coco_to_retreever(
    coco_data_dir,
    output_dir,
    save_images=True,
    train_val_split_ratio=0.95,  # Use 95% of train for train, 5% for val
    seed=42
):
    """
    Convert COCO dataset to ReTreever format.
    
    Args:
        coco_data_dir: Directory containing COCO data (annotations, train2014, val2014)
        output_dir: Directory for organized dataset
        save_images: Whether to copy images to new structure (recommended)
        train_val_split_ratio: Ratio of train2014 to use for validation 
        seed: Random seed for val/test split
    """
    
    coco_data_dir = Path(coco_data_dir)
    output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("COCO DATASET CONVERSION TO RETREEVER FORMAT")
    print("="*80)
    print(f"COCO data directory: {coco_data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Save images to disk: {save_images}")
    print("="*80)
    
    # Define paths
    ann_dir = coco_data_dir / "annotations"
    train_img_dir = coco_data_dir / "train2014"
    val_img_dir = coco_data_dir / "val2014"
    
    # Check if directories exist
    for path, name in [(ann_dir, "annotations"), (train_img_dir, "train2014"), (val_img_dir, "val2014")]:
        if not path.exists():
            print(f"ERROR: {name} directory not found at {path}")
            return None
    
    # Load annotations
    print("\nLoading COCO annotations...")
    train_ann_file = ann_dir / "captions_train2014.json"
    val_ann_file = ann_dir / "captions_val2014.json"
    
    if not train_ann_file.exists() or not val_ann_file.exists():
        print(f"ERROR: Caption files not found")
        print(f"  Looking for: {train_ann_file}")
        print(f"  Looking for: {val_ann_file}")
        return None
    
    train_coco = load_coco_annotations(train_ann_file)
    val_coco = load_coco_annotations(val_ann_file)
    
    print(f"✓ Train: {len(train_coco['images'])} images, {len(train_coco['annotations'])} captions")
    print(f"✓ Val: {len(val_coco['images'])} images, {len(val_coco['annotations'])} captions")
    
    # Create images directory if saving images
    if save_images:
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nImages will be saved to: {images_dir}")
    
    # Track unique images across all splits
    image_to_uid = {}  # Maps image_id to context_uid
    next_uid = [0]
    
    # Store image metadata for cuid2text
    image_metadata = {}  # context_uid -> {path, filename, image_id}
    
    def process_coco_split(coco_data, source_img_dir, split_name, image_ids_to_use=None):
        """Process COCO data for a split."""
        print(f"\n{'='*80}")
        print(f"Processing {split_name} split")
        print(f"{'='*80}")
        
        # Build image_id to image info mapping
        image_id_to_info = {img['id']: img for img in coco_data['images']}
        
        # Build image_id to captions mapping
        image_id_to_captions = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in image_id_to_captions:
                image_id_to_captions[img_id] = []
            image_id_to_captions[img_id].append(ann['caption'])
        
        # Filter by image_ids_to_use if provided
        if image_ids_to_use is not None:
            image_ids = [img_id for img_id in image_id_to_info.keys() if img_id in image_ids_to_use]
        else:
            image_ids = list(image_id_to_info.keys())
        
        print(f"Processing {len(image_ids)} images...")
        
        converted_examples = []
        sample_idx = 0
        
        for img_id in tqdm(image_ids, desc=f"Converting {split_name}"):
            img_info = image_id_to_info[img_id]
            filename = img_info['file_name']
            captions = image_id_to_captions.get(img_id, [])
            
            if len(captions) == 0:
                continue
            
            # Assign unique context_uid for this image (shared across all splits)
            if img_id not in image_to_uid:
                context_uid = next_uid[0]
                image_to_uid[img_id] = context_uid
                next_uid[0] += 1
                
                # Save/copy image to disk
                if save_images:
                    source_image_path = source_img_dir / filename
                    dest_image_path = images_dir / f"img_{context_uid:06d}.jpg"
                    
                    if source_image_path.exists():
                        try:
                            # Copy image
                            img = Image.open(source_image_path)
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            img.save(dest_image_path, 'JPEG', quality=95)
                        except Exception as e:
                            print(f"\nWARNING: Error processing {source_image_path}: {e}")
                            continue
                    else:
                        print(f"\nWARNING: Image not found: {source_image_path}")
                        continue
                    
                    image_path_str = f"images/img_{context_uid:06d}.jpg"
                else:
                    image_path_str = filename
                
                # Store metadata
                image_metadata[context_uid] = {
                    'context_uid': context_uid,
                    'context': image_path_str,
                    'filename': filename,
                    'image_id': img_id,
                }
            else:
                context_uid = image_to_uid[img_id]
                image_path_str = image_metadata[context_uid]['context']
            
            # Create one sample per caption
            for cap_idx, caption in enumerate(captions):
                converted_example = {
                    'context': image_path_str,
                    'contexts_list': [image_path_str],
                    'titles_list': [filename],
                    'useful_contexts': [1],
                    'question': caption.strip(),  # Caption is the query
                    'answer': filename,
                    'sample_idx': sample_idx,
                    'dataset': 'coco',
                    'context_uid': context_uid,
                    # Retain COCO-specific fields
                    'image_id': img_id,
                    'filename': filename,
                    'caption_id': cap_idx,
                }
                
                converted_examples.append(converted_example)
                sample_idx += 1
        
        print(f"  ✓ Converted {len(converted_examples)} caption-image pairs")
        print(f"  ✓ From {len(set(ex['image_id'] for ex in converted_examples))} unique images")
        
        return converted_examples
    
    # Split train2014 into train and val
    print("\nSplitting train2014 into train and val...")
    train_image_ids = [img['id'] for img in train_coco['images']]
    
    np.random.seed(seed)
    np.random.shuffle(train_image_ids)
    
    split_idx = int(len(train_image_ids) * train_val_split_ratio)
    train_image_ids_set = set(train_image_ids[:split_idx])
    val_image_ids_set = set(train_image_ids[split_idx:])
    
    print(f"Train images: {len(train_image_ids_set)}")
    print(f"Val images: {len(val_image_ids_set)}")
    
    # Process train and val from train2014
    train_converted = process_coco_split(train_coco, train_img_dir, 'train', train_image_ids_set)
    val_converted = process_coco_split(train_coco, train_img_dir, 'val', val_image_ids_set)
    
    # Use complete val2014 as test
    print(f"\nUsing complete val2014 as test ({len(val_coco['images'])} images)...")
    test_converted = process_coco_split(val_coco, val_img_dir, 'test')
    
    
    # Create cuid2text mapping (sorted by uid)
    print(f"\nCreating cuid2text mapping with {len(image_metadata)} unique images...")
    cuid2text_examples = sorted(image_metadata.values(), key=lambda x: x['context_uid'])
    
    # Create the final DatasetDict
    print("\nCreating final DatasetDict...")
    retreever_data = DatasetDict({
        'train': Dataset.from_list(train_converted),
        'val': Dataset.from_list(val_converted),
        'test': Dataset.from_list(test_converted),
        'cuid2text': Dataset.from_list(cuid2text_examples),
    })
    
    return retreever_data


def verify_dataset(output_dir):
    """Verify the converted dataset."""
    output_dir = Path(output_dir)
    
    print("\n" + "="*80)
    print("VERIFYING CONVERTED DATASET")
    print("="*80)
    
    try:
        from datasets import load_from_disk
        
        retreever_data = load_from_disk(str(output_dir / "dataset"))
        
        print("\nDataset structure:")
        print(retreever_data)
        
        # Check each split
        for split_name in ['train', 'val', 'test']:
            split_data = retreever_data[split_name]
            print(f"\n{split_name.upper()} split:")
            print(f"  ✓ {len(split_data)} samples")
            
            if len(split_data) > 0:
                sample = split_data[0]
                print(f"  Sample fields: {list(sample.keys())}")
                print(f"  Sample question: {sample['question'][:80]}...")
                print(f"  Sample context: {sample['context']}")
        
        # Check cuid2text
        cuid2text = retreever_data['cuid2text']
        print(f"\ncuid2text mapping:")
        print(f"  ✓ {len(cuid2text)} unique images")
        if len(cuid2text) > 0:
            sample = cuid2text[0]
            print(f"  Sample: context_uid={sample['context_uid']}, path={sample['context']}")
        
        # Verify required fields
        print(f"\nRequired fields check:")
        required_fields = ['context', 'contexts_list', 'titles_list', 'useful_contexts',
                          'question', 'answer', 'sample_idx', 'dataset', 'context_uid']
        for field in required_fields:
            present = field in retreever_data['train'].features
            print(f"  {'✓' if present else '✗'} {field}")
        
        # Check if images exist
        images_dir = output_dir / "images"
        if images_dir.exists():
            image_files = list(images_dir.glob("*.jpg"))
            print(f"\nImages on disk:")
            print(f"  ✓ {len(image_files)} image files found")
            
            if image_files:
                sample_img = Image.open(image_files[0])
                print(f"  ✓ Sample image loads: {sample_img.size}, {sample_img.mode}")
        
        print("\n✓ Verification complete!")
        
    except Exception as e:
        print(f"\n✗ Verification failed: {e}")


def show_statistics(output_dir):
    """Show dataset statistics."""
    output_dir = Path(output_dir)
    
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    
    try:
        from datasets import load_from_disk
        
        retreever_data = load_from_disk(str(output_dir / "dataset"))
        
        for split_name in ['train', 'val', 'test']:
            split_data = retreever_data[split_name]
            unique_images = len(set(ex['context_uid'] for ex in split_data))
            
            print(f"\n{split_name.upper()} Split:")
            print(f"  Total samples (caption-image pairs): {len(split_data):,}")
            print(f"  Unique images: {unique_images:,}")
            print(f"  Captions per image: {len(split_data) / unique_images:.1f} (avg)")
        
        print(f"\nTotal unique images across all splits: {len(retreever_data['cuid2text']):,}")
        
        # Disk usage
        images_dir = output_dir / "images"
        if images_dir.exists():
            image_files = list(images_dir.glob("*.jpg"))
            if image_files:
                sample_size = image_files[0].stat().st_size
                total_size_gb = (len(image_files) * sample_size) / (1024**3)
                print(f"\nEstimated disk usage for images: ~{total_size_gb:.2f} GB")
        
    except Exception as e:
        print(f"Could not load statistics: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert COCO dataset to ReTreever format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python prepare_coco.py --coco_data_dir ./data/coco_data --output_dir ./data/coco

  # Verify existing dataset
  python prepare_coco.py --output_dir ./data/coco --verify_only

  # Show statistics
  python prepare_coco.py --output_dir ./data/coco --stats

Note: This creates a text-image retrieval dataset where:
- Questions are image captions (text)
- Contexts are images (saved to disk)
- Each image has ~5 captions, creating 5 samples per image
- Train2014 is split 95/5 into train and validation sets
- Val2014 is used as the complete test set
        """
    )
    
    parser.add_argument(
        '--coco_data_dir',
        type=str,
        default='./data/coco_data',
        help='Directory containing COCO data (annotations, train2014, val2014)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data/coco',
        help='Directory for organized dataset'
    )
    
    parser.add_argument(
        '--no_save_images',
        action='store_true',
        help='Do not copy images to new structure (not recommended)'
    )

    parser.add_argument(
        '--train_val_split_ratio',
        type=float,
        default=0.95,
        help='Ratio of train2014 to use for training (rest goes to validation)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for val/test split'
    )
    
    parser.add_argument(
        '--verify_only',
        action='store_true',
        help='Only verify existing dataset'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show dataset statistics'
    )
    
    args = parser.parse_args()
    
    # Verify only
    if args.verify_only:
        verify_dataset(args.output_dir)
        return
    
    # Show stats only
    if args.stats:
        show_statistics(args.output_dir)
        return
    
    # Convert dataset
    print("\nStarting conversion...")
    retreever_data = convert_coco_to_retreever(
        args.coco_data_dir,
        args.output_dir,
        save_images=not args.no_save_images,
        train_val_split_ratio=args.train_val_split_ratio,
        seed=args.seed
    )
    
    if retreever_data is None:
        print("\n✗ Conversion failed!")
        return
    
    # Save dataset
    dataset_dir = Path(args.output_dir) / "dataset"
    print(f"\nSaving dataset to {dataset_dir}...")
    retreever_data.save_to_disk(str(dataset_dir))
    print("✓ Dataset saved successfully")
    
    # Verify
    verify_dataset(args.output_dir)
    
    # Show stats
    show_statistics(args.output_dir)
    
    print("\n" + "="*80)
    print("CONVERSION COMPLETE!")
    print("="*80)
    print(f"\nYour dataset is ready at: {args.output_dir}")
    print(f"\nDataset structure:")
    print(f"  {args.output_dir}/")
    print(f"  ├── images/           # Image files")
    print(f"  └── dataset/          # ReTreever format data")
    print(f"      ├── train/")
    print(f"      ├── val/")
    print(f"      ├── test/")
    print(f"      └── cuid2text/")
    print(f"\nTo load in your code:")
    print(f"  from datasets import load_from_disk")
    print(f"  data = load_from_disk('{dataset_dir}')")
    print("="*80)


if __name__ == '__main__':
    main()