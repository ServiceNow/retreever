#!/usr/bin/env python3
"""
Convert Flickr30k dataset to ReTreever format for text-image retrieval.

This script:
1. Downloads Flickr30k from HuggingFace
2. Saves images to disk with unique IDs
3. Creates multiple text queries per image (one per caption)
4. Organizes into ReTreever format with 'train', 'val', 'test', and 'cuid2text' splits
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset, Dataset, DatasetDict

def convert_flickr30k_to_retreever(
    cache_dir,
    output_dir,
    save_images=True
):
    """
    Convert Flickr30k dataset to ReTreever format.
    
    Args:
        cache_dir: Directory to cache HuggingFace downloads
        output_dir: Directory for organized dataset
        save_images: Whether to save images to disk (recommended)
    """
    
    cache_dir = Path(cache_dir)
    output_dir = Path(output_dir)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("FLICKR30K DATASET CONVERSION TO RETREEVER FORMAT")
    print("="*80)
    print(f"Cache directory: {cache_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Save images to disk: {save_images}")
    print("="*80)
    
    # Load Flickr30k dataset
    print("\nLoading Flickr30k from HuggingFace...")
    ds = load_dataset("AnyModal/flickr30k", cache_dir=str(cache_dir))
    print("✓ Dataset loaded successfully")
    print(ds)
    
    # Create images directory if saving images
    if save_images:
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nImages will be saved to: {images_dir}")
    
    # Track unique images across all splits
    image_to_uid = {}  # Maps img_id to context_uid
    next_uid = [0]
    
    # Store image metadata for cuid2text
    image_metadata = {}  # context_uid -> {path, filename, img_id}
    
    def process_split(split_data, split_name):
        """Process a single split of the dataset."""
        print(f"\n{'='*80}")
        print(f"Processing {split_name} split")
        print(f"{'='*80}")
        
        converted_examples = []
        sample_idx = 0
        
        for example in tqdm(split_data, desc=f"Converting {split_name}"):
            img_id = example['img_id']
            filename = example['filename']
            image = example['image']
            original_captions = example['original_alt_text']
            sentids = example['sentids']
            
            # Assign unique context_uid for this image (shared across all splits)
            if img_id not in image_to_uid:
                context_uid = next_uid[0]
                image_to_uid[img_id] = context_uid
                next_uid[0] += 1
                
                # Save image to disk
                if save_images:
                    image_path = images_dir / f"img_{context_uid:06d}.jpg"
                    
                    # Convert to RGB if necessary
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    image.save(image_path, 'JPEG', quality=95)
                    image_path_str = str(image_path.relative_to(output_dir))
                else:
                    image_path_str = filename
                
                # Store metadata
                image_metadata[context_uid] = {
                    'context_uid': context_uid,
                    'context': image_path_str,  # Relative path to image
                    'filename': filename,
                    'img_id': img_id,
                }
            else:
                context_uid = image_to_uid[img_id]
                image_path_str = image_metadata[context_uid]['context']
            
            # Create one sample per caption
            for caption, sentid in zip(original_captions, sentids):
                converted_example = {
                    'context': image_path_str,  # Path to image
                    'contexts_list': [image_path_str],
                    'titles_list': [filename],  # Use filename as title
                    'useful_contexts': [1],
                    'question': caption,  # Caption is the query
                    'answer': filename,  # Filename as answer
                    'sample_idx': sample_idx,
                    'dataset': 'flickr30k',
                    'context_uid': context_uid,
                    # Retain Flickr-specific fields
                    'img_id': img_id,
                    'filename': filename,
                    'sentid': sentid,
                }
                
                converted_examples.append(converted_example)
                sample_idx += 1
        
        print(f"  ✓ Converted {len(converted_examples)} caption-image pairs")
        print(f"  ✓ From {len(set(ex['img_id'] for ex in split_data))} unique images")
        
        return converted_examples
    
    # Process all splits
    train_converted = process_split(ds['train'], 'train')
    val_converted = process_split(ds['validation'], 'val')
    test_converted = process_split(ds['test'], 'test')
    
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
            
            # Show sample
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
            
            # Verify a sample image
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
                # Estimate total size
                sample_size = image_files[0].stat().st_size
                total_size_gb = (len(image_files) * sample_size) / (1024**3)
                print(f"\nEstimated disk usage for images: ~{total_size_gb:.2f} GB")
        
    except Exception as e:
        print(f"Could not load statistics: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert Flickr30k dataset to ReTreever format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python prepare_flickr30k.py --output_dir ./data/flickr30k

  # With custom cache
  python prepare_flickr30k.py --cache_dir ./cache --output_dir ./data/flickr30k

  # Verify existing dataset
  python prepare_flickr30k.py --output_dir ./data/flickr30k --verify_only

  # Show statistics
  python prepare_flickr30k.py --output_dir ./data/flickr30k --stats

Note: This creates a text-image retrieval dataset where:
- Questions are image captions (text)
- Contexts are images (saved to disk)
- Each image has ~5 captions, creating 5 samples per image
        """
    )
    
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='./hf_cache',
        help='Directory to cache HuggingFace downloads'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data/flickr30k',
        help='Directory for organized dataset'
    )
    
    parser.add_argument(
        '--no_save_images',
        action='store_true',
        help='Do not save images to disk (not recommended)'
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
    retreever_data = convert_flickr30k_to_retreever(
        args.cache_dir,
        args.output_dir,
        save_images=not args.no_save_images
    )
    
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