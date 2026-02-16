from torch.utils.data import Dataset
import torch
from PIL import Image
from pathlib import Path
import os
from typing import List, Dict

class TextImageRetrievalDataset(Dataset):
    """
    Dataset for text-image retrieval (e.g., Flickr30k).
    
    Unlike ImageNet which creates pairs, this dataset loads pre-existing
    text-image pairs from the ReTreever format.
    """
    def __init__(
        self,
        data,  # HuggingFace dataset split (e.g., data['train'])
        images_base_dir=None,  # Base directory where images are stored
        subset_size=None,  # Optional: limit dataset size for debugging
    ):
        """
        Args:
            data: HuggingFace dataset split containing text-image pairs
            images_base_dir: Base directory to resolve relative image paths
        """
        self.data = data
        self.images_base_dir = Path(images_base_dir) if images_base_dir else None        
        self.subset_size = subset_size
        
        if subset_size is not None:
            self.data = self.data.select(range(min(subset_size, len(self.data))))
        
        print(f"Text-Image Retrieval Dataset initialized with {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        
        # Get image path
        image_path = example['context']
        
        # Resolve full path if base directory is provided
        if self.images_base_dir is not None:
            full_image_path = self.images_base_dir / image_path
        else:
            full_image_path = Path(image_path)
        
        # Load image
        try:
            image = Image.open(full_image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {full_image_path}: {e}")
            # Return a blank image as fallback
            image = Image.new('RGB', (224, 224), color='white')
        
        return {
            'question': example['question'],  # Text caption
            'context_image': image,  # PIL Image
            'context_uid': example['context_uid'],
            'sample_idx': example['sample_idx'],
            'dataset': example['dataset'],
            # Retain other fields if needed
            'answer': example.get('answer', ''),
            'img_id': example.get('img_id', ''),
            'filename': example.get('filename', ''),
        }



# Example usage
if __name__ == "__main__":
    from datasets import load_from_disk
    from transformers import AutoTokenizer, AutoImageProcessor
    
    # Load dataset
    data_path = "/mnt/dssk/data_rw/flickr30k/dataset"
    images_dir = "/mnt/dssk/data_rw/flickr30k/"
    
    dataset_dict = load_from_disk(data_path)
    
    # Create dataset instance
    train_dataset = TextImageRetrievalDataset(
        data=dataset_dict['train'],
        images_base_dir=images_dir
    )
    
    print(f"\nDataset size: {len(train_dataset)}")
    
    # Test a sample
    sample = train_dataset[0]
    print(f"\nSample 0:")
    print(f"  Question (caption): {sample['question'][:100]}...")
    print(f"  Context image size: {sample['context_image'].size}")
    print(f"  Context UID: {sample['context_uid']}")
    print(sample)
