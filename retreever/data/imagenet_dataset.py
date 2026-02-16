from torch.utils.data import Dataset
import numpy as np
from torchvision import datasets
import os
from tqdm import tqdm

class ImageNetRetrievalDataset(Dataset):
    def __init__(self, 
                 data_dir, split='train', 
                 num_contexts=100, 
                 seed=42, 
                 subset=None,
                 for_eval=False):
        self.dataset = datasets.ImageFolder(os.path.join(data_dir, split))
        self.num_contexts = num_contexts
        self.for_eval = for_eval
        
        # Group by class
        print("Grouping images by class...")
        self.class_to_indices = {}
        for idx, label in enumerate(tqdm(self.dataset.targets, desc="Building class index")):
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)
        
        # Pre-sample contexts for each query (compact storage)
        print(f"Pre-sampling {num_contexts} contexts per query...")
        np.random.seed(seed)
        self.query_contexts = []  # List of context indices for each query
        
        if subset is not None:
            all_classes = list(self.class_to_indices.keys())
            selected_classes = np.random.choice(all_classes, size=subset, replace=False)
            self.class_to_indices = {c: self.class_to_indices[c] for c in selected_classes}
        
            # Build list of valid indices
            self.valid_indices = []
            for c in selected_classes:
                self.valid_indices.extend(self.class_to_indices[c])
            self.valid_indices = sorted(self.valid_indices)
            
            print(f"Filtered to {len(selected_classes)} classes with {len(self.valid_indices)} total images")
        else:
            self.valid_indices = list(range(len(self.dataset)))
            
        for idx in tqdm(self.valid_indices, desc="Sampling contexts"):
            if self.for_eval:
                context_indices = [idx]
            else:
                label = self.dataset.targets[idx]
                class_indices = [i for i in self.class_to_indices[label] if i != idx]
                
                if len(class_indices) >= self.num_contexts:
                    context_indices = np.random.choice(class_indices, size=self.num_contexts, replace=False).tolist()
                else:
                    context_indices = class_indices.copy()

            self.query_contexts.append(context_indices)
        
        # Build cumulative sum for fast indexing
        self.num_contexts_per_query = [len(contexts) for contexts in self.query_contexts]
        self.cumsum = np.cumsum([0] + self.num_contexts_per_query)
        
        print(f"Dataset ready: {len(self.dataset)} queries -> {self.cumsum[-1]} total pairs")
    
    def __len__(self):
        return self.cumsum[-1]
    
    def __getitem__(self, idx):
        # Map flat index to (query_idx, context_offset)
        query_pos = np.searchsorted(self.cumsum[1:], idx, side='right')
        context_offset = idx - self.cumsum[query_pos]
        
        # Get actual query and context index
        query_idx = self.valid_indices[query_pos] 
        context_idx = self.query_contexts[query_pos][context_offset]
        
        # Load images
        query_img, label = self.dataset[query_idx]
        context_img, _ = self.dataset[context_idx]
        
        return {
            'query_idx': query_idx,
            'query_image': query_img,
            'context_idx': context_idx,
            'context_image': context_img,
            'label': label,
            'context_uid': context_idx,
        }


if __name__ == "__main__":
    dataset = ImageNetRetrievalDataset(
        data_dir='/mnt/dssk/data_rw/imagenet1k', 
        split='val',
        num_contexts=5,
        subset=50,
        for_eval=True,
    )
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Number of classes: {len(dataset.class_to_indices)}")
    
    # Test a few samples
    for i in [0, 100, 1000]:
        sample = dataset[i]
        print(f"\nPair {i}:")
        print(f"  Query idx: {sample['query_idx']}, Context idx: {sample['context_idx']}")
        print(f"  Label: {sample['label']}")
        print(f"  Query image shape: {sample['query_image'].size}")
        print(f"  Context image shape: {sample['context_image'].size}")
        print(f"  Image data type: {type(sample['context_image'])}")