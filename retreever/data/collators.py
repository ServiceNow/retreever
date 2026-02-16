import torch

from transformers import PreTrainedTokenizerBase, DefaultDataCollator
from typing import List, Dict
from transformers import AutoImageProcessor


def concatenate_ctxs(example):
    return " ".join(
        [
            example["contexts_list"][i]
            for i, flag in enumerate(example["useful_contexts"])
            if flag == 1
        ]
    )


def single_context(example):
    return example["context"]


class SupervisedCollator(DefaultDataCollator):
    def __init__(
        self,
        query_tokenizer: PreTrainedTokenizerBase,
        ctx_tokenizer: PreTrainedTokenizerBase,
        query_prefix: str,
        ctx_prefix: str,
        context_field: str = "contexts_list",
        query_truncate_from_left: bool = False, 
    ) -> None:
        """Creates instance of collator.

        Args:
            query_tokenizer (PreTrainedTokenizerFast): Tokenizer used to convert query to token ids.
            ctx_tokenizer (PreTrainedTokenizerFast): Tokenizer used to convert ctx to token ids.
            query_prefix (str): prefix to prepend to query
            ctx_prefix (str): prefix to prepend to context
        """

        super(SupervisedCollator, self).__init__(
            return_tensors="pt",
        )
        self.query_tokenizer = query_tokenizer
        self.ctx_tokenizer = ctx_tokenizer
        self.query_prefix = query_prefix
        self.ctx_prefix = ctx_prefix
        self.query_truncate_from_left = query_truncate_from_left
        
        if query_truncate_from_left:
            self.query_tokenizer.truncation_side = "left"
        else:
            self.query_tokenizer.truncation_side = "right"

        self.output_keys = [
            "question_ids",
            "context_ids",
            "question_attn_mask",
            "context_attn_mask",
            "context_uid",
        ]

        if context_field == "contexts_list":
            self.get_context = concatenate_ctxs
        elif context_field == "context":
            self.get_context = single_context
        else:
            raise NotImplementedError("Unknown context_field", context_field)

    def __call__(self, batch_list: List[Dict]) -> Dict[str, torch.Tensor]:
        """Maps list of dicts that must contain the 'question' and the 'context' keys (they might contain also 'answers' and 'id').

        Args:
            batch_list (List[Dict]): List of examples.

        Returns:
            Dict[str, torch.Tensor]: Batches of tensors for 'question_ids', 'context_ids', 'question_attn_mask', 'context_attn_mask'.
        """
        
        if isinstance(batch_list, dict):
            batch_list = [batch_list]

        # format and tokenize question
        question_tokens = self.query_tokenizer(
            [self.query_prefix + example["question"] for example in batch_list],
            return_tensors=self.return_tensors,
            padding=True,
            truncation=True,
        )

        # format and tokenize context
        context_tokens = self.ctx_tokenizer(
            [self.ctx_prefix + self.get_context(example) for example in batch_list],
            return_tensors=self.return_tensors,
            padding=True,
            truncation=True,
        )

        batch = {
            "question_ids": question_tokens["input_ids"],
            "question_attn_mask": question_tokens["attention_mask"],
            "context_ids": context_tokens["input_ids"],
            "context_attn_mask": context_tokens["attention_mask"],
        }
        try:
            batch["context_uid"] = torch.tensor([example["context_uid"] for example in batch_list])
        except KeyError:
            batch["context_uid"] = torch.tensor([example["context_id"] for example in batch_list])

        return batch
    

class ImageSupervisedCollator:
    def __init__(
        self,
        query_processor,
        ctx_processor,
        image_mappings = None,
    ) -> None:
        """Creates instance of collator for image data.

        Args:
            query_processor: Image processor for query images
            ctx_processor: Image processor for context images
        """
        self.query_processor = query_processor
        self.ctx_processor = ctx_processor
        
        self.image_mappings = (
            {ex["image_idx"]: ex["img"] for ex in image_mappings}
            if image_mappings is not None else None
        )        
        
        self.output_keys = [
            "question_ids",  # Will contain pixel_values
            "context_ids",   # Will contain pixel_values
            "question_attn_mask",
            "context_attn_mask",
            "context_uid",
            'label',
        ]

    # def __call__(self, batch_list: List[Dict]) -> Dict[str, torch.Tensor]:
    #     """Maps list of dicts that contain 'query_image' and 'context_image' keys."""
        
    #     if isinstance(batch_list, dict):
    #         batch_list = [batch_list]

    #     # Extract PIL Images
    #     query_images = [self.image_mappings[example["query_idx"]] for example in batch_list]
    #     context_images = [self.image_mappings[example["context_idx"]] for example in batch_list]
        
    #     # Process images
    #     query_inputs = self.query_processor(images=query_images, return_tensors="pt")
    #     context_inputs = self.ctx_processor(images=context_images, return_tensors="pt")

    #     batch = {
    #         "question_ids": query_inputs["pixel_values"],
    #         "context_ids": context_inputs["pixel_values"],
    #         "question_attn_mask": None,
    #         "context_attn_mask": None,
    #     }

    #     # Add context_uid
    #     if "context_uid" in batch_list[0]:
    #         batch["context_uid"] = torch.tensor([ex["context_uid"] for ex in batch_list])
    #     else:
    #         batch["context_uid"] = torch.tensor([ex["context_idx"] for ex in batch_list])

    #     # Add labels if present
    #     if "label" in batch_list[0]:
    #         batch["label"] = torch.tensor([example["label"] for example in batch_list])
    #     # if "coarse_label" in batch_list[0]:
    #     #     batch["coarse_label"] = torch.tensor([example["coarse_label"] for example in batch_list])

    #     return batch
    
    def __call__(self, batch_list: List[Dict]) -> Dict[str, torch.Tensor]:
        """Maps list of dicts that contain 'query_image' and 'context_image' keys."""
        
        if isinstance(batch_list, dict):
            batch_list = [batch_list]

        # Images already in batch (ImageNet dataset)
        query_images = [example["query_image"] for example in batch_list]
        context_images = [example["context_image"] for example in batch_list]
       
        # Process images
        query_inputs = self.query_processor(images=query_images, return_tensors="pt")
        context_inputs = self.ctx_processor(images=context_images, return_tensors="pt")

        batch = {
            "question_ids": query_inputs["pixel_values"],
            "context_ids": context_inputs["pixel_values"],
            "question_attn_mask": None,
            "context_attn_mask": None,
            "context_uid": torch.tensor([ex["context_uid"] for ex in batch_list]),
            "label": torch.tensor([example["label"] for example in batch_list])
        }

        return batch


class AudioSupervisedCollator:
    def __init__(
        self,
        query_processor,
        ctx_processor,
    ) -> None:
        """Creates instance of collator for audio data.
        
        This collator has the exact same interface as ImageSupervisedCollator,
        but processes audio waveforms instead of images.

        Args:
            query_processor: Audio processor for query audio (can be None for raw waveforms)
            ctx_processor: Audio processor for context audio (can be None for raw waveforms)
        """
        self.query_processor = query_processor
        self.ctx_processor = ctx_processor
        
        self.output_keys = [
            "question_ids",  # Will contain audio waveforms or processed features
            "context_ids",   # Will contain audio waveforms or processed features
            "question_attn_mask",  # Not used for audio but kept for interface compatibility
            "context_attn_mask",   # Not used for audio but kept for interface compatibility
            "context_uid",
            'label',
        ]

    def __call__(self, batch_list: List[Dict]) -> Dict[str, torch.Tensor]:
        """Maps list of dicts that contain 'query_audio' and 'context_audio' keys.
        
        This method signature matches ImageSupervisedCollator exactly.
        
        Args:
            batch_list: List of dictionaries with keys:
                - 'query_audio': tensor of shape (1, num_samples) or (num_samples,)
                - 'context_audio': tensor of shape (1, num_samples) or (num_samples,)
                - 'context_uid': unique identifier for the context
                - 'label': speaker label
                
        Returns:
            Dictionary with keys matching ImageSupervisedCollator output:
                - 'question_ids': tensor of shape (batch_size, num_samples) or processed features
                - 'context_ids': tensor of shape (batch_size, num_samples) or processed features
                - 'question_attn_mask': None (kept for interface compatibility)
                - 'context_attn_mask': None (kept for interface compatibility)
                - 'context_uid': tensor of shape (batch_size,)
                - 'label': tensor of shape (batch_size,)
        """
        
        if isinstance(batch_list, dict):
            batch_list = [batch_list]

        # Extract audio waveforms from batch
        # Audio is already loaded by the dataset
        query_audios = [example["query_audio"] for example in batch_list]
        context_audios = [example["context_audio"] for example in batch_list]
        
        # Stack into batch tensors
        # Ensure shape is (batch_size, num_samples) by squeezing channel dimension if present
        query_batch = torch.stack([
            audio.squeeze(0) if audio.dim() > 1 else audio 
            for audio in query_audios
        ])
        context_batch = torch.stack([
            audio.squeeze(0) if audio.dim() > 1 else audio 
            for audio in context_audios
        ])
        
        # If processors are provided, use them (though for raw waveforms we typically don't need this)
        # The audio encoders will handle their own preprocessing
        if self.query_processor is not None:
            # Some processors might be callable
            query_inputs = query_batch
        else:
            query_inputs = query_batch
            
        if self.ctx_processor is not None:
            context_inputs = context_batch
        else:
            context_inputs = context_batch

        # Create batch dictionary matching ImageSupervisedCollator interface
        batch = {
            "question_ids": query_inputs,    # Shape: (batch_size, num_samples)
            "context_ids": context_inputs,    # Shape: (batch_size, num_samples)
            "question_attn_mask": None,       # Not used for audio, but kept for interface compatibility
            "context_attn_mask": None,        # Not used for audio, but kept for interface compatibility
            "context_uid": torch.tensor([ex["context_uid"] for ex in batch_list]),
            "label": torch.tensor([example["label"] for example in batch_list])
        }

        return batch
    
    

class TextImageSupervisedCollator:
    """
    Collator for text-image retrieval.
    
    Signature matches ImageSupervisedCollator and AudioSupervisedCollator.
    - Queries are text (tokenized)
    - Contexts are images (processed)
    """
    def __init__(
        self,
        query_tokenizer: PreTrainedTokenizerBase,
        ctx_processor,  # Image processor
        query_prefix: str = "",
        ctx_prefix: str = "",
        query_truncate_from_left: bool = False,
    ) -> None:
        """
        Creates instance of collator for text-image retrieval.

        Args:
            query_tokenizer: Tokenizer for text queries
            ctx_processor: Image processor for context images
            query_prefix: prefix to prepend to text query
            ctx_prefix: Not used for images, kept for interface compatibility
            query_truncate_from_left: Whether to truncate text from left
        """
        self.query_tokenizer = query_tokenizer
        self.ctx_processor = ctx_processor
        self.query_prefix = query_prefix
        self.ctx_prefix = ctx_prefix  # Not used but kept for compatibility
        
        # Set truncation side for query tokenizer
        if query_truncate_from_left:
            self.query_tokenizer.truncation_side = "left"
        else:
            self.query_tokenizer.truncation_side = "right"
        
        self.output_keys = [
            "question_ids",      # Will contain tokenized text
            "context_ids",       # Will contain pixel_values
            "question_attn_mask",
            "context_attn_mask",
            "context_uid",
        ]

    def __call__(self, batch_list: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Maps list of dicts that contain 'question' (text) and 'context_image' (PIL Image).
        
        Args:
            batch_list: List of dictionaries with keys:
                - 'question': text string (caption)
                - 'context_image': PIL Image
                - 'context_uid': unique identifier for the context
                
        Returns:
            Dictionary with keys:
                - 'question_ids': tensor of tokenized text, shape (batch_size, seq_len)
                - 'context_ids': tensor of pixel values, shape (batch_size, channels, height, width)
                - 'question_attn_mask': attention mask for text, shape (batch_size, seq_len)
                - 'context_attn_mask': None (not used for images)
                - 'context_uid': tensor of shape (batch_size,)
        """
        
        if isinstance(batch_list, dict):
            batch_list = [batch_list]

        # Extract text queries and images
        text_queries = [self.query_prefix + example["question"] for example in batch_list]
        context_images = [example["context_image"] for example in batch_list]
        
        # Tokenize text queries
        query_tokens = self.query_tokenizer(
            text_queries,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        
        # Process images
        context_inputs = self.ctx_processor(images=context_images, return_tensors="pt")

        # Create batch dictionary matching the standard interface
        batch = {
            "question_ids": query_tokens["input_ids"],
            "question_attn_mask": query_tokens["attention_mask"],
            "context_ids": context_inputs["pixel_values"],
            "context_attn_mask": None,  # Not used for images
            "context_uid": torch.tensor([ex["context_uid"] for ex in batch_list]),
        }

        return batch

class ClusteringCollator(DefaultDataCollator):
    def __init__(
        self,
    ) -> None:
        """Creates instance of collator for clustering toy settings."""

        super(ClusteringCollator, self).__init__(
            return_tensors="pt",
        )

        self.output_keys = [
            "question_ids",
            "context_ids",
            "question_attn_mask",
            "context_attn_mask",
            "context_uid",
        ]

    def __call__(self, batch_list: List[Dict]) -> Dict[str, torch.Tensor]:
        """Maps list of dicts that must contain 'context', 'centroid' and 'centroid_id' keys.

        Args:
            batch_list (List[Dict]): List of examples.

        Returns:
            Dict[str, torch.Tensor]: Batches of tensors for 'question_ids', 'context_ids', 'question_attn_mask', 'context_attn_mask', "context_uid".
        """

        batch = {
            "question_ids": torch.stack([example["centroid"] for example in batch_list]),
            "question_attn_mask": None,
            "context_ids": torch.stack([example["context"] for example in batch_list]),
            "context_attn_mask": None,
            "sample_idx": torch.tensor([example["centroid_id"] for example in batch_list]),
        }

        return batch
