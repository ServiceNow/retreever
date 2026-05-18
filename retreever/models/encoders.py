import torch

from abc import ABC, abstractmethod
from transformers import (
    BertModel,
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    CLIPModel,
    CLIPProcessor,
    AutoImageProcessor,
    Wav2Vec2Model,
    Wav2Vec2Processor,
    HubertModel,
    AutoProcessor,
    AutoModel,
    AutoConfig,
)
from typing import Optional, List, Union
from transformers import AutoImageProcessor
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
import torchaudio


class BaseEncoder(ABC, torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        max_length: int = None,
        normalize: bool = False,
        tag: str = "ctx",
    ):
        """Generic wrapper of encoders.

        Args:
            model_name (str): name of tokenizer to instantiate
            max_length (int, optional): maximal tokenizer's context length
            normalize (bool, optional): whether returning normalized embeddings. Defaults to False.
            tag (str): either "question" to load query encoder or "ctx" to load context encoder
        """
        super(BaseEncoder, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=max_length)

        self.normalize = normalize

        # set default prefixes
        if tag == "question":
            self.prefix = "question: "
        elif tag == "ctx":
            self.prefix = "context: "
        else:
            raise ValueError(f"tag must be 'question' or 'ctx', got {tag!r}")

    @abstractmethod
    def forward(self):
        pass

    def encode_sentences(
        self,
        sentences: torch.Tensor,
        tag: str = "query",
        rep_level: int = None,
        device: str = "cpu",
    ):
        """Needed for MTEB evaluation."""

        if tag == "query":
            prefix = self.prefix
        else:
            prefix = ""

        tokens = self.tokenizer(
            [prefix + s for s in sentences],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        embeddings = self.forward(
            tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            output_attentions=False,
        )

        if rep_level is not None:
            return embeddings[:, : 2**rep_level]

        return embeddings

    def _normalize(self, embeddings):
        if self.normalize:
            # normalize across embedding dimension; supports 2D (B, D) and 3D (B, T, D)
            ndims = embeddings.dim()
            if ndims == 2:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            elif ndims == 3:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=2)
            else:
                # fallback: normalize last dim
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        return embeddings


class BertEncoder(BaseEncoder):
    def __init__(
        self,
        tag: str,
        model_name: str,
        cache_dir: str = None,
        token_level: bool = False,
        normalize: bool = True,
        max_length: int = 512,
        *args,
        **kwargs,
    ):
        """Wrapper of BERT-type encoders.

        Args:
            tag (str): either "question" to load query encoder or "ctx" to load context encoder
            model_name (str, optional): model path or name in HF hub. Defaults to "bert-base-uncased".
            cache_dir (str, optional): where to cache HF files. Defaults to None.
            token_level (bool, optional): whether returning token embeddings or sentence embeddings. Defaults to False (sentence embeddings).
        """
        super(BertEncoder, self).__init__(model_name, max_length, normalize, tag)

        self.model = BertModel.from_pretrained(model_name, cache_dir=cache_dir)

        self.output_size = self.model.config.hidden_size
        self.emb_type = "last_hidden_state" if token_level else "pooler_output"

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs):
        """Use encoder to embed inputs.

        Args:
            input_ids (torch.Tensor): tokens to be encoded
            attention_mask (torch.Tensor, optional): attention mask to indicate which tokens to attend to or not. Defaults to None.
            kwargs: optional arguments to pass to the encoder

        Returns:
            str: token or sentence embeddings depending on <token_level> flag.
        """

        embs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)[
            self.emb_type
        ]
        embs = self._normalize(embs)

        return embs


class DPREncoder(BaseEncoder):
    def __init__(
        self,
        tag: str,
        model_name: str,
        cache_dir: str = None,
        token_level: bool = False,
        normalize: bool = True,
        max_length: int = 512,  # can go up to 2048
        *args,
        **kwargs,
    ):
        """Wrapper of DPR-type encoders. Loads either the question or ctx encoder.

        Args:
            tag (str): either "question" to load query encoder or "ctx" to load context encoder
            model_name (str, optional): model path or name in HF hub. Defaults to "facebook/dpr-{}_encoder-single-nq-base".
            cache_dir (str, optional): where to cache HF files. Defaults to None.
            token_level (bool, optional): whether returning token embeddings or sentence embeddings. Defaults to False (sentence embeddings).
        """
        super(DPREncoder, self).__init__(model_name.format(tag), max_length, normalize, tag)

        self.model = AutoModel.from_pretrained(model_name.format(tag), cache_dir=cache_dir)

        self.output_size = self.model.config.hidden_size
        self.output_hidden_states = token_level
        self.emb_type = "hidden_states" if token_level else "pooler_output"

        self.prefix = ""

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs):
        """Use encoder to embed inputs.

        Args:
            input_ids (torch.Tensor): tokens to be encoded
            attention_mask (torch.Tensor, optional): attention mask to indicate which tokens to attend to or not. Defaults to None.
            kwargs: optional arguments to pass to the encoder

        Returns:
            str: token or sentence embeddings depending on <token_level> flag.
        """

        embs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=self.output_hidden_states,
            **kwargs,
        )[self.emb_type]
        embs = self._normalize(embs)

        return embs
    
class DistilBERTEncoder(BaseEncoder):
    def __init__(
        self,
        tag: str,
        model_name: str,
        cache_dir: str = None,
        token_level: bool = False,
        normalize: bool = True,
        max_length: int = 512,
        *args,
        **kwargs,
    ):
        """Wrapper of DistilBERT encoders.

        Args:
            tag (str): either "question" to load query encoder or "ctx" to load context encoder
            model_name (str, optional): model path or name in HF hub.
            cache_dir (str, optional): where to cache HF files. Defaults to None.
            token_level (bool, optional): whether returning token embeddings or sentence embeddings. Defaults to False (sentence embeddings).
        """
        super(DistilBERTEncoder, self).__init__(model_name, max_length, normalize, tag)

        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

        self.output_size = self.model.config.hidden_size
        self.token_level = token_level
        self.prefix = ""

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs):
        """Use encoder to embed inputs.

        Args:
            input_ids (torch.Tensor): tokens to be encoded
            attention_mask (torch.Tensor, optional): attention mask to indicate which tokens to attend to or not. Defaults to None.
            kwargs: optional arguments to pass to the encoder

        Returns:
            torch.Tensor: token or sentence embeddings depending on <token_level> flag.
        """
        embs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)[0]

        if not self.token_level:  # mean pooling (recommended for DistilBERT sentence embeddings)
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(embs.size()).float()
            sum_embeddings = torch.sum(embs * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embs = sum_embeddings / sum_mask

        embs = self._normalize(embs)

        return embs


class ContrieverEncoder(BaseEncoder):
    def __init__(
        self,
        tag: str,
        model_name: str,
        cache_dir: str = None,
        token_level: bool = False,
        normalize: bool = True,
        max_length: int = 512,
        *args,
        **kwargs,
    ):
        """Wrapper of Contriever encoders.

        Args:
            tag (str): either "question" to load query encoder or "ctx" to load context encoder
            model_name (str, optional): model path or name in HF hub.
            cache_dir (str, optional): where to cache HF files. Defaults to None.
            token_level (bool, optional): whether returning token embeddings or sentence embeddings. Defaults to False (sentence embeddings).
        """
        super(ContrieverEncoder, self).__init__(model_name, max_length, normalize, tag)

        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

        self.output_size = self.model.config.hidden_size
        self.token_level = token_level
        self.prefix = ""

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs):
        """Use encoder to embed inputs.

        Args:
            input_ids (torch.Tensor): tokens to be encoded
            attention_mask (torch.Tensor, optional): attention mask to indicate which tokens to attend to or not. Defaults to None.
            kwargs: optional arguments to pass to the encoder

        Returns:
            torch.Tensor: token or sentence embeddings depending on <token_level> flag.
        """
        embs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)[0]

        if not self.token_level:  # mean pooling (Contriever's pooling strategy)
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(embs.size()).float()
            sum_embeddings = torch.sum(embs * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embs = sum_embeddings / sum_mask

        embs = self._normalize(embs)

        return embs

class SimCSEEncoder(BaseEncoder):
    def __init__(
        self,
        tag: str,
        model_name: str,
        cache_dir: str = None,
        token_level: bool = False,
        normalize: bool = True,
        max_length: int = 512,
        *args,
        **kwargs,
    ):
        """Wrapper of SimCSE encoders.

        Args:
            tag (str): either "question" to load query encoder or "ctx" to load context encoder
            model_name (str, optional): model path or name in HF hub.
            cache_dir (str, optional): where to cache HF files. Defaults to None.
            token_level (bool, optional): whether returning token embeddings or sentence embeddings. Defaults to False (sentence embeddings).
        """
        super(SimCSEEncoder, self).__init__(model_name, max_length, normalize, tag)

        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

        self.output_size = self.model.config.hidden_size
        self.token_level = token_level
        self.prefix = ""

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs):
        """Use encoder to embed inputs.

        Args:
            input_ids (torch.Tensor): tokens to be encoded
            attention_mask (torch.Tensor, optional): attention mask to indicate which tokens to attend to or not. Defaults to None.
            kwargs: optional arguments to pass to the encoder

        Returns:
            torch.Tensor: token or sentence embeddings depending on <token_level> flag.
        """
        embs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)[0]

        if not self.token_level:  # mean pooling (SimCSE's pooling strategy)
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(embs.size()).float()
            sum_embeddings = torch.sum(embs * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embs = sum_embeddings / sum_mask

        embs = self._normalize(embs)

        return embs

class BGEEncoder(BaseEncoder):
    def __init__(
        self,
        tag: str,
        model_name: str,
        cache_dir: str = None,
        token_level: bool = False,
        normalize: bool = True,
        max_length: int = 512,
        *args,
        **kwargs,
    ):
        """Wrapper of BGE-type encoders.

        Args:
            tag (str): either "question" to load query encoder or "ctx" to load context encoder
            model_name (str, optional): model path or name in HF hub.
            cache_dir (str, optional): where to cache HF files. Defaults to None.
            token_level (bool, optional): whether returning token embeddings or sentence embeddings. Defaults to False (sentence embeddings).
        """
        super(BGEEncoder, self).__init__(model_name, max_length, normalize, tag)

        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

        self.output_size = self.model.config.hidden_size
        self.token_level = token_level

        if tag == "question" and "bge-m3" not in model_name:
            self.prefix = "Represent this sentence for searching relevant passages: "
        else:
            self.prefix = ""

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs):
        """Use encoder to embed inputs.

        Args:
            input_ids (torch.Tensor): tokens to be encoded
            attention_mask (torch.Tensor, optional): attention mask to indicate which tokens to attend to or not. Defaults to None.
            kwargs: optional arguments to pass to the encoder

        Returns:
            str: token or sentence embeddings depending on <token_level> flag.
        """
        embs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)[0]

        if not self.token_level:  # pooling by cls tag
            embs = embs[:, 0]

        embs = self._normalize(embs)

        return embs
    
    
class LLMEncoder(BaseEncoder):
    def __init__(
        self,
        tag: str,
        model_name: str,
        normalize: bool = True,
        max_length: int = 512,
        *args,
        **kwargs,
    ):
        super(LLMEncoder, self).__init__(model_name, max_length, normalize, tag)
        self.prefix = ""
        config = AutoConfig.from_pretrained(model_name)
        self.output_size = config.hidden_size
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs):
        return input_ids
    
class DinoV2Encoder(BaseEncoder):
    def __init__(
        self,
        tag: str,
        model_name: str = "facebook/dinov2-base",
        cache_dir: str = None,
        token_level: bool = False,
        normalize: bool = True,
        max_length: int = None,  # Not used for images, but kept for interface compatibility
        *args,
        **kwargs,
    ):
        """Wrapper of DinoV2 image encoders.

        Args:
            tag (str): either "question" to load query encoder or "ctx" to load context encoder
            model_name (str, optional): model path or name in HF hub. Defaults to "facebook/dinov2-base".
            cache_dir (str, optional): where to cache HF files. Defaults to None.
            token_level (bool, optional): whether returning patch embeddings or CLS embeddings. Defaults to False (CLS embeddings).
        """
        # Initialize without calling super().__init__() since we don't need text tokenizer
        torch.nn.Module.__init__(self)
        
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        
        self.output_size = self.model.config.hidden_size
        self.token_level = token_level
        self.normalize = normalize
        self.prefix = ""
        
    def forward(self, pixel_values: torch.Tensor, **kwargs):
        """Use encoder to embed images.

        Args:
            pixel_values (torch.Tensor): preprocessed image tensors
            kwargs: optional arguments to pass to the encoder

        Returns:
            torch.Tensor: patch or CLS embeddings depending on <token_level> flag.
        """
        outputs = self.model(pixel_values=pixel_values)
        
        if self.token_level:
            # Return all patch embeddings
            embs = outputs.last_hidden_state  # Shape: (B, 1 + num_patches, hidden_size)
        else:
            # Return CLS token embedding
            embs = outputs.last_hidden_state[:, 0, :]  # Shape: (B, hidden_size)
        
        embs = self._normalize(embs)
        return embs
    
    def encode_images(
        self,
        images: List[Union[Image.Image, str]],
        device: str = "cpu",
    ):
        """Encode images for evaluation.
        
        Args:
            images: List of PIL Images or paths to images
            device: Device to run inference on
        """
        # Load images if paths are provided
        loaded_images = []
        for img in images:
            if isinstance(img, str):
                loaded_images.append(Image.open(img).convert('RGB'))
            else:
                loaded_images.append(img)
        
        # Preprocess images
        inputs = self.processor(images=loaded_images, return_tensors="pt").to(device)
        
        # Get embeddings
        embeddings = self.forward(inputs["pixel_values"])
        
        return embeddings


class ResNetProcessor:
    """Wrapper to make torchvision transforms compatible with HuggingFace processor interface."""
    
    def __init__(self):
        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])
    
    def __call__(self, images: List[Image.Image], return_tensors: str = "pt"):
        """Process images to match HuggingFace processor interface.
        
        Args:
            images: List of PIL Images
            return_tensors: Type of tensors to return (only "pt" supported)
            
        Returns:
            Dict with "pixel_values" key containing processed image tensor
        """
        if not isinstance(images, list):
            images = [images]
        
        # Apply transforms to each image
        pixel_values = torch.stack([self.transforms(img) for img in images])
        
        return {"pixel_values": pixel_values}


class ResNetEncoder(BaseEncoder):
    def __init__(
        self,
        tag: str,
        model_name: str = "resnet50",
        cache_dir: str = None,
        token_level: bool = False,
        normalize: bool = True,
        max_length: int = None,  # Not used for images, but kept for interface compatibility
        *args,
        **kwargs,
    ):
        """Wrapper of ResNet image encoders.

        Args:
            tag (str): either "question" to load query encoder or "ctx" to load context encoder
            model_name (str, optional): ResNet variant - resnet18, resnet34, resnet50, resnet101, resnet152
            cache_dir (str, optional): where to cache model files. Defaults to None.
            token_level (bool, optional): whether returning spatial feature maps or pooled features. Defaults to False (pooled features).
        """
        # Initialize without calling super().__init__() since we don't need text tokenizer
        torch.nn.Module.__init__(self)
        
        # Load ResNet model
        model_dict = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152,
        }
        
        if model_name not in model_dict:
            raise ValueError(f"Unsupported ResNet model: {model_name}. Choose from {list(model_dict.keys())}")
        
        # Load pretrained model
        import os
        if cache_dir is not None:
            # Set torch hub directory for model downloads
            os.environ['TORCH_HOME'] = cache_dir
            torch.hub.set_dir(cache_dir)
        
        # Use weights parameter instead of deprecated pretrained
        from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
        
        weights_dict = {
            'resnet18': ResNet18_Weights.IMAGENET1K_V1,
            'resnet34': ResNet34_Weights.IMAGENET1K_V1,
            'resnet50': ResNet50_Weights.IMAGENET1K_V1,
            'resnet101': ResNet101_Weights.IMAGENET1K_V1,
            'resnet152': ResNet152_Weights.IMAGENET1K_V1,
        }
        
        self.model = model_dict[model_name](weights=weights_dict[model_name])
        
        # Get output size based on model architecture
        output_sizes = {
            'resnet18': 512,
            'resnet34': 512,
            'resnet50': 2048,
            'resnet101': 2048,
            'resnet152': 2048,
        }
        self.output_size = output_sizes[model_name]
        
        self.token_level = token_level
        self.normalize = normalize
        self.prefix = ""
        
        # Remove the final fully connected layer for feature extraction
        if token_level:
            # Keep everything up to avg pool to get spatial features
            # Shape will be: (B, C, H, W) where H, W are spatial dimensions (e.g., 7x7)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-2])
        else:
            # Keep everything up to final pooling
            # Shape will be: (B, C, 1, 1)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        
        # Use custom processor that matches HuggingFace interface
        self.processor = ResNetProcessor()
        
    def forward(self, pixel_values: torch.Tensor, **kwargs):
        """Use encoder to embed images.

        Args:
            pixel_values (torch.Tensor): preprocessed image tensors
                Shape: (B, 3, 224, 224)

        Returns:
            torch.Tensor: 
                - if token_level=False: (B, output_size) - single embedding per image
                - if token_level=True: (B, num_patches, output_size) - embedding per spatial location
        """
        model_dtype = next(self.model.parameters()).dtype
        if pixel_values.dtype != model_dtype:
            pixel_values = pixel_values.to(model_dtype)
        
        embs = self.model(pixel_values)
        
        if not self.token_level:
            # After avgpool, we get: (B, output_size, 1, 1)
            # We want: (B, output_size) - a single embedding vector per image
            embs = embs.squeeze(-1).squeeze(-1)  # Remove the 1x1 spatial dims
            # Result: (B, output_size)
        else:
            # After conv layers, we get: (B, output_size, H, W) e.g., (B, 2048, 7, 7)
            # We want: (B, H*W, output_size) - one embedding per spatial location (patch)
            # This matches transformer format: (B, num_patches, hidden_dim)
            B, C, H, W = embs.shape  # e.g., (B, 2048, 7, 7)
            embs = embs.view(B, C, H * W)  # (B, 2048, 49) - flatten spatial
            embs = embs.permute(0, 2, 1)   # (B, 49, 2048) - swap to match transformer format
            # Result: (B, num_patches, hidden_dim)
        
        embs = self._normalize(embs)
        return embs
    
    def encode_images(
        self,
        images: List[Union[Image.Image, str]],
        device: str = "cpu",
    ):
        """Encode images for evaluation.
        
        Args:
            images: List of PIL Images or paths to images
            device: Device to run inference on
        """
        # Load images if paths are provided
        loaded_images = []
        for img in images:
            if isinstance(img, str):
                loaded_images.append(Image.open(img).convert('RGB'))
            else:
                loaded_images.append(img)
        
        # Preprocess images using the processor
        inputs = self.processor(images=loaded_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        
        # Get embeddings
        embeddings = self.forward(pixel_values)
        
        return embeddings

class CLIPEncoder(BaseEncoder):
    def __init__(
        self,
        tag: str,
        model_name: str = "openai/clip-vit-base-patch32",
        cache_dir: str = None,
        token_level: bool = False,
        normalize: bool = True,
        max_length: int = None,  # Not used for images, but kept for interface compatibility
        *args,
        **kwargs,
    ):
        """Wrapper of CLIP image encoders.

        Args:
            tag (str): either "question" to load query encoder or "ctx" to load context encoder
            model_name (str, optional): model path or name in HF hub. Defaults to "openai/clip-vit-base-patch32".
            cache_dir (str, optional): where to cache HF files. Defaults to None.
            token_level (bool, optional): whether returning patch embeddings or CLS embeddings. Defaults to False (CLS embeddings).
        """
        # Initialize without calling super().__init__() since we don't need text tokenizer
        torch.nn.Module.__init__(self)
        
        self.model = CLIPModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.processor = CLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        
        self.output_size = self.model.config.vision_config.hidden_size
        self.token_level = token_level
        self.normalize = normalize
        self.prefix = ""
        
    def forward(self, pixel_values: torch.Tensor, **kwargs):
        """Use encoder to embed images.

        Args:
            pixel_values (torch.Tensor): preprocessed image tensors
            kwargs: optional arguments to pass to the encoder

        Returns:
            torch.Tensor: patch or CLS embeddings depending on <token_level> flag.
        """
        # Get vision model outputs
        vision_outputs = self.model.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        
        if self.token_level:
            # Return all patch embeddings (excluding CLS token)
            # last_hidden_state shape: (B, num_patches + 1, hidden_size)
            embs = vision_outputs.last_hidden_state  # Shape: (B, num_patches + 1, hidden_size)
        else:
            # Return pooled output (CLS token after projection)
            embs = vision_outputs.pooler_output  # Shape: (B, hidden_size)
        
        embs = self._normalize(embs)
        return embs
    
    def encode_images(
        self,
        images: List[Union[Image.Image, str]],
        device: str = "cpu",
    ):
        """Encode images for evaluation.
        
        Args:
            images: List of PIL Images or paths to images
            device: Device to run inference on
        """
        # Load images if paths are provided
        loaded_images = []
        for img in images:
            if isinstance(img, str):
                loaded_images.append(Image.open(img).convert('RGB'))
            else:
                loaded_images.append(img)
        
        # Preprocess images
        inputs = self.processor(images=loaded_images, return_tensors="pt").to(device)
        
        # Get embeddings
        embeddings = self.forward(inputs["pixel_values"])
        
        return embeddings
    

class BaseAudioEncoder(ABC, torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        normalize: bool = False,
        tag: str = "ctx",
        sample_rate: int = 16000,
    ):
        """Generic wrapper of audio encoders.

        Args:
            model_name (str): name of model to instantiate
            normalize (bool, optional): whether returning normalized embeddings. Defaults to False.
            tag (str): either "question" to load query encoder or "ctx" to load context encoder
            sample_rate (int): Expected sample rate for audio
        """
        super(BaseAudioEncoder, self).__init__()
        
        self.normalize = normalize
        self.sample_rate = sample_rate
        
        # Audio encoders typically don't use prefixes
        self.prefix = ""

    @abstractmethod
    def forward(self):
        pass

    def encode_audios(
        self,
        waveforms: torch.Tensor,
        tag: str = "query",
        rep_level: int = None,
        device: str = "cpu",
    ):
        """Needed for MTEB-style evaluation.
        
        Args:
            waveforms: Tensor of shape (batch_size, num_samples) or (batch_size, 1, num_samples)
            tag: "query" or "context"
            rep_level: If specified, return only first 2**rep_level dimensions
            device: Device to run on
        """
        # Ensure correct shape: (batch_size, num_samples)
        if waveforms.dim() == 3:
            waveforms = waveforms.squeeze(1)
        
        waveforms = waveforms.to(device)
        
        embeddings = self.forward(waveforms)
        
        if rep_level is not None:
            return embeddings[:, :2**rep_level]
        
        return embeddings

    def _normalize(self, embeddings):
        if self.normalize:
            # normalize across embedding dimension; supports 2D (B, D) and 3D (B, T, D)
            ndims = embeddings.dim()
            if ndims == 2:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            elif ndims == 3:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=2)
            else:
                # fallback: normalize last dim
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        
        return embeddings


class Wav2Vec2Encoder(BaseAudioEncoder):
    def __init__(
        self,
        tag: str,
        model_name: str = "facebook/wav2vec2-base-960h",
        cache_dir: str = None,
        token_level: bool = False,
        normalize: bool = True,
        sample_rate: int = 16000,
        *args,
        **kwargs,
    ):
        """Wrapper of Wav2Vec2 audio encoders.

        Args:
            tag (str): either "question" to load query encoder or "ctx" to load context encoder
            model_name (str, optional): model path or name in HF hub
            cache_dir (str, optional): where to cache HF files
            token_level (bool, optional): whether returning frame embeddings or utterance embeddings
            normalize (bool, optional): whether to normalize embeddings
            sample_rate (int): Expected sample rate
        """
        super(Wav2Vec2Encoder, self).__init__(model_name, normalize, tag, sample_rate)
        
        self.model = Wav2Vec2Model.from_pretrained(model_name, cache_dir=cache_dir)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=cache_dir)
        
        self.output_size = self.model.config.hidden_size
        self.token_level = token_level
        
    def forward(self, pixel_values: torch.Tensor, **kwargs):
        """Use encoder to embed audio.

        Args:
            pixel_values (torch.Tensor): audio waveforms of shape (batch_size, num_samples)
            kwargs: optional arguments to pass to the encoder

        Returns:
            torch.Tensor: frame or utterance embeddings depending on <token_level> flag.
        """
        # Process audio through Wav2Vec2 processor
        # The processor normalizes the waveform
        inputs = self.processor(
            pixel_values.cpu().numpy() if pixel_values.is_cuda else pixel_values.numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        
        # Move inputs to same device as model
        inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}
        
        # Get model outputs
        outputs = self.model(**inputs, **kwargs)
        
        if self.token_level:
            # Return all frame embeddings
            embs = outputs.last_hidden_state  # Shape: (batch, time, hidden_size)
        else:
            # Return mean-pooled utterance embedding
            embs = outputs.last_hidden_state.mean(dim=1)  # Shape: (batch, hidden_size)
        
        embs = self._normalize(embs)
        return embs


class HuBERTEncoder(BaseAudioEncoder):
    def __init__(
        self,
        tag: str,
        model_name: str = "facebook/hubert-base-ls960",
        cache_dir: str = None,
        token_level: bool = False,
        normalize: bool = True,
        sample_rate: int = 16000,
        *args,
        **kwargs,
    ):
        """Wrapper of HuBERT audio encoders.

        Args:
            tag (str): either "question" to load query encoder or "ctx" to load context encoder
            model_name (str, optional): model path or name in HF hub
            cache_dir (str, optional): where to cache HF files
            token_level (bool, optional): whether returning frame embeddings or utterance embeddings
            normalize (bool, optional): whether to normalize embeddings
            sample_rate (int): Expected sample rate
        """
        super(HuBERTEncoder, self).__init__(model_name, normalize, tag, sample_rate)
        
        self.model = HubertModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        
        self.output_size = self.model.config.hidden_size
        self.token_level = token_level
        
    def forward(self, pixel_values: torch.Tensor, **kwargs):
        """Use encoder to embed audio.

        Args:
            pixel_values (torch.Tensor): audio waveforms of shape (batch_size, num_samples)
            kwargs: optional arguments to pass to the encoder

        Returns:
            torch.Tensor: frame or utterance embeddings depending on <token_level> flag.
        """
        # Process audio
        inputs = self.processor(
            pixel_values.cpu().numpy() if pixel_values.is_cuda else pixel_values.numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        
        # Move inputs to same device as model
        inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}
        
        # Get model outputs
        outputs = self.model(**inputs, **kwargs)
        
        if self.token_level:
            # Return all frame embeddings
            embs = outputs.last_hidden_state  # Shape: (batch, time, hidden_size)
        else:
            # Return mean-pooled utterance embedding
            embs = outputs.last_hidden_state.mean(dim=1)  # Shape: (batch, hidden_size)
        
        embs = self._normalize(embs)
        return embs


class WavLMEncoder(BaseAudioEncoder):
    def __init__(
        self,
        tag: str,
        model_name: str = "microsoft/wavlm-base-plus",
        cache_dir: str = None,
        token_level: bool = False,
        normalize: bool = True,
        sample_rate: int = 16000,
        *args,
        **kwargs,
    ):
        """Wrapper of WavLM audio encoders.

        Args:
            tag (str): either "question" to load query encoder or "ctx" to load context encoder
            model_name (str, optional): model path or name in HF hub
            cache_dir (str, optional): where to cache HF files
            token_level (bool, optional): whether returning frame embeddings or utterance embeddings
            normalize (bool, optional): whether to normalize embeddings
            sample_rate (int): Expected sample rate
        """
        super(WavLMEncoder, self).__init__(model_name, normalize, tag, sample_rate)
        
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        
        self.output_size = self.model.config.hidden_size
        self.token_level = token_level
        
    def forward(self, pixel_values: torch.Tensor, **kwargs):
        """Use encoder to embed audio.

        Args:
            pixel_values (torch.Tensor): audio waveforms of shape (batch_size, num_samples)
            kwargs: optional arguments to pass to the encoder

        Returns:
            torch.Tensor: frame or utterance embeddings depending on <token_level> flag.
        """
        # Process audio
        inputs = self.processor(
            pixel_values.cpu().numpy() if pixel_values.is_cuda else pixel_values.numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        
        # Move inputs to same device as model
        inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}
        
        # Get model outputs
        outputs = self.model(**inputs, **kwargs)
        
        if self.token_level:
            # Return all frame embeddings
            embs = outputs.last_hidden_state  # Shape: (batch, time, hidden_size)
        else:
            # Return mean-pooled utterance embedding
            embs = outputs.last_hidden_state.mean(dim=1)  # Shape: (batch, hidden_size)
        
        embs = self._normalize(embs)
        return embs


class BEATsEncoder(BaseAudioEncoder):
    def __init__(
        self,
        tag: str,
        model_name: str = "microsoft/beats-base",
        cache_dir: str = None,
        token_level: bool = False,
        normalize: bool = True,
        sample_rate: int = 16000,
        *args,
        **kwargs,
    ):
        """Wrapper of BEATs audio encoders.

        Args:
            tag (str): either "question" to load query encoder or "ctx" to load context encoder
            model_name (str, optional): model path or name in HF hub
            cache_dir (str, optional): where to cache HF files
            token_level (bool, optional): whether returning frame embeddings or utterance embeddings
            normalize (bool, optional): whether to normalize embeddings
            sample_rate (int): Expected sample rate
        """
        super(BEATsEncoder, self).__init__(model_name, normalize, tag, sample_rate)
        
        # Note: BEATs might not be in transformers yet, you may need to use the original repo
        # For now, I'll use a similar interface assuming it will be available
        try:
            self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
            self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
        except:
            print("WARNING: BEATs model not found in transformers. Please install from the official repo:")
            print("https://github.com/microsoft/unilm/tree/master/beats")
            raise
        
        self.output_size = self.model.config.hidden_size
        self.token_level = token_level
        
    def forward(self, pixel_values: torch.Tensor, **kwargs):
        """Use encoder to embed audio.

        Args:
            pixel_values (torch.Tensor): audio waveforms of shape (batch_size, num_samples)
            kwargs: optional arguments to pass to the encoder

        Returns:
            torch.Tensor: frame or utterance embeddings depending on <token_level> flag.
        """
        # Process audio
        inputs = self.processor(
            pixel_values.cpu().numpy() if pixel_values.is_cuda else pixel_values.numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        
        # Move inputs to same device as model
        inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}
        
        # Get model outputs
        outputs = self.model(**inputs, **kwargs)
        
        if self.token_level:
            # Return all frame embeddings
            embs = outputs.last_hidden_state  # Shape: (batch, time, hidden_size)
        else:
            # Return mean-pooled utterance embedding
            embs = outputs.last_hidden_state.mean(dim=1)  # Shape: (batch, hidden_size)
        
        embs = self._normalize(embs)
        return embs


class PANNsEncoder(BaseAudioEncoder):
    def __init__(
        self,
        tag: str,
        model_name: str = "Cnn14",
        cache_dir: str = None,
        normalize: bool = True,
        sample_rate: int = 32000,
        *args,
        **kwargs,
    ):
        """Wrapper of PANNs (Pretrained Audio Neural Networks) audio encoders.
        
        PANNs are CNNs trained on AudioSet for general audio classification.
        Best for environmental sounds, music, and non-speech audio.
        
        Args:
            tag (str): "question" or "ctx"
            model_name (str): One of ["Cnn6", "Cnn10", "Cnn14", "Cnn14_16k"]
            cache_dir (str): Cache directory for model weights
            normalize (bool): Whether to normalize embeddings
            sample_rate (int): Expected sample rate (32000 for most PANNs, 16000 for Cnn14_16k)
        
        Model sizes:
            - Cnn6: 4.5M params, 512-dim embeddings
            - Cnn10: 4.9M params, 512-dim embeddings
            - Cnn14: 79.6M params, 2048-dim embeddings (best, default)
            - Cnn14_16k: Same as Cnn14 but for 16kHz audio
        
        Requires: pip install panns-inference
        """
        super(PANNsEncoder, self).__init__(model_name, normalize, tag, sample_rate)
        
        try:
            from panns_inference import AudioTagging
        except ImportError:
            raise ImportError(
                "PANNs requires panns-inference. Install with: pip install panns-inference"
            )
        
         # Initialize PANNs model with cache directory
        import os
        if cache_dir:
            os.environ['TORCH_HOME'] = cache_dir
            
        # Initialize PANNs model
        self.model = AudioTagging(
            checkpoint_path=None,  # Will download automatically
            device='cpu'  # We'll handle device placement externally
        )
        self.model.model_name = model_name
        
        # Set output size based on model
        if model_name in ["Cnn6", "Cnn10"]:
            self.output_size = 512
        elif model_name in ["Cnn14", "Cnn14_16k"]:
            self.output_size = 2048
        else:
            raise ValueError(f"Unknown PANNs model: {model_name}")
        
        # PANNs expects 32kHz (or 16kHz for Cnn14_16k)
        if model_name == "Cnn14_16k":
            self.expected_sample_rate = 16000
        else:
            self.expected_sample_rate = 32000
        
        if sample_rate != self.expected_sample_rate:
            print(f"WARNING: PANNs {model_name} expects {self.expected_sample_rate}Hz, "
                  f"but got {sample_rate}Hz. Will resample.")
    
    def forward(self, pixel_values: torch.Tensor, token_level: bool = False):
        """
        Args:
            pixel_values: (batch_size, num_samples) tensor
            token_level: If True, return frame-level features; else return clip-level
        
        Returns:
            embeddings: (batch_size, output_size) tensor
        """
        # Ensure 2D tensor
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.squeeze(1)
        
        # Resample if needed
        if self.sample_rate != self.expected_sample_rate:
            resampler = torchaudio.transforms.Resample(
                self.sample_rate, self.expected_sample_rate
            ).to(pixel_values.device)
            pixel_values = resampler(pixel_values)
        
        # PANNs expects numpy arrays
        waveforms_np = pixel_values.cpu().numpy()
        
        # Get embeddings (returns dict with 'clipwise_output' and 'embedding')
        batch_embeddings = []
        for i in range(len(waveforms_np)):
            with torch.no_grad():
                result = self.model.inference(waveforms_np[i:i+1])
            batch_embeddings.append(torch.from_numpy(result['embedding']))
        
        embs = torch.cat(batch_embeddings, dim=0).to(pixel_values.device)
        embs = self._normalize(embs)
        return embs


class ASTEncoder(BaseAudioEncoder):
    def __init__(
        self,
        tag: str,
        model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        cache_dir: str = None,
        token_level: bool = False,
        normalize: bool = True,
        sample_rate: int = 16000,
        *args,
        **kwargs,
    ):
        """Wrapper of Audio Spectrogram Transformer (AST) encoders.
        
        AST applies Vision Transformer to audio spectrograms.
        Trained on AudioSet for general audio classification.
        
        Args:
            tag (str): "question" or "ctx"
            model_name (str): HuggingFace model name
            cache_dir (str): Cache directory
            token_level (bool): Return token-level or pooled embeddings
            normalize (bool): Whether to normalize embeddings
            sample_rate (int): Expected sample rate (16000)
        
        Model: 88M params, 768-dim embeddings
        Requires: pip install transformers
        """
        super(ASTEncoder, self).__init__(model_name, normalize, tag, sample_rate)
        
        try:
            from transformers import ASTModel, ASTFeatureExtractor
        except ImportError:
            raise ImportError(
                "AST requires transformers. Install with: pip install transformers"
            )
        
        self.token_level = token_level
        
        # Load model and processor
        self.model = ASTModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.processor = ASTFeatureExtractor.from_pretrained(model_name, cache_dir=cache_dir)
        
        # Set output size
        self.output_size = self.model.config.hidden_size  # 768
        
        if sample_rate != 16000:
            print(f"WARNING: AST expects 16kHz audio, but got {sample_rate}Hz. Will resample.")
    
    def forward(self, pixel_values: torch.Tensor, **kwargs):
        """
        Args:
            pixel_values: (batch_size, num_samples) tensor
            token_level: If True, return frame-level features; else return pooled
        
        Returns:
            embeddings: (batch_size, output_size) or (batch_size, num_frames, output_size)
        """
        # Ensure 2D tensor
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.squeeze(1)
        
        # Resample if needed
        if self.sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                self.sample_rate, 16000
            ).to(pixel_values.device)
            pixel_values = resampler(pixel_values)
        
        # Process with AST processor
        inputs = self.processor(
            pixel_values.cpu().float().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        # Match model dtype
        model_dtype = next(self.model.parameters()).dtype
        inputs = {k: v.to(pixel_values.device).to(model_dtype) for k, v in inputs.items()}
        # inputs = {k: v.to(pixel_values.device) for k, v in inputs.items()}
        
        # Forward pass
        outputs = self.model(**inputs)
        
        if self.token_level:
            # Return all tokens (batch_size, num_patches, hidden_size)
            embs = outputs.last_hidden_state
        else:
            # Return mean pooling over tokens
            embs = outputs.last_hidden_state.mean(dim=1)
        
        embs = self._normalize(embs)
        return embs


class CLAPEncoder(BaseAudioEncoder):
    def __init__(
        self,
        tag: str,
        model_name: str = "laion/clap-htsat-fused",
        cache_dir: str = None,
        normalize: bool = True,
        sample_rate: int = 48000,
        *args,
        **kwargs,
    ):
        """Wrapper of CLAP (Contrastive Language-Audio Pretraining) encoders.
        
        CLAP learns joint embeddings of audio and text.
        Can be used for audio-only retrieval or zero-shot classification.
        
        Args:
            tag (str): "question" or "ctx"
            model_name (str): One of ["laion/clap-htsat-fused", "laion/clap-htsat-unfused"]
            cache_dir (str): Cache directory
            normalize (bool): Whether to normalize embeddings
            sample_rate (int): Expected sample rate (48000)
        
        Models:
            - clap-htsat-fused: 112M params, 512-dim embeddings (recommended)
            - clap-htsat-unfused: 138M params, 512-dim embeddings
        
        Requires: pip install transformers
        """
        super(CLAPEncoder, self).__init__(model_name, normalize, tag, sample_rate)
        
        try:
            from transformers import ClapModel, ClapProcessor
        except ImportError:
            raise ImportError(
                "CLAP requires transformers>=4.30. Install with: pip install transformers"
            )
        
        # Load model and processor
        self.model = ClapModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.processor = ClapProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        
        # Set output size
        self.output_size = self.model.config.projection_dim  # 512
        
        if sample_rate != 48000:
            print(f"WARNING: CLAP expects 48kHz audio, but got {sample_rate}Hz. Will resample.")
    
    def forward(self, pixel_values: torch.Tensor):
        """
        Args:
            pixel_values: (batch_size, num_samples) tensor
        
        Returns:
            embeddings: (batch_size, 512) tensor
        """
        # Ensure 2D tensor
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.squeeze(1)
        
        # Resample if needed
        if self.sample_rate != 48000:
            resampler = torchaudio.transforms.Resample(
                self.sample_rate, 48000
            ).to(pixel_values.device)
            pixel_values = resampler(pixel_values)
        
        # Process with CLAP processor
        inputs = self.processor(
            audios=pixel_values.cpu().numpy(),
            sampling_rate=48000,
            return_tensors="pt"
        )
        inputs = {k: v.to(pixel_values.device) for k, v in inputs.items()}
        
        # Get audio embeddings
        audio_embeds = self.model.get_audio_features(**inputs)
        
        embs = self._normalize(audio_embeds)
        return embs


def get_encoders(encoder_type, **encoder_kwargs):
    """Build a (query_encoder, context_encoder) pair for the given encoder_type.

    ``encoder_type`` must be a key in :data:`encoder_dict`.  Extra keyword
    arguments are forwarded to the encoder constructor.
    """
    model_name = encoder_dict[encoder_type][1]
    context_encoder = encoder_dict[encoder_type][0](
        tag="ctx", model_name=model_name, **encoder_kwargs
    )
    query_encoder = encoder_dict[encoder_type][0](
        tag="question", model_name=model_name, **encoder_kwargs
    )
    return query_encoder, context_encoder


def merge_lora_weights(checkpoint_path, query_encoder, context_encoder, lora_alpha=16, lora_r=8):
    """Load a checkpoint with LoRA weights and merge them into the encoders."""
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    lora_scaling = lora_alpha / lora_r

    for encoder_name, encoder_model in [
        ('query_encoder', query_encoder),
        ('context_encoder', context_encoder),
    ]:
        print(f"\nProcessing {encoder_name}...")
        base_weights = {}
        lora_a_weights = {}
        lora_b_weights = {}

        for key, value in state_dict.items():
            if not key.startswith(encoder_name):
                continue
            relative_key = key[len(encoder_name) + 1:]  # +1 for the dot
            if '.lora_A.default.weight' in key:
                base_key = relative_key.replace('.lora_A.default.weight', '.weight')
                lora_a_weights[base_key] = value
            elif '.lora_B.default.weight' in key:
                base_key = relative_key.replace('.lora_B.default.weight', '.weight')
                lora_b_weights[base_key] = value
            elif '.base_layer.weight' in key:
                base_key = relative_key.replace('.base_layer.weight', '.weight')
                base_weights[base_key] = value
            elif '.base_layer.bias' in key:
                base_key = relative_key.replace('.base_layer.bias', '.bias')
                base_weights[base_key] = value
            else:
                base_weights[relative_key] = value

        merged_state_dict = {}
        for key, base_weight in base_weights.items():
            if key in lora_a_weights and key in lora_b_weights:
                lora_update = (lora_b_weights[key] @ lora_a_weights[key]) * lora_scaling
                merged_state_dict[key] = base_weight + lora_update
                print(f"  Merged LoRA for {key}: {base_weight.shape}")
            else:
                merged_state_dict[key] = base_weight

        # Drop the "base_model.model." prefix that PEFT adds, so keys match the
        # raw encoder's parameter names.
        clean_state_dict = {
            (k[len('base_model.model.'):] if k.startswith('base_model.model.') else k): v
            for k, v in merged_state_dict.items()
        }

        missing_keys, unexpected_keys = encoder_model.load_state_dict(
            clean_state_dict, strict=False
        )
        print(f"  Loaded {len(clean_state_dict)} parameters into {encoder_name}")
        if missing_keys:
            print(f"  Missing keys (first 5): {missing_keys[:5]}")
        if unexpected_keys:
            print(f"  Unexpected keys (first 5): {unexpected_keys[:5]}")

    print("\nSuccessfully loaded and merged LoRA weights.")
    return query_encoder, context_encoder


encoder_dict = {  # supported encoder modules
    "bert": (BertEncoder, "bert-base-uncased"),
    "bge": (BGEEncoder, "BAAI/bge-large-en-v1.5"),
    "bgem3": (BGEEncoder, "BAAI/bge-m3"),
    "dpr": (DPREncoder, "facebook/dpr-{}_encoder-single-nq-base"),
    "distilbert_msmarco": (DistilBERTEncoder, "sentence-transformers/msmarco-distilbert-cos-v5"),
    "contriever_msmarco": (ContrieverEncoder, "facebook/contriever-msmarco"), 
    "simcse": (SimCSEEncoder, "princeton-nlp/sup-simcse-bert-base-uncased"),
    # "llm": (LLMEncoder, "meta-llama/Meta-Llama-3-8B-Instruct"),
    "llm": (LLMEncoder, "meta-llama/Llama-3.2-1B-Instruct"),
    "dinov2-small": (DinoV2Encoder, "facebook/dinov2-small"),
    "dinov2-base": (DinoV2Encoder, "facebook/dinov2-base"),
    "dinov2-large": (DinoV2Encoder, "facebook/dinov2-large"),
    "dinov2-giant": (DinoV2Encoder, "facebook/dinov2-giant"),
    # ResNet encoders
    "resnet18": (ResNetEncoder, "resnet18"),
    "resnet34": (ResNetEncoder, "resnet34"),
    "resnet50": (ResNetEncoder, "resnet50"),
    "resnet101": (ResNetEncoder, "resnet101"),
    "resnet152": (ResNetEncoder, "resnet152"),
    # CLIP encoders
    "clip-vit-base-patch32": (CLIPEncoder, "openai/clip-vit-base-patch32"),
    "clip-vit-base-patch16": (CLIPEncoder, "openai/clip-vit-base-patch16"),
    "clip-vit-large-patch14": (CLIPEncoder, "openai/clip-vit-large-patch14"),
    # audio
    "wav2vec2-base": (Wav2Vec2Encoder, "facebook/wav2vec2-base-960h"),
    "wav2vec2-large": (Wav2Vec2Encoder, "facebook/wav2vec2-large-960h"),
    "wav2vec2-xlsr": (Wav2Vec2Encoder, "facebook/wav2vec2-large-xlsr-53"),
    "hubert-base": (HuBERTEncoder, "facebook/hubert-base-ls960"),
    "hubert-large": (HuBERTEncoder, "facebook/hubert-large-ll60k"),
    "wavlm-base": (WavLMEncoder, "microsoft/wavlm-base"),
    "wavlm-base-plus": (WavLMEncoder, "microsoft/wavlm-base-plus"),
    "wavlm-large": (WavLMEncoder, "microsoft/wavlm-large"),
    # General audio models (best for ESC-50, environmental sounds)
    "panns-cnn6": (PANNsEncoder, "Cnn6"),           # 4.5M params, 512-dim
    "panns-cnn10": (PANNsEncoder, "Cnn10"),         # 4.9M params, 512-dim
    "panns-cnn14": (PANNsEncoder, "Cnn14"),         # 79.6M params, 2048-dim (best)
    "panns-cnn14-16k": (PANNsEncoder, "Cnn14_16k"), # Same as cnn14, 16kHz version
    
    "ast": (ASTEncoder, "MIT/ast-finetuned-audioset-10-10-0.4593"),  # 88M params, 768-dim
    
    "clap-fused": (CLAPEncoder, "laion/clap-htsat-fused"),     # 112M params, 512-dim (recommended)
    "clap-unfused": (CLAPEncoder, "laion/clap-htsat-unfused"), # 138M params, 512-dim
}
