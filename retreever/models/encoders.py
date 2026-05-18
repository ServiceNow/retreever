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
            NotImplementedError(f"Unsupported tag value: {tag}")

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


class DummyEncoder(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(DummyEncoder, self).__init__()

    def forward(self, input_ids: torch.Tensor, *args, **kwargs):
        return input_ids


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


class CLAPTextAudioEncoder(torch.nn.Module):
    """Unified CLAP encoder for cross-modal text-audio retrieval (e.g. Clotho).

    Mirrors FlavaEncoder / HybridDistilBERTCLIPEncoder:
        - tag="question"  →  text tower:  ClapModel.get_text_features(...)  → 512D
        - tag="ctx"       →  audio tower: ClapModel.get_audio_features(...) → 512D

    The two instantiations share the same HuggingFace model weights, so no
    parameter duplication occurs as long as they are created from the same
    checkpoint.

    Audio inputs are expected to arrive as raw float32 waveforms at 48000 Hz
    (matching CLAP's native rate).  Use TextAudioRetrievalDataset with
    sample_rate=48000 so resampling is done at load time.
    """

    def __init__(
        self,
        tag: str,
        model_name: str = "laion/clap-htsat-fused",
        cache_dir: str = None,
        normalize: bool = True,
        max_length: int = 77,
        sample_rate: int = 44100,   # native sample rate of incoming audio (Clotho=44100)
        token_level: bool = False,
        *args,
        **kwargs,
    ):
        """
        Args:
            tag: "question" for text queries, "ctx" for audio contexts.
            model_name: "laion/clap-htsat-fused" or "laion/clap-htsat-unfused".
            cache_dir: HuggingFace cache directory.
            normalize: L2-normalise output embeddings.
            max_length: Max token length for text inputs.
            sample_rate: Sample rate of incoming audio tensors (default 48000).

        Models:
            clap-htsat-fused  : 112M params, 512-dim embeddings (recommended)
            clap-htsat-unfused: 138M params, 512-dim embeddings
        """
        super(CLAPTextAudioEncoder, self).__init__()

        try:
            from transformers import ClapModel, ClapProcessor
        except ImportError:
            raise ImportError(
                "CLAPTextAudioEncoder requires transformers>=4.30. "
                "Install with: pip install transformers"
            )

        self.tag = tag
        self.normalize = normalize
        self.sample_rate = sample_rate
        self.token_level = token_level
        self.prefix = ""
        self.output_size = 768  # pre-projection hidden dim of both CLAP towers

        self.model = ClapModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.processor = ClapProcessor.from_pretrained(model_name, cache_dir=cache_dir)

        if tag == "question":
            # Expose the tokenizer so the text-side collator can use it.
            self.tokenizer = self.processor.tokenizer
            self.tokenizer.model_max_length = max_length
        elif tag == "ctx":
            # ClapProcessor accepts any sampling_rate and internally resamples
            # to the 48 kHz that CLAP requires, so no manual resampling is needed.
            # print(
            #     f"CLAPTextAudioEncoder (ctx): incoming audio sample_rate={sample_rate} Hz. "
            #     f"ClapProcessor will resample to 48000 Hz internally."
            # )
            pass
        else:
            raise ValueError(f"Unknown tag '{tag}'. Use 'question' or 'ctx'.")

    def _normalize_emb(self, embs: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            import torch.nn.functional as F
            embs = F.normalize(embs, p=2, dim=-1)
        return embs

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Unified forward dispatching on self.tag.

        Both paths return 768D features (pre-projection hidden dim of each tower),
        bypassing CLAP's ClapProjectionLayer (768→512 MLP) entirely.

        For tag="question":
            input_ids      : (B, seq_len) tokenized text
            attention_mask : (B, seq_len) attention mask
            token_level=False → (B, 768)    pooler_output (tanh-projected CLS)
            token_level=True  → (B, L, 768) last_hidden_state

        For tag="ctx":
            input_ids      : (B, num_samples) float32 waveform at self.sample_rate Hz
            attention_mask : ignored
            token_level=False → (B, 768)    pooler_output (HTSAT attention pooling)
            token_level=True  → (B, T, 768) last_hidden_state
        """
        if self.tag == "question":
            text_outputs = self.model.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            if self.token_level:
                embs = text_outputs.last_hidden_state  # (B, L, 768)
            else:
                embs = text_outputs.pooler_output      # (B, 768)

        else:  # tag == "ctx"
            # input_ids here is the waveform tensor (B, num_samples) already at
            # 48000 Hz — resampling is handled by TextAudioRetrievalDataset.
            waveforms = input_ids

            if waveforms.dim() > 2:
                # -------------------------------------------------------
                # Fast path: pre-computed mel spectrograms from DataLoader
                # workers (shape B×C×F×T for fused, B×1×F×T for unfused).
                # Skip the CPU ClapProcessor call entirely — GPU is never
                # blocked by mel spectrogram computation.
                # -------------------------------------------------------
                model_dtype = next(self.model.parameters()).dtype
                input_features = waveforms.to(device=waveforms.device, dtype=model_dtype)
                # Always mark every sample as "longer" so the full batch goes
                # through CLAP's fusion path together.  This avoids BatchNorm
                # failures when only 1 sample in the batch would be "longer".
                is_longer = torch.ones(input_features.shape[0], dtype=torch.bool,
                                       device=input_features.device)
                audio_kwargs = {
                    "input_features": input_features,
                    "is_longer":      is_longer.to(device=input_features.device),
                }
                audio_outputs = self.model.audio_model(**audio_kwargs)

            else:
                # -------------------------------------------------------
                # Legacy path: raw waveforms — call ClapProcessor on CPU.
                # Slow: mel spectrogram happens on the main training thread.
                # Prefer using feature_extractor=... in TextAudioRetrievalDataset.
                # -------------------------------------------------------
                waveforms_np = waveforms.cpu().float().numpy()

                inputs = self.processor(
                    audios=waveforms_np,
                    sampling_rate=48000,
                    return_tensors="pt",
                )
                model_dtype = next(self.model.parameters()).dtype
                inputs = {k: v.to(device=waveforms.device, dtype=model_dtype if v.is_floating_point() else v.dtype)
                          for k, v in inputs.items()}
                # Override is_longer to all-True (same reason as fast path above).
                inputs["is_longer"] = torch.ones(waveforms.shape[0], dtype=torch.bool,
                                                  device=waveforms.device)
                audio_outputs = self.model.audio_model(**inputs)
            if self.token_level:
                # HTSAT last_hidden_state is (B, C, freq, time) — a 4D spatial map.
                # Reshape to (B, freq*time, C) so the tree/split_fn sees (B, T, D).
                lhs = audio_outputs.last_hidden_state   # (B, C, F, T)
                B, C, F, T = lhs.shape
                embs = lhs.permute(0, 2, 3, 1).reshape(B, F * T, C)  # (B, F*T, C)
            else:
                embs = audio_outputs.pooler_output      # (B, 768)

        return self._normalize_emb(embs)


from transformers import FlavaModel, FlavaProcessor

class FlavaEncoder(BaseEncoder):
    def __init__(
        self,
        tag: str,  # "question" for text, "ctx" for image
        model_name: str = "facebook/flava-full",
        cache_dir: str = None,
        token_level: bool = False,
        normalize: bool = True,
        max_length: int = 77,
        *args,
        **kwargs,
    ):
        """Unified FLAVA encoder for both text and images.

        Args:
            tag (str): "question" for text encoding, "ctx" for image encoding
            model_name (str): HuggingFace model name
            cache_dir (str): Cache directory
            token_level (bool): 
                - False: return pooled features (CLS token) - 768D
                - True: return per-token/patch embeddings - (seq_len/num_patches, 768)
            normalize (bool): Whether to L2-normalize embeddings
            max_length (int): Max sequence length for text
        """
        torch.nn.Module.__init__(self)
        
        self.tag = tag
        self.model = FlavaModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.processor = FlavaProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        
        # Set up tokenizer for text mode
        if tag == "question":
            self.tokenizer = self.processor.tokenizer
        
        # FLAVA has 768D for both text and vision
        self.output_size = 768
        self.token_level = token_level
        self.normalize = normalize
        self.max_length = max_length
        self.prefix = ""
        
    def forward(self, input_ids: torch.Tensor = None, pixel_values: torch.Tensor = None, 
                attention_mask: torch.Tensor = None, **kwargs):
        """Unified forward that handles both text and images based on tag.
        
        Args:
            input_ids: Tokenized text (for tag="question")
            pixel_values: Preprocessed images (for tag="ctx")
            attention_mask: Attention mask for text (for tag="question")

        Returns:
            torch.Tensor: Embeddings (768D pooled or token/patch-level)
        """
        
        if self.tag == "question":
            # Text encoding path
            text_outputs = self.model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            
            if self.token_level:
                embs = text_outputs  # Shape: (B, seq_len, 768)
            else:
                embs = text_outputs[:, 0, :]  # Shape: (B, 768) - CLS token
                
        elif self.tag == "ctx":
            # Image encoding path
            model_dtype = next(self.model.parameters()).dtype
            if input_ids.dtype != model_dtype:
                input_ids = input_ids.to(model_dtype)
            image_outputs = self.model.get_image_features(
                pixel_values=input_ids,
                return_dict=True,
            )
            
            if self.token_level:
                embs = image_outputs  # Shape: (B, num_patches + 1, 768)
            else:
                embs = image_outputs[:, 0, :]  # Shape: (B, 768) - CLS token
        else:
            raise ValueError(f"Unknown tag: {self.tag}. Must be 'question' or 'ctx'")
        
        embs = self._normalize(embs)
        return embs
    
    def encode_images(
        self,
        images: List[Union[Image.Image, str]],
        device: str = "cpu",
    ):
        """Encode images for evaluation (only works if tag="ctx").
        
        Args:
            images: List of PIL Images or paths to images
            device: Device to run inference on
        """
        if self.tag != "ctx":
            raise ValueError("encode_images only works with tag='ctx'")
        
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
        embeddings = self.forward(pixel_values=inputs["pixel_values"])
        
        return embeddings

class HybridDistilBERTCLIPEncoder(BaseEncoder):
    def __init__(
        self,
        tag: str,  # "question" for text, "ctx" for image
        model_name: str = None,  # Not used, kept for interface compatibility
        cache_dir: str = None,
        token_level: bool = False,
        normalize: bool = True,
        max_length: int = 512,
        *args,
        **kwargs,
    ):
        """Hybrid encoder using DistilBERT for text and CLIP for images.

        Args:
            tag (str): "question" for text encoding, "ctx" for image encoding
            model_name (str): Ignored, kept for interface compatibility
            cache_dir (str): Cache directory
            token_level (bool): 
                - False: return pooled features - 768D
                - True: return per-token/patch embeddings
            normalize (bool): Whether to L2-normalize embeddings
            max_length (int): Max sequence length for text
        """
        torch.nn.Module.__init__(self)
        
        self.tag = tag
        self.token_level = token_level
        self.normalize = normalize
        self.max_length = max_length
        self.prefix = ""
        
        # Both output 768D
        self.output_size = 768
        
        if tag == "question":
            # Load DistilBERT for text
            self.tokenizer = AutoTokenizer.from_pretrained(
                "sentence-transformers/msmarco-distilbert-cos-v5",
                model_max_length=max_length
            )
            self.model = AutoModel.from_pretrained(
                "sentence-transformers/msmarco-distilbert-cos-v5",
                cache_dir=cache_dir
            )
        elif tag == "ctx":
            # Load CLIP for images
            self.model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                cache_dir=cache_dir
            )
            self.processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32",
                cache_dir=cache_dir
            )
        else:
            raise ValueError(f"Unknown tag: {tag}. Must be 'question' or 'ctx'")
        
        for param in self.model.parameters():
            param.data = param.data.contiguous()
    
    def forward(self, input_ids: torch.Tensor = None, pixel_values: torch.Tensor = None,
                attention_mask: torch.Tensor = None, **kwargs):
        """Unified forward that handles both text and images based on tag.
        
        Args:
            input_ids: Tokenized text (for tag="question") or pixel values (for tag="ctx")
            pixel_values: Not used, kept for compatibility
            attention_mask: Attention mask for text (for tag="question")

        Returns:
            torch.Tensor: Embeddings (768D pooled or token/patch-level)
        """
        
        if self.tag == "question":
            # Text encoding with DistilBERT
            embs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)[0]
            
            if not self.token_level:  # Mean pooling for sentence embeddings
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(embs.size()).float()
                sum_embeddings = torch.sum(embs * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embs = sum_embeddings / sum_mask
                
        elif self.tag == "ctx":
            # Image encoding with CLIP
            vision_outputs = self.model.vision_model(
                pixel_values=input_ids,  # input_ids contains pixel_values for images
                output_hidden_states=True,
                return_dict=True,
            )
            
            if self.token_level:
                # Return all patch embeddings
                embs = vision_outputs.last_hidden_state
            else:
                # Return pooled output (CLS token)
                embs = vision_outputs.pooler_output
        
        embs = self._normalize(embs)
        model_dtype = next(self.model.parameters()).dtype
        if embs.dtype != model_dtype:
            embs = embs.to(model_dtype)
            
        return embs
    
    def encode_images(
        self,
        images: List[Union[Image.Image, str]],
        device: str = "cpu",
    ):
        """Encode images for evaluation (only works if tag="ctx").
        
        Args:
            images: List of PIL Images or paths to images
            device: Device to run inference on
        """
        if self.tag != "ctx":
            raise ValueError("encode_images only works with tag='ctx'")
        
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
        embeddings = self.forward(input_ids=inputs["pixel_values"])
        
        return embeddings


def get_encoders(encoder_type, **encoder_kwargs):
    model_name = encoder_dict[encoder_type][1]
    context_encoder = encoder_dict[encoder_type][0](
        tag="ctx", model_name=model_name, **encoder_kwargs
    )
    query_encoder = encoder_dict[encoder_type][0](
        tag="question", model_name=model_name, **encoder_kwargs
    )

    return query_encoder, context_encoder


def merge_lora_weights(checkpoint_path, query_encoder, context_encoder, lora_alpha=16, lora_r=8):
    """
    Load checkpoint with LoRA weights and merge them into the encoders.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        query_encoder: Your query encoder model
        context_encoder: Your context encoder model
        lora_alpha: LoRA alpha parameter (scaling factor)
        lora_r: LoRA rank
    """
    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    
    # Calculate LoRA scaling factor
    lora_scaling = lora_alpha / lora_r
    
    # Process each encoder
    for encoder_name, encoder_model in [('query_encoder', query_encoder), 
                                          ('context_encoder', context_encoder)]:
        print(f"\nProcessing {encoder_name}...")
        
        # Organize weights by parameter name
        base_weights = {}
        lora_a_weights = {}
        lora_b_weights = {}
        other_weights = {}
        
        # Separate different types of weights
        for key, value in state_dict.items():
            if not key.startswith(encoder_name):
                continue
                
            # Remove encoder prefix to get model-relative key
            relative_key = key[len(encoder_name) + 1:]  # +1 for the dot
            
            if '.lora_A.default.weight' in key:
                # Extract base parameter name
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
                # Regular weights without LoRA
                base_weights[relative_key] = value
        
        # Merge LoRA weights into base weights
        merged_state_dict = {}
        
        for key, base_weight in base_weights.items():
            if key in lora_a_weights and key in lora_b_weights:
                # Merge: W_merged = W_base + scaling * (B @ A)
                lora_a = lora_a_weights[key]
                lora_b = lora_b_weights[key]
                
                # Compute LoRA update: B @ A
                lora_update = (lora_b @ lora_a) * lora_scaling
                
                # Merge with base weights
                merged_weight = base_weight + lora_update
                merged_state_dict[key] = merged_weight
                print(f"  Merged LoRA for {key}: {base_weight.shape} + {lora_update.shape}")
            else:
                # No LoRA adaptation for this parameter
                merged_state_dict[key] = base_weight
        
        # Load merged weights into encoder
        # Remove 'base_model.model.' prefix if present to match encoder structure
        clean_state_dict = {}
        for key, value in merged_state_dict.items():
            if key.startswith('base_model.model.'):
                clean_key = key[len('base_model.model.'):]
                clean_state_dict[clean_key] = value
            else:
                clean_state_dict[key] = value
        
        # Load into model
        missing_keys, unexpected_keys = encoder_model.load_state_dict(clean_state_dict, strict=False)
        
        print(f"  Loaded {len(clean_state_dict)} parameters into {encoder_name}")
        if missing_keys:
            print(f"  Missing keys: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"  Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"  Unexpected keys: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"  Unexpected keys: {unexpected_keys}")
    
    print("\n✓ Successfully loaded and merged LoRA weights!")
    return query_encoder, context_encoder


class MultiModalContextEncoder(torch.nn.Module):
    """Multi-modal encoder for simultaneous text-image and text-audio retrieval.

    Used with encoder_type="multimodal-dino-ast" to train a shared tree over
    Clotho (text→audio) + COCO (text→image) + Flickr30k (text→image).

    tag="question"  → DistilBERT (768D) for all text queries.  Exposes .tokenizer.
    tag="ctx"       → DINOv2-base (768D) for images, AST (768D) for audio.
                      Both sub-encoders are always frozen.
                      forward() requires modality="image" or modality="audio".
    """

    def __init__(
        self,
        tag: str,
        model_name: str = None,          # ignored; kept for get_encoders() compat
        cache_dir: str = None,
        token_level: bool = False,
        normalize: bool = True,
        max_length: int = 512,
        image_model_name: str = "facebook/dinov2-base",
        audio_model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        audio_sample_rate: int = 48000,
        *args,
        **kwargs,
    ):
        torch.nn.Module.__init__(self)
        self.tag = tag
        self.token_level = token_level
        self.normalize = normalize
        self.prefix = ""
        self.output_size = 768  # DINOv2-base, AST, and DistilBERT all output 768D

        if tag == "question":
            self._encoder = DistilBERTEncoder(
                tag="question",
                model_name="sentence-transformers/msmarco-distilbert-cos-v5",
                cache_dir=cache_dir,
                token_level=token_level,
                normalize=normalize,
                max_length=max_length,
            )
            self.tokenizer = self._encoder.tokenizer

        elif tag == "ctx":
            self.image_encoder = DinoV2Encoder(
                tag="ctx",
                model_name=image_model_name,
                cache_dir=cache_dir,
                token_level=token_level,
                normalize=normalize,
            )
            self.audio_encoder = ASTEncoder(
                tag="ctx",
                model_name=audio_model_name,
                cache_dir=cache_dir,
                token_level=token_level,
                normalize=normalize,
                sample_rate=audio_sample_rate,
            )
            # Both sub-encoders are always frozen
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            for param in self.audio_encoder.parameters():
                param.requires_grad = False

        else:
            raise ValueError(f"Unknown tag '{tag}'. Use 'question' or 'ctx'.")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        modality: str = None,
        **kwargs,
    ) -> torch.Tensor:
        """Dispatch to text, image, or audio sub-encoder based on tag/modality.

        For tag="question":
            Standard text forward — input_ids are token ids, attention_mask is used.
        For tag="ctx":
            modality="image" → input_ids contains pixel_values → DINOv2Encoder
            modality="audio" → input_ids contains waveform tensor → ASTEncoder
        """
        if self.tag == "question":
            return self._encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
        else:  # ctx
            if modality == "image":
                return self.image_encoder(pixel_values=input_ids)
            elif modality == "audio":
                return self.audio_encoder(pixel_values=input_ids)
            else:
                raise ValueError(
                    f"MultiModalContextEncoder (ctx) requires modality='image' or "
                    f"'audio', got {modality!r}"
                )


encoder_dict = {  # supported encoder modules
    "dummy": (DummyEncoder, ""),
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
    # Text-Audio encoders (text query → audio context, e.g. Clotho)
    "clap-fused-ta": (CLAPTextAudioEncoder, "laion/clap-htsat-fused"),     # 112M, 512-dim
    "clap-unfused-ta": (CLAPTextAudioEncoder, "laion/clap-htsat-unfused"), # 138M, 512-dim
    # Text-Image encoders
    'flava': (FlavaEncoder, 'facebook/flava-full'), # ~245M params, 768-dim
    'hybrid-distilbert-clip': (HybridDistilBERTCLIPEncoder, None),  # model_name not used
    # Multi-modal: DistilBERT (text query) + DINOv2-base (image ctx) + AST (audio ctx)
    # Trains simultaneously on Clotho (text→audio) + COCO/Flickr30k (text→image)
    'multimodal-dino-ast': (MultiModalContextEncoder, None),  # 768D, both ctx sub-encoders frozen

}
