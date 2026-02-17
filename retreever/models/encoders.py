"""Minimal encoder implementations for ReTreever.

Supports text (DistilBERT, BGE), images (DinoV2, ResNet, CLIP),
audio (AST), and multimodal (FLAVA) encoders.
"""
import torch

from abc import ABC, abstractmethod
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    CLIPModel,
    CLIPProcessor,
    AutoImageProcessor,
    FlavaModel,
    FlavaProcessor,
)
from typing import Optional, List, Union
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
            model_name: Name of tokenizer to instantiate
            max_length: Maximal tokenizer's context length
            normalize: Whether returning normalized embeddings
            tag:  "question" for query encoder or "ctx" for context encoder
        """
        super(BaseEncoder, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=max_length)
        self.normalize = normalize

        if tag == "question":
            self.prefix = "question: "
        elif tag == "ctx":
            self.prefix = "context: "
        else:
            raise NotImplementedError(f"Unsupported tag value: {tag}")

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
        prefix = self.prefix if tag == "query" else ""

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
            ndims = embeddings.dim()
            if ndims == 2:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            elif ndims == 3:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=2)
            else:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        return embeddings


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
            tag: "question" for query encoder or "ctx" for context encoder
            model_name: Model path or name in HF hub
            cache_dir: Where to cache HF files
            token_level: Whether returning token embeddings or sentence embeddings
        """
        super(DistilBERTEncoder, self).__init__(model_name, max_length, normalize, tag)

        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.output_size = self.model.config.hidden_size
        self.token_level = token_level
        self.prefix = ""

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs):
        """Embed text inputs.

        Args:
            input_ids: Tokens to be encoded
            attention_mask: Attention mask

        Returns:
            Token or sentence embeddings depending on token_level flag
        """
        embs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)[0]

        if not self.token_level:  # mean pooling
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
            tag: "question" for query encoder or "ctx" for context encoder
            model_name: Model path or name in HF hub
            cache_dir: Where to cache HF files
            token_level: Whether returning token embeddings or sentence embeddings
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
        """Embed text inputs.

        Args:
            input_ids: Tokens to be encoded
            attention_mask: Attention mask

        Returns:
            Token or sentence embeddings depending on token_level flag
        """
        embs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)[0]

        if not self.token_level:  # CLS token pooling
            embs = embs[:, 0]

        embs = self._normalize(embs)
        return embs


class DinoV2Encoder(BaseEncoder):
    def __init__(
        self,
        tag: str,
        model_name: str = "facebook/dinov2-base",
        cache_dir: str = None,
        token_level: bool = False,
        normalize: bool = True,
        max_length: int = None,
        *args,
        **kwargs,
    ):
        """Wrapper of DinoV2 image encoders.

        Args:
            tag: "question" for query encoder or "ctx" for context encoder
            model_name: Model path or name in HF hub
            cache_dir: Where to cache HF files
            token_level: Whether returning patch embeddings or CLS embeddings
        """
        torch.nn.Module.__init__(self)
        
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        
        self.output_size = self.model.config.hidden_size
        self.token_level = token_level
        self.normalize = normalize
        self.prefix = ""
        
    def forward(self, pixel_values: torch.Tensor, **kwargs):
        """Embed images.

        Args:
            pixel_values: Preprocessed image tensors

        Returns:
            Patch or CLS embeddings depending on token_level flag
        """
        outputs = self.model(pixel_values=pixel_values)
        
        if self.token_level:
            embs = outputs.last_hidden_state  # All patch embeddings
        else:
            embs = outputs.last_hidden_state[:, 0, :]  # CLS token
        
        embs = self._normalize(embs)
        return embs
    
    def encode_images(
        self,
        images: List[Union[Image.Image, str]],
        device: str = "cpu",
    ):
        """Encode images for evaluation."""
        loaded_images = []
        for img in images:
            if isinstance(img, str):
                loaded_images.append(Image.open(img).convert('RGB'))
            else:
                loaded_images.append(img)
        
        inputs = self.processor(images=loaded_images, return_tensors="pt").to(device)
        embeddings = self.forward(inputs["pixel_values"])
        return embeddings


class ResNetProcessor:
    """Wrapper to make torchvision transforms compatible with HF processor interface."""
    
    def __init__(self):
        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])
    
    def __call__(self, images: List[Image.Image], return_tensors: str = "pt"):
        """Process images to match HF processor interface."""
        if not isinstance(images, list):
            images = [images]
        
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
        max_length: int = None,
        *args,
        **kwargs,
    ):
        """Wrapper of ResNet image encoders.

        Args:
            tag: "question" for query encoder or "ctx" for context encoder
            model_name: ResNet variant (resnet18, resnet34, resnet50, resnet101, resnet152)
            cache_dir: Where to cache model files
            token_level: Whether returning spatial feature maps or pooled features
        """
        torch.nn.Module.__init__(self)
        
        model_dict = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152,
        }
        
        if model_name not in model_dict:
            raise ValueError(f"Unsupported ResNet model: {model_name}. Choose from {list(model_dict.keys())}")
        
        if cache_dir is not None:
            import os
            os.environ['TORCH_HOME'] = cache_dir
            torch.hub.set_dir(cache_dir)
        
        from torchvision.models import (
            ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
            ResNet101_Weights, ResNet152_Weights
        )
        
        weights_dict = {
            'resnet18': ResNet18_Weights.IMAGENET1K_V1,
            'resnet34': ResNet34_Weights.IMAGENET1K_V1,
            'resnet50': ResNet50_Weights.IMAGENET1K_V1,
            'resnet101': ResNet101_Weights.IMAGENET1K_V1,
            'resnet152': ResNet152_Weights.IMAGENET1K_V1,
        }
        
        self.model = model_dict[model_name](weights=weights_dict[model_name])
        
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
        
        if token_level:
            self.model = torch.nn.Sequential(*list(self.model.children())[:-2])
        else:
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        
        self.processor = ResNetProcessor()
        
    def forward(self, pixel_values: torch.Tensor, **kwargs):
        """Embed images.

        Args:
            pixel_values: Preprocessed image tensors (B, 3, 224, 224)

        Returns:
            - if token_level=False: (B, output_size) - single embedding per image
            - if token_level=True: (B, num_patches, output_size) - embedding per spatial location
        """
        model_dtype = next(self.model.parameters()).dtype
        if pixel_values.dtype != model_dtype:
            pixel_values = pixel_values.to(model_dtype)
        
        embs = self.model(pixel_values)
        
        if not self.token_level:
            embs = embs.squeeze(-1).squeeze(-1)  # (B, output_size)
        else:
            B, C, H, W = embs.shape
            embs = embs.view(B, C, H * W).permute(0, 2, 1)  # (B, num_patches, output_size)
        
        embs = self._normalize(embs)
        return embs
    
    def encode_images(
        self,
        images: List[Union[Image.Image, str]],
        device: str = "cpu",
    ):
        """Encode images for evaluation."""
        loaded_images = []
        for img in images:
            if isinstance(img, str):
                loaded_images.append(Image.open(img).convert('RGB'))
            else:
                loaded_images.append(img)
        
        inputs = self.processor(images=loaded_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
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
        max_length: int = None,
        *args,
        **kwargs,
    ):
        """Wrapper of CLIP image encoders.

        Args:
            tag: "question" for query encoder or "ctx" for context encoder
            model_name: Model path or name in HF hub
            cache_dir: Where to cache HF files
            token_level: Whether returning patch embeddings or CLS embeddings
        """
        torch.nn.Module.__init__(self)
        
        self.model = CLIPModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.processor = CLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        
        self.output_size = self.model.config.vision_config.hidden_size
        self.token_level = token_level
        self.normalize = normalize
        self.prefix = ""
        
    def forward(self, pixel_values: torch.Tensor, **kwargs):
        """Embed images.

        Args:
            pixel_values: Preprocessed image tensors

        Returns:
            Patch or CLS embeddings depending on token_level flag
        """
        vision_outputs = self.model.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        
        if self.token_level:
            embs = vision_outputs.last_hidden_state
        else:
            embs = vision_outputs.pooler_output
        
        embs = self._normalize(embs)
        return embs
    
    def encode_images(
        self,
        images: List[Union[Image.Image, str]],
        device: str = "cpu",
    ):
        """Encode images for evaluation."""
        loaded_images = []
        for img in images:
            if isinstance(img, str):
                loaded_images.append(Image.open(img).convert('RGB'))
            else:
                loaded_images.append(img)
        
        inputs = self.processor(images=loaded_images, return_tensors="pt").to(device)
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
            model_name: Name of model to instantiate
            normalize: Whether returning normalized embeddings
            tag: "question" for query encoder or "ctx" for context encoder
            sample_rate: Expected sample rate for audio
        """
        super(BaseAudioEncoder, self).__init__()
        
        self.normalize = normalize
        self.sample_rate = sample_rate
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
        """Needed for MTEB-style evaluation."""
        if waveforms.dim() == 3:
            waveforms = waveforms.squeeze(1)
        
        waveforms = waveforms.to(device)
        embeddings = self.forward(waveforms)
        
        if rep_level is not None:
            return embeddings[:, :2**rep_level]
        
        return embeddings

    def _normalize(self, embeddings):
        if self.normalize:
            ndims = embeddings.dim()
            if ndims == 2:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            elif ndims == 3:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=2)
            else:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        return embeddings


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

        Args:
            tag: "question" or "ctx"
            model_name: HuggingFace model name
            cache_dir: Cache directory
            token_level: Return token-level or pooled embeddings
        """
        super(ASTEncoder, self).__init__(model_name, normalize, tag, sample_rate)
        
        try:
            from transformers import ASTModel, ASTFeatureExtractor
        except ImportError:
            raise ImportError(
                "AST requires transformers. Install with: pip install transformers"
            )
        
        self.token_level = token_level
        self.model = ASTModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.processor = ASTFeatureExtractor.from_pretrained(model_name, cache_dir=cache_dir)
        self.output_size = self.model.config.hidden_size
        
        if sample_rate != 16000:
            print(f"WARNING: AST expects 16kHz audio, but got {sample_rate}Hz. Will resample.")
    
    def forward(self, pixel_values: torch.Tensor, **kwargs):
        """Embed audio.

        Args:
            pixel_values: Audio waveforms (batch_size, num_samples)

        Returns:
            Token-level or pooled embeddings depending on token_level flag
        """
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.squeeze(1)
        
        if self.sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                self.sample_rate, 16000
            ).to(pixel_values.device)
            pixel_values = resampler(pixel_values)
        
        inputs = self.processor(
            pixel_values.cpu().float().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        model_dtype = next(self.model.parameters()).dtype
        inputs = {k: v.to(pixel_values.device).to(model_dtype) for k, v in inputs.items()}
        
        outputs = self.model(**inputs)
        
        if self.token_level:
            embs = outputs.last_hidden_state
        else:
            embs = outputs.last_hidden_state.mean(dim=1)
        
        embs = self._normalize(embs)
        return embs


class FlavaEncoder(BaseEncoder):
    def __init__(
        self,
        tag: str,
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
            tag: "question" for text encoding, "ctx" for image encoding
            model_name: HuggingFace model name
            cache_dir: Cache directory
            token_level: Return per-token/patch embeddings or pooled features (CLS token)
        """
        torch.nn.Module.__init__(self)
        
        self.tag = tag
        self.model = FlavaModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.processor = FlavaProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        
        if tag == "question":
            self.tokenizer = self.processor.tokenizer
        
        self.output_size = 768  # FLAVA has 768D for both text and vision
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
            Embeddings (768D pooled or token/patch-level)
        """
        
        if self.tag == "question":
            text_outputs = self.model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            
            if self.token_level:
                embs = text_outputs
            else:
                embs = text_outputs[:, 0, :]  # CLS token
                
        elif self.tag == "ctx":
            model_dtype = next(self.model.parameters()).dtype
            if input_ids.dtype != model_dtype:
                input_ids = input_ids.to(model_dtype)
            image_outputs = self.model.get_image_features(
                pixel_values=input_ids,
                return_dict=True,
            )
            
            if self.token_level:
                embs = image_outputs
            else:
                embs = image_outputs[:, 0, :]  # CLS token
        else:
            raise ValueError(f"Unknown tag: {self.tag}. Must be 'question' or 'ctx'")
        
        embs = self._normalize(embs)
        return embs
    
    def encode_images(
        self,
        images: List[Union[Image.Image, str]],
        device: str = "cpu",
    ):
        """Encode images for evaluation (only works if tag="ctx")."""
        if self.tag != "ctx":
            raise ValueError("encode_images only works with tag='ctx'")
        
        loaded_images = []
        for img in images:
            if isinstance(img, str):
                loaded_images.append(Image.open(img).convert('RGB'))
            else:
                loaded_images.append(img)
        
        inputs = self.processor(images=loaded_images, return_tensors="pt").to(device)
        embeddings = self.forward(pixel_values=inputs["pixel_values"])
        return embeddings


def get_encoders(encoder_type, **encoder_kwargs):
    """Get query and context encoders for a given encoder type."""
    model_name = encoder_dict[encoder_type][1]
    context_encoder = encoder_dict[encoder_type][0](
        tag="ctx", model_name=model_name, **encoder_kwargs
    )
    query_encoder = encoder_dict[encoder_type][0](
        tag="question", model_name=model_name, **encoder_kwargs
    )
    return query_encoder, context_encoder


# Supported encoder modules (minimal set)
encoder_dict = {
    # Text encoders
    "distilbert": (DistilBERTEncoder, "sentence-transformers/msmarco-distilbert-cos-v5"),
    "bge": (BGEEncoder, "BAAI/bge-large-en-v1.5"),
    # Image encoders - DinoV2
    "dinov2-small": (DinoV2Encoder, "facebook/dinov2-small"),
    "dinov2-base": (DinoV2Encoder, "facebook/dinov2-base"),
    "dinov2-large": (DinoV2Encoder, "facebook/dinov2-large"),
    "dinov2-giant": (DinoV2Encoder, "facebook/dinov2-giant"),
    # Image encoders - ResNet
    "resnet18": (ResNetEncoder, "resnet18"),
    "resnet34": (ResNetEncoder, "resnet34"),
    "resnet50": (ResNetEncoder, "resnet50"),
    "resnet101": (ResNetEncoder, "resnet101"),
    "resnet152": (ResNetEncoder, "resnet152"),
    # Image encoders - CLIP
    "clip-vit-base-patch32": (CLIPEncoder, "openai/clip-vit-base-patch32"),
    "clip-vit-base-patch16": (CLIPEncoder, "openai/clip-vit-base-patch16"),
    "clip-vit-large-patch14": (CLIPEncoder, "openai/clip-vit-large-patch14"),
    # Audio encoders
    "ast": (ASTEncoder, "MIT/ast-finetuned-audioset-10-10-0.4593"),
    # Text-Image encoders
    "flava": (FlavaEncoder, "facebook/flava-full"),
}
