
import faiss
import torch
import numpy as np

from collections import OrderedDict

from retreever.models.encoders import get_encoders


class FaissInnerProductRetriever(torch.nn.Module):
    def __init__(
        self,
        encoder_type: str = "bert",
        cache_dir: str = None,
        to_device: str = "cpu",
        use_ivf: bool = False,
        n_clusters: int = 100,
        **module_params,
    ):
        super(FaissInnerProductRetriever, self).__init__()

        self.encoder_type = encoder_type
        self.use_ivf = use_ivf
        self.n_clusters = n_clusters

        # get encoder parameters
        token_level_enc = module_params.pop("encoder_token_level", False)
        normalize_emb = module_params.pop(
            "encoder_normalize", True
        )  # default to cosine similarity

        self.query_encoder, self.context_encoder = get_encoders(
            self.encoder_type,
            cache_dir=cache_dir,
            token_level=token_level_enc,
            normalize=normalize_emb,
        )

        if module_params["rep_level"] is not None:
            self.rep_size = min(2 ** module_params["rep_level"], self.context_encoder.model.config.hidden_size)
        else:
            self.rep_size = self.context_encoder.model.config.hidden_size

        # self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.rep_size))
        if use_ivf:
            print("Using IVF index with Faiss")
            quantizer = faiss.IndexFlatIP(self.rep_size)
            self.index = faiss.IndexIVFFlat(quantizer, self.rep_size, n_clusters, faiss.METRIC_INNER_PRODUCT)
            self.index = faiss.IndexIDMap2(self.index)
            self.is_trained = False
        else:
            self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.rep_size))
            self.is_trained = True
            
        print(f"Faiss index initialized with dimension {self.rep_size}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if to_device == "cuda":
            print("Moving Faiss index and models to GPU")
            ngpus = faiss.get_num_gpus()
            print("Number of GPUs available: ", ngpus)
            resources = [faiss.StandardGpuResources() for i in range(ngpus)]
            self.index = faiss.index_cpu_to_gpu_multiple_py(resources, self.index)
            print(f"Faiss index moved to {ngpus} GPU(s)")

        self.query_encoder = self.query_encoder.to(self.device)
        self.context_encoder = self.context_encoder.to(self.device)

        self.reset_index()

    def reset_index(self):
        self.index.reset()
        self.empty_index = True
        if self.use_ivf:
            self.is_trained = False

    def index_ctxs(
        self,
        context_ids: torch.Tensor,
        context_attn_mask: torch.Tensor,
        context_names: torch.Tensor,
    ):
        """Populates the index matrix that maps a context index to its embedding.
        Supports iterative calls to populate index batch by batch.

        Args:
            context_ids (torch.Tensor): tokens to index
            context_attn_mask (torch.Tensor): mask indicating which tokens to ignore
            context_names (torch.Tensor): unique context identifiers
        """
        with torch.no_grad():
            self.embeddings = self.context_encoder(
                context_ids.to(self.device),
                attention_mask=context_attn_mask.to(self.device),
            )
            
        embeddings_np = self.embeddings.cpu().numpy()[:, :self.rep_size] #.astype(np.float32)
        context_names_np = context_names.cpu().numpy() #.astype(np.int64)
        
        # Ensure C-contiguous for FAISS
        embeddings_np = np.ascontiguousarray(embeddings_np, dtype=np.float32)
        context_names_np = np.ascontiguousarray(context_names_np, dtype=np.int64)
        
        # Train IVF if needed
        if self.use_ivf and not self.is_trained:
            if not hasattr(self, "_training_buffer_embs"):
                self._training_buffer_embs = []
                self._training_buffer_ids = []

            self._training_buffer_embs.append(embeddings_np)
            self._training_buffer_ids.append(context_names_np)

            total_samples = sum(arr.shape[0] for arr in self._training_buffer_embs)
            
            if total_samples >= self.n_clusters * 40:
                train_embs = np.vstack(self._training_buffer_embs)
                train_ids  = np.concatenate(self._training_buffer_ids)
                
                print(f"Training IVF with {train_embs.shape[0]} samples...")
                self.index.train(train_embs)
                self.is_trained = True
                
                # Add all buffered points to IVF index
                self.index.add_with_ids(train_embs, train_ids)
                self.empty_index = False
                
                del self._training_buffer_embs
                del self._training_buffer_ids
                return

        # Add to index if trained (or flat index)
        if self.is_trained:
            self.index.add_with_ids(embeddings_np, context_names_np)
            self.empty_index = False

        # self.index.add_with_ids(
        #     self.embeddings.cpu().numpy()[:, : self.rep_size], context_names.cpu().numpy()
        # )
        # self.empty_index = False
        
    def set_search_percentage(self, percentage: float):
        """Set IVF search percentage."""
        if self.use_ivf:
            self.search_percentage = percentage
            n_probe = max(1, int(self.n_clusters * percentage))
            ps = faiss.ParameterSpace()
            ps.set_index_parameter(self.index, "nprobe", n_probe)

    def top_contexts(
        self,
        question_ids: torch.Tensor,
        question_attn_mask: torch.Tensor,
        k: int = 100,
    ):
        """Based on the index, returns the top-k contexts with the highest dot products with the query.

        Args:
            question_ids (_type_): tensor of tokens to route and assign to the leaves, of shape (num_questions, seq_length)
            question_attn_mask (_type_): mask indicating which tokens to ignore, of shape (num_questions, seq_length)
            k (int, optional): number of contexts to return. Defaults to 100.

        Returns:
            (list): list of per question's set of top-k contexts.
        """
        assert (
            not self.empty_index
        ), "Need to populate index with contexts first by calling self.index_ctxs()."

        assert k > 0, "invalid number of top contexts"

        with torch.no_grad():
            q_embeddings = self.query_encoder(
                question_ids.to(self.device),
                attention_mask=question_attn_mask.to(self.device),
            )

        if k > self.index.ntotal:
            k = self.index.ntotal
            
        topk_scores, topk_ids = self.index.search(
            q_embeddings.cpu().numpy()[:, : self.rep_size], k
        )

        # if self.use_ivf:
        topk = [
            # key: question index, values: tuple of embedding and score
            OrderedDict(
                (id, (emb, score))
                for emb, id, score in zip(ids, ids, scores)
            )
            for ids, scores in zip(topk_ids, topk_scores)
        ]
        # else:
        #     topk = [
        #         # key: question index, values: tuple of embedding and score
        #         OrderedDict(
        #             (id, (emb, score))
        #             for emb, id, score in zip(self.index.reconstruct_batch(ids), ids, scores)
        #         )
        #         for ids, scores in zip(topk_ids, topk_scores)
        #     ]

        return topk
