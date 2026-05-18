# adapted from https://github.com/bigcode-project/bigcode-encoder/blob/master/src/utils.py#L42

import torch
import torch.distributed as dist
import torch.nn.functional as F

from typing import List

"""Utilities doing the kind of things that could have been a torch utility.

If it is small, could be in many models, and deals with tensors, then it probably belongs here.
If it gets too big, then give it its own file.
"""


class AllGather(torch.autograd.Function):
    """
    all_gather with gradient back-propagation
    Adapted from https://github.com/Lightning-AI/lightning-bolts/blob/5577453a6d7072724d9ae24184daf8f45d4baff7/pl_bolts/models/self_supervised/simclr/simclr_module.py#L20-L40
    """

    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]

        dist.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        dist.all_reduce(grad_input, op=dist.ReduceOp.SUM, async_op=False)
        # grad_input = grad_input / dist.get_world_size()

        idx_from = dist.get_rank() * ctx.batch_size
        idx_to = (dist.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]


all_gather = AllGather.apply


class TempCoef(torch.nn.Module):
    """Module wrapping a temperature coefficient used to compute contrastive losses."""

    def __init__(self, initial_value: float = 1.0, max_tmp: float = 30.0) -> None:
        """Constructs TempCoef instance.

        Args:
            initial_value (float): Startting value of the temperature.
        """
        super().__init__()
        self.temp_coef = torch.nn.Parameter(torch.log(torch.Tensor([initial_value])))
        self.max_tmp = max_tmp

    def forward(self, logits_matrix: torch.Tensor) -> torch.Tensor:
        """Forward pass of the module: Multiply input tensor by the temperature value.

        Args:
            logits_matrix (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Logits matrix multiplied by temp.
        """
        # Apply learnable temperature factor on similarities
        # Clamping after op to avoid numerical instabilities
        logits_matrix = logits_matrix * torch.exp(self.temp_coef).clamp(1e-4, self.max_tmp)

        return logits_matrix

    def get_temp_coef(self) -> float:
        """Get temperature value.

        Returns:
            float: temperature value.
        """
        return torch.exp(self.temp_coef).data.detach()


def matrix_neg_cross_entropy(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-10):
    """Computes negative Cross Entropy between any pred_i and target_j.
        (Pytorch cross_entropy computes it element-wise)

    Args:
        pred (torch.Tensor): predicted logits (expected size [N, C])
        target (torch.Tensor): target logits (expected size [N, C])
    Returns:
        torch.Tensor: [N, N] negative cross entropy matrix
    """

    lognorm_pred = torch.log(pred / pred.sum(1, keepdim=True) + eps)
    norm_target = target / (target.sum(1, keepdim=True) + eps)

    return lognorm_pred @ norm_target.T


def cosine_similarity_matrix(pred: torch.Tensor, target: torch.Tensor):
    """Computes cosine similarity between any pred_i and target_j.

    Args:
        pred (torch.Tensor): predicted embeddings (expected size [N, C])
        target (torch.Tensor): target embeddings (expected size [N, C])
    Returns:
        torch.Tensor: [N, N] cosine similarity matrix
    """

    pred = torch.nn.functional.normalize(pred, p=2, dim=1)
    target = torch.nn.functional.normalize(target, p=2, dim=1)

    return torch.mm(pred, target.transpose(0, 1))


def total_variation_distance_matrix(pred, target):
    """
    Computes the pairwise inverse Total Variation Distance between batches of predicted and target distributions.
    Args:
        pred (torch.Tensor): Predicted distributions of shape (B, N).
        target (torch.Tensor): Target distributions of shape (B, N).
    Returns:
        torch.Tensor: A matrix of shape (B, B) where each element [i, j] represents the inverse TVD between pred_dist[i] and target_dist[j].
    """
    # Ensure the distributions are normalized
    pred = pred / pred.sum(dim=1, keepdim=True)
    target = target / target.sum(dim=1, keepdim=True)

    # memory-friendly version
    return -0.5 * torch.cdist(pred.float(), target.float(), p=1)


class ContrastiveLoss(torch.nn.Module):
    def __init__(
        self,
        init_tmp: float = 1.0,
        local_loss: bool = False,
        freeze_tmp: bool = False,
        sim_measure: str = "ce",
        optimize_prior_levels_too: bool = False,
        optimize_token_loss: bool = False,
        encoder_dim: int = 1024,
        use_siglip_loss: bool = False,
        max_tmp: float = 30.0,
        use_separate_temps: bool = False,
        max_tmp_qc: float = None, 
        max_tmp_cq: float = None,
        use_depth_wise_temps: bool = False, 
        optimize_whole_emb: bool = False,
    ):
        """Contrastive loss based on custom similarity between question and context representations.

        Args:
            init_tmp (float, optional): initial value for temperature parameter
            local_loss (bool, optional): If set, contrastive loss will only use data in current device.
            freeze_tmp (bool, optional): If set, temperature is not optimized
            sim_measure (str, optional): The similarity measure to use. Default: "ce" for cross entropy between categoricals

        """
        super(ContrastiveLoss, self).__init__()

        # self.temp_param = TempCoef(init_tmp, max_tmp)  # Module wrapping trainable temperature parameter.

        # if freeze_tmp:
        #     self.temp_param.temp_coef.requires_grad = False
        self.use_separate_temps = use_separate_temps
        self.use_separate_temps = use_separate_temps
        self.use_depth_wise_temps = use_depth_wise_temps
        self.optimize_whole_emb = optimize_whole_emb
    
        if self.use_depth_wise_temps:
            import math
            depth_wise_max_temps = {str(x): math.exp(x/2) for x in range(12)}
            
            self.depth_wise_temps = torch.nn.ModuleDict()
            for depth, max_tmp_val in depth_wise_max_temps.items():
                self.depth_wise_temps[str(depth)] = TempCoef(init_tmp, max_tmp_val)
                if freeze_tmp:
                    self.depth_wise_temps[str(depth)].temp_coef.requires_grad = False
            
            # Store default for unseen depths
            self.default_max_tmp = max_tmp
            
        elif self.use_separate_temps:
            # Use separate temperatures for q->c and c->q directions
            max_tmp_qc = max_tmp_qc if max_tmp_qc is not None else max_tmp
            max_tmp_cq = max_tmp_cq if max_tmp_cq is not None else max_tmp
            
            self.temp_param_qc = TempCoef(init_tmp, max_tmp_qc)
            self.temp_param_cq = TempCoef(init_tmp, max_tmp_cq)
            
            if freeze_tmp:
                self.temp_param_qc.temp_coef.requires_grad = False
                self.temp_param_cq.temp_coef.requires_grad = False
        else:
            # Use single shared temperature (backward compatible)
            self.temp_param = TempCoef(init_tmp, max_tmp)
    
            if freeze_tmp:
                self.temp_param.temp_coef.requires_grad = False

        self.local_loss = local_loss
        self.sim_measure = sim_measure
        self.optimize_prior_levels_too = optimize_prior_levels_too
        
        self.optimize_token_loss = optimize_token_loss
        self.encoder_dim = encoder_dim
        
        self.use_siglip_loss = use_siglip_loss
        if self.use_siglip_loss:
            # SigLIP-specific: learnable bias parameter
            self.siglip_bias = torch.nn.Parameter(torch.tensor(-10.0), requires_grad=True)

    def __call__(
        self,
        q_embs: torch.Tensor,
        c_embs: torch.Tensor,
        depth: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> torch.Tensor:
        """Computes contrastive loss based on custom similarity between question and context representations.

        Args:
            q_embs (torch.Tensor): question representations (e.g. leaf assignment).
            c_embs (torch.Tensor): positive context representations
        Returns:
            torch.Tensor: Contrastive loss.
        """
        _, d = q_embs.shape
        
        depth = depth.item()
        if self.optimize_prior_levels_too:
            start_index = 0
            end_index = 2 ** (depth + 1) - 1
        elif self.optimize_token_loss:
            start_index = d - self.encoder_dim
            end_index = d
        elif self.optimize_whole_emb:
            start_index = 0
            end_index = q_embs.shape[-1]
        else:
            start_index = 2 ** depth - 1
            end_index = 2 ** (depth + 1) - 1

        if self.local_loss:
            q_embs = q_embs[:, start_index:end_index]
            c_embs= c_embs[:, start_index:end_index]
            q_embs_all, c_embs_all = q_embs, c_embs
        else:
            q_embs = q_embs[:, start_index:end_index]
            c_embs= c_embs[:, start_index:end_index]
            # Gathers categoricals across devices.
            q_embs_all = all_gather(q_embs.contiguous())
            c_embs_all = all_gather(c_embs.contiguous())
            # q_embs_all = q_embs_all[:, start_index:end_index]
            # c_embs_all= c_embs_all[:, start_index:end_index]
            
        

        # Compute Similarity between probability distributions
        qc_nce_matrix = sim_measure_dict[self.sim_measure](q_embs, c_embs_all)
        
        if self.use_siglip_loss:
            # Only one matrix needed for SigLIP loss, return here directly.
            return self._compute_siglip_loss(qc_nce_matrix)
        
        cq_nce_matrix = sim_measure_dict[self.sim_measure](c_embs, q_embs_all)

        # Multiply by learnable temperature
        # self.temp_param = self.temp_param.to(qc_nce_matrix.device)
        # qc_nce_matrix = self.temp_param(qc_nce_matrix)
        # cq_nce_matrix = self.temp_param(cq_nce_matrix)
        # Multiply by learnable temperature
        if self.use_depth_wise_temps:
            depth_key = str(depth)
            temp_module = self.depth_wise_temps[depth_key]
            temp_module = temp_module.to(qc_nce_matrix.device)
            qc_nce_matrix = temp_module(qc_nce_matrix)
            cq_nce_matrix = temp_module(cq_nce_matrix)
            
        elif self.use_separate_temps:
            self.temp_param_qc = self.temp_param_qc.to(qc_nce_matrix.device)
            self.temp_param_cq = self.temp_param_cq.to(cq_nce_matrix.device)
            qc_nce_matrix = self.temp_param_qc(qc_nce_matrix)
            cq_nce_matrix = self.temp_param_cq(cq_nce_matrix)
        else:
            self.temp_param = self.temp_param.to(qc_nce_matrix.device)
            qc_nce_matrix = self.temp_param(qc_nce_matrix)
            cq_nce_matrix = self.temp_param(cq_nce_matrix)

        # Matching representations of positive pairs located on the diagonal of the matrix
        positive_idx = torch.arange(qc_nce_matrix.size(0)).long().to(qc_nce_matrix.device)
        if not self.local_loss:
            positive_idx = positive_idx + dist.get_rank() * qc_nce_matrix.size(0)

        # We use a cross-entropy criterion to encourage positive query and context to be routed similarly
        # and negative query and context to be routed differently
        # We follow CLIP and apply the loss across columns (queries) and rows (contexts).
        loss = 0.5 * (
            F.cross_entropy(qc_nce_matrix, positive_idx)
            + F.cross_entropy(cq_nce_matrix, positive_idx)
        )

        return loss
    
    def _compute_siglip_loss(
        self, 
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SigLIP loss following Algorithm 1 from the paper https://arxiv.org/abs/2303.15343.
        
        Paper Algorithm:
        1. t = exp(t_prime)  # temperature (handled by TempCoef)
        2. logits = similarity * t + b  # scale and shift
        3. labels = 2 * eye(n) - ones(n)  # +1 diagonal, -1 elsewhere
        4. loss = -sum(log_sigmoid(labels * logits)) / n
        
        IMPORTANT: Unlike InfoNCE, SigLIP only needs ONE matrix because the loss
        is symmetric. The matrix encodes both image->text and text->image directions.
        
        Args:
            logits: [batch_size, num_all] similarity matrix
            
        Returns:
            SigLIP loss value
        """
        batch_size = logits.size(0)
        num_all = logits.size(1)
        
        # # Move temp_param to correct device
        # self.temp_param = self.temp_param.to(logits.device)
        
        # # Step 1 & 2: Apply temperature scaling then add bias
        # # TempCoef.forward() multiplies by exp(log_temp) and clamps
        # logits = self.temp_param(logits) + self.siglip_bias
        # Move temp_param to correct device
        if self.use_separate_temps:
            self.temp_param_qc = self.temp_param_qc.to(logits.device)
            logits = self.temp_param_qc(logits) + self.siglip_bias
        else:
            self.temp_param = self.temp_param.to(logits.device)
            logits = self.temp_param(logits) + self.siglip_bias
        
        # Step 3: Create labels: 2 * eye(n) - ones(n)
        # +1 for positive pairs (diagonal), -1 for negatives
        labels = -torch.ones(batch_size, num_all, device=logits.device, dtype=logits.dtype)

        if self.local_loss or batch_size == num_all:
            # Local training: diagonal is at [i, i]
            labels.fill_diagonal_(1.0)
        else:
            # Distributed training: offset diagonal by rank
            offset = dist.get_rank() * batch_size
            num_valid = min(batch_size, num_all - offset)
            indices = torch.arange(num_valid, device=labels.device)
            labels[indices, indices + offset] = 1.0
                
        # Step 4: Compute loss = -sum(log_sigmoid(labels * logits)) / n
        # Sum over ALL entries in the matrix (both directions implicitly)
        loss = -F.logsigmoid(labels * logits).sum() / batch_size
        
        return loss
    

class MultiLabelContrastiveLoss(torch.nn.Module):
    def __init__(
        self,
        init_tmp: float = 1.0,
        local_loss: bool = False,
        freeze_tmp: bool = False,
        sim_measure: str = "ce",
        optimize_prior_levels_too: bool = False,
        max_tmp: float = 30.0,
        optimize_whole_emb: bool = False,
    ):
        """Contrastive loss for multi-label classification where multiple samples share the same label.

        Args:
            init_tmp (float): Initial temperature parameter
            local_loss (bool): If True, only use data on current device
            freeze_tmp (bool): If True, don't optimize temperature
            sim_measure (str): Similarity measure to use (e.g., "ce" for negative TVD)
            optimize_prior_levels_too (bool): If True, optimize all tree levels
            max_tmp (float): Maximum temperature value
        """
        super(MultiLabelContrastiveLoss, self).__init__()

        self.temp_param = TempCoef(init_tmp, max_tmp)
        
        if freeze_tmp:
            self.temp_param.temp_coef.requires_grad = False

        self.local_loss = local_loss
        self.sim_measure = sim_measure
        self.optimize_prior_levels_too = optimize_prior_levels_too
        self.optimize_whole_emb = optimize_whole_emb

    def __call__(
        self,
        q_embs: torch.Tensor,
        c_embs: torch.Tensor,
        depth: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Computes multi-label contrastive loss.

        Args:
            q_embs (torch.Tensor): Query representations [B, D]
            c_embs (torch.Tensor): Context representations [B, D]
            labels (torch.Tensor): Class labels [B], samples with same label are positives
            depth (torch.Tensor): Current tree depth (for level selection)

        Returns:
            torch.Tensor: Multi-label contrastive loss
        """
        _, d = q_embs.shape
        
        depth = depth.item()
        if self.optimize_prior_levels_too:
            start_index = 0
            end_index = 2 ** (depth + 1) - 1
        elif self.optimize_whole_emb:
            start_index = 0
            end_index = q_embs.shape[-1]
        else:
            start_index = 2 ** depth - 1
            end_index = 2 ** (depth + 1) - 1

        # Select tree level
        if self.local_loss:
            q_embs = q_embs[:, start_index:end_index]
            c_embs = c_embs[:, start_index:end_index]
            q_embs_all, c_embs_all = q_embs, c_embs
            labels_all = labels
        else:
            q_embs = q_embs[:, start_index:end_index]
            c_embs = c_embs[:, start_index:end_index]
            
            # Gather across devices
            q_embs_all = all_gather(q_embs.contiguous())
            c_embs_all = all_gather(c_embs.contiguous())
            labels_all = all_gather(labels.contiguous())

        # Compute similarity matrices
        qc_sim = sim_measure_dict[self.sim_measure](q_embs, c_embs_all)  # [B, B_all]
        cq_sim = sim_measure_dict[self.sim_measure](c_embs, q_embs_all)  # [B, B_all]

        # Create positive masks: positive_mask[i,j] = 1 if labels[i] == labels_all[j]
        qc_positive_mask = (labels.unsqueeze(1) == labels_all.unsqueeze(0)).float()  # [B, B_all]
        cq_positive_mask = (labels.unsqueeze(1) == labels_all.unsqueeze(0)).float()  # [B, B_all]

        # Apply temperature
        self.temp_param = self.temp_param.to(qc_sim.device)
        qc_logits = self.temp_param(qc_sim)
        cq_logits = self.temp_param(cq_sim)

        # Query-to-context loss
        qc_loss = self._multi_positive_nce(qc_logits, qc_positive_mask)
        
        # Context-to-query loss
        cq_loss = self._multi_positive_nce(cq_logits, cq_positive_mask)

        return 0.5 * (qc_loss + cq_loss)

    def _multi_positive_nce(self, logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
        """Compute NCE loss with multiple positives.
        
        For each query i with K positives:
            loss_i = -log(sum_k exp(sim[i,k]) / sum_j exp(sim[i,j]))
                   = -log_sum_exp(sim[i, positives]) + log_sum_exp(sim[i, all])
        
        Args:
            logits: Scaled similarity scores [B, B_all]
            positive_mask: Binary mask for positives [B, B_all]
            
        Returns:
            Loss value
        """
        # Mask out negatives by setting them to very negative values for log_sum_exp
        masked_logits = logits.clone()
        masked_logits[positive_mask == 0] = torch.finfo(masked_logits.dtype).min
        
        # log(sum(exp(sim[i, positives])))
        log_sum_positives = torch.logsumexp(masked_logits, dim=1)
        
        # log(sum(exp(sim[i, all])))
        log_sum_all = torch.logsumexp(logits, dim=1)
        
        # -log(sum(exp(positives)) / sum(exp(all)))
        loss = -(log_sum_positives - log_sum_all).mean()
        
        return loss


class L1Regularization(torch.nn.Module):
    def __init__(self,):
        """
        Computes the L1 regularization term for embeddings.
        """
        super(L1Regularization, self).__init__()

    def forward(self, q_embs: torch.Tensor, c_embs: torch.Tensor) -> torch.Tensor:
        """
        Computes the L1 loss for q_embs and c_embs.

        Args:
            q_embs (torch.Tensor): Question embeddings.
            c_embs (torch.Tensor): Context embeddings.

        Returns:
            torch.Tensor: L1 loss term.
        """
        l1_loss = q_embs.abs().mean() + c_embs.abs().mean()
        return l1_loss

class EntropyMinimizationLoss(torch.nn.Module):
    def __init__(self, eps: float = 1e-10):
        """
        Computes entropy minimization loss to encourage spiky distributions.
        
        Args:
            eps: Small constant for numerical stability.
        """
        super(EntropyMinimizationLoss, self).__init__()
        self.eps = eps

    def forward(self, q_embs: torch.Tensor, c_embs: torch.Tensor) -> torch.Tensor:
        """
        Computes entropy for each embedding to encourage spikiness.
        
        The loss encourages embeddings to have low entropy (be spiky)
        by treating each embedding vector as an unnormalized distribution.

        Args:
            q_embs (torch.Tensor): Question embeddings.
            c_embs (torch.Tensor): Context embeddings.

        Returns:
            torch.Tensor: Entropy loss term (lower is spikier).
        """
        # # Convert embeddings to probability distributions via softmax
        # q_probs = torch.softmax(q_embs, dim=-1)
        # c_probs = torch.softmax(c_embs, dim=-1)
        
        # Compute entropy: H(p) = -sum(p * log(p))
        q_entropy = -(q_embs * torch.log(q_embs + self.eps)).sum(dim=-1).mean()
        c_entropy = -(c_embs * torch.log(c_embs + self.eps)).sum(dim=-1).mean()
        
        # Sum entropies just like L1
        return q_entropy + c_entropy
    

class HardAssignmentLoss(torch.nn.Module):
    def __init__(
        self,
        local_loss: bool = False,
        optimize_prior_levels_too: bool = False,
    ):
        """Loss that forces query and positive doc to map to the same node.
        
        Uses argmax of doc as hard target for query (and vice versa).
        
        Args:
            local_loss: If True, only use data from current device
            optimize_prior_levels_too: Whether to use all tree levels
        """
        super(HardAssignmentLoss, self).__init__()
        
        self.local_loss = local_loss
        self.optimize_prior_levels_too = optimize_prior_levels_too

    def __call__(
        self,
        q_embs: torch.Tensor,
        c_embs: torch.Tensor,
        depth: torch.Tensor,
    ) -> torch.Tensor:
        """Computes hard assignment loss.
        
        Args:
            q_embs: Query node probabilities [batch_size, num_nodes]
            c_embs: Context node probabilities [batch_size, num_nodes]
            depth: Current depth level
            
        Returns:
            Loss encouraging same node assignment
        """
        # Extract relevant depth slice
        _, d = q_embs.shape
        depth_val = depth.item()
        
        if self.optimize_prior_levels_too:
            start_index = 0
            end_index = 2 ** (depth_val + 1) - 1
        else:
            start_index = 2 ** depth_val - 1
            end_index = 2 ** (depth_val + 1) - 1
        
        q_embs = q_embs[:, start_index:end_index]
        c_embs = c_embs[:, start_index:end_index]
        
        # Get argmax targets
        doc_targets = c_embs.argmax(dim=-1)
        query_targets = q_embs.argmax(dim=-1)
        
        # Directly maximize probability at target position
        # Gather probabilities at argmax positions
        loss_q2c = -torch.log(q_embs.gather(1, doc_targets.unsqueeze(1)) + 1e-10).mean()
        loss_c2q = -torch.log(c_embs.gather(1, query_targets.unsqueeze(1)) + 1e-10).mean()
        
        return 0.5 * (loss_q2c + loss_c2q)

class MaxProbabilityLoss(torch.nn.Module):
    def __init__(self, eps: float = 1e-10):
        """
        Encourages spiky distributions by maximizing the max probability.
        
        Args:
            eps: Small constant for numerical stability.
        """
        super(MaxProbabilityLoss, self).__init__()
        self.eps = eps

    def forward(self, q_embs: torch.Tensor, c_embs: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        Maximizes the max probability for each embedding to encourage spikiness.

        Args:
            q_embs (torch.Tensor): Question embeddings.
            c_embs (torch.Tensor): Context embeddings.

        Returns:
            torch.Tensor: Negative max probability loss (lower is spikier).
        """
        depth = depth.item()
        start_index = 2 ** depth - 1
        end_index = 2 ** (depth + 1) - 1
        
        # Get max probability for each embedding
        q_max_prob = q_embs[:, start_index:end_index].max(dim=-1)[0].mean()
        c_max_prob = c_embs[:, start_index:end_index].max(dim=-1)[0].mean()
        
        # Return negative sum (we minimize this, which maximizes the max probs)
        return -(q_max_prob + c_max_prob)
    
import torch

class PathConsistencyLoss(torch.nn.Module):
    def __init__(self,
                 max_depth: int,
                 eps: float = 1e-10,
                 temperature: float = 0.1,
                 compute_at_all_levels: bool = False):
        """
        Encourages high node probabilities along the path from root to the 
        highest-probability node at the target depth.
        """
        super(PathConsistencyLoss, self).__init__()
        self.max_depth = max_depth
        self.eps = eps
        self.temperature = temperature
        self.compute_at_all_levels = compute_at_all_levels
        
        # Precompute ancestor paths as tensors for vectorization
        self.ancestor_indices = self._compute_ancestor_indices(max_depth)
    
    def _compute_ancestor_indices(self, max_depth):
        """
        For each depth, create a tensor of ancestor paths.
        Returns: {depth: tensor of shape [num_nodes_at_depth, path_length]}
        """
        indices = {}
        
        for d in range(max_depth + 1):
            start_idx = 2**d - 1
            end_idx = 2**(d+1) - 1
            num_nodes = end_idx - start_idx
            path_length = d + 1  # root to node at depth d
            
            # Create tensor to hold all paths
            paths_tensor = torch.zeros(num_nodes, path_length, dtype=torch.long)
            
            for node_offset in range(num_nodes):
                node_idx = start_idx + node_offset
                path = []
                current = node_idx
                
                # Trace back to root
                while True:
                    path.append(current)
                    if current == 0:
                        break
                    current = (current - 1) // 2
                
                # Reverse to get root → node order
                path = path[::-1]
                paths_tensor[node_offset] = torch.tensor(path, dtype=torch.long)
            
            indices[d] = paths_tensor
        
        return indices
    
    def forward(self, q_embs: torch.Tensor, c_embs: torch.Tensor, depth: torch.Tensor, labels: torch.Tensor=None):
        """
        Args:
            q_embs: [batch_size, num_nodes] - node probabilities for queries  
            c_embs: [batch_size, num_nodes] - node probabilities for contexts
            depth: current depth level to evaluate at
        """
        depth_val = depth.item()
        
        if not self.compute_at_all_levels:
            q_loss = self._compute_path_consistency(q_embs, depth_val)
            c_loss = self._compute_path_consistency(c_embs, depth_val)
            return q_loss + c_loss
        else:
            total_loss = 0.0
            # Iterate through all levels up to current depth
            for d in range(1, depth_val + 1):
                q_loss = self._compute_path_consistency(q_embs, d)
                c_loss = self._compute_path_consistency(c_embs, d)
                total_loss += q_loss + c_loss
            
            # Normalize by the number of levels considered
            return total_loss / depth_val
            
        
    
    def _compute_path_consistency(self, node_probs, depth_val):
        """
        Vectorized computation of path consistency.
        """
        batch_size = node_probs.shape[0]
        
        # Extract probabilities at target depth
        start_idx = 2**depth_val - 1
        end_idx = 2**(depth_val + 1) - 1
        depth_node_probs = node_probs[:, start_idx:end_idx]  # [batch_size, num_nodes_at_depth]
        
        # Soft weighting
        node_weights = torch.softmax(depth_node_probs / self.temperature, dim=-1)
        
        # Get ancestor indices: [num_nodes_at_depth, path_length]
        ancestor_indices = self.ancestor_indices[depth_val].to(node_probs.device)
        
        # Vectorized gathering of all path probabilities
        # node_probs: [batch_size, num_nodes]
        # ancestor_indices: [num_nodes_at_depth, path_length]
        # Result: [batch_size, num_nodes_at_depth, path_length]
        path_probs = node_probs[:, ancestor_indices]
        
        # Compute log-probabilities and sum along paths
        path_log_probs = torch.log(path_probs.clamp(min=self.eps)).sum(dim=-1)  # [batch_size, num_nodes_at_depth]
        
        # Weighted sum
        weighted_score = (node_weights * path_log_probs).sum(dim=-1)  # [batch_size]
        
        return -weighted_score.mean()

import torch
import torch.nn as nn
import math

class PathConsistencySkipConnectionsLoss(nn.Module):
    def __init__(self,
                 max_depth: int,
                 eps: float = 1e-10,
                 temperature: float = 0.1):
        """
        Leaf-Guided Consistency Loss with Shortcuts.
        
        The active 'leaf' layer (defined by depth in forward) acts as a Teacher.
        It supervises ALL intermediate layers (Students) simultaneously via 
        direct shortcut connections.
        """
        super(PathConsistencySkipConnectionsLoss, self).__init__()
        self.max_depth = max_depth
        self.eps = eps
        self.temperature = temperature
        
        # Precompute the global ancestry map
        self.register_buffer('global_ancestry_map', self._precompute_global_map(max_depth))
    
    def _precompute_global_map(self, max_depth):
        """
        Precomputes lookup table: M[d, k] = Global Index of ancestor of node k at depth d.
        Shape: [max_depth + 1, total_nodes]
        """
        total_nodes = 2**(max_depth + 1) - 1
        ancestry_map = torch.full((max_depth + 1, total_nodes), -1, dtype=torch.long)
        
        for k in range(total_nodes):
            node_depth = int(math.floor(math.log2(k + 1)))
            level_start = 2**node_depth - 1
            node_offset = k - level_start
            
            for d in range(node_depth + 1):
                shift = node_depth - d
                ancestor_offset = node_offset >> shift
                ancestor_level_start = 2**d - 1
                ancestry_map[d, k] = ancestor_level_start + ancestor_offset
                
        return ancestry_map

    def forward(self, q_embs: torch.Tensor, c_embs: torch.Tensor, depth: torch.Tensor, labels: torch.Tensor=None):
        """
        Args:
            q_embs, c_embs: [Batch, Total_Nodes] (Preorder flattened)
            depth: Scalar tensor indicating the current 'Teacher' depth.
        """
        depth_val = depth.item()
        
        # No students to teach if depth is 0 or 1
        if depth_val <= 1:
            return torch.tensor(0.0, device=q_embs.device)

        q_loss = self._vectorized_consistency(q_embs, depth_val)
        c_loss = self._vectorized_consistency(c_embs, depth_val)
        
        return q_loss + c_loss

    def _vectorized_consistency(self, node_probs, teacher_depth):   
        """
        Computes consistency loss for all student layers [1 ... teacher_depth-1]
        in a single vectorized operation, NORMALIZED by student depth.
        """
        batch_size = node_probs.shape[0]
        
        # 1. Identify Teacher Nodes
        start_idx = 2**teacher_depth - 1
        end_idx = 2**(teacher_depth + 1) - 1
        teacher_probs = node_probs[:, start_idx:end_idx].detach()
        
        # 2. Vectorized Ancestor Lookup
        ancestor_indices = self.global_ancestry_map[1:teacher_depth, start_idx:end_idx]
        ancestor_indices = ancestor_indices.to(node_probs.device)
        
        # 3. Expand for Batch Gathering
        expanded_indices = ancestor_indices.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 4. Gather Student Probabilities
        student_probs = torch.gather(node_probs.unsqueeze(1).expand(-1, teacher_depth - 1, -1), 
                                     2, 
                                     expanded_indices)
        
        # 5. Compute Loss
        teacher_probs_expanded = teacher_probs.unsqueeze(1)
        
        # Loss per layer: [Batch, Num_Student_Layers]
        # Summing over the node dimension (-1)
        layer_losses = - (teacher_probs_expanded * torch.log(student_probs.clamp(min=self.eps))).sum(dim=-1)
        
        # --- NEW: DEPTH NORMALIZATION ---
        # Create a tensor of depths [1, 2, ..., teacher_depth-1]
        # Shape: [1, Num_Student_Layers] to broadcast over batch
        student_depths = torch.arange(1, teacher_depth, device=node_probs.device, dtype=torch.float32).unsqueeze(0)
        
        # Normalize: Divide loss by depth to make L1 contribution equal to L9
        normalized_layer_losses = layer_losses / student_depths
        
        # Now taking the mean is safe and balanced
        return normalized_layer_losses.mean()


class TreeConsistencyRegularization(torch.nn.Module):
    def __init__(self, max_depth: int):
        """
        Computes TVD regularization between tree levels to encourage consistency.
        At each level, compares the distribution with the one obtained by summing 
        adjacent pairs from the child level.
        
        Args:
            max_depth (int): Maximum depth of the tree (0-indexed, so depth=2 means 3 levels: 0,1,2)
        """
        super(TreeConsistencyRegularization, self).__init__()
        
        # Precompute indices for efficient forward pass
        if max_depth < 2:
            # No comparisons possible
            self.register_buffer('parent_indices', torch.tensor([], dtype=torch.long))
            self.register_buffer('left_child_indices', torch.tensor([], dtype=torch.long))
            self.register_buffer('right_child_indices', torch.tensor([], dtype=torch.long))
        else:
            # Get all parent node indices from levels 1 to max_depth-1
            # These are nodes that have children to compare against
            parent_start = 1  # Skip root (level 0)
            parent_end = (2 ** max_depth) - 1  # Nodes before the last level
            parent_indices = torch.arange(parent_start, parent_end, dtype=torch.long)
            
            # Compute children indices for each parent (binary tree property)
            # Node i has children at 2*i+1 and 2*i+2
            left_child_indices = 2 * parent_indices + 1
            right_child_indices = 2 * parent_indices + 2
            
            # Register as buffers so they move with the model to GPU
            self.register_buffer('parent_indices', parent_indices)
            self.register_buffer('left_child_indices', left_child_indices)
            self.register_buffer('right_child_indices', right_child_indices)
    
    def forward(self, q_embs: torch.Tensor, c_embs: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        Computes the TVD regularization for tree consistency.
        Args:
            q_embs (torch.Tensor): Question scores [batch, num_nodes]. 
                                   Scores at each level sum to 1.
            c_embs (torch.Tensor): Context scores [batch, num_nodes].
                                   Scores at each level sum to 1.
        Returns:
            torch.Tensor: TVD regularization term.
        """
        # Edge case: no comparisons possible
        if len(self.parent_indices) == 0:
            return torch.tensor(0.0, device=q_embs.device, dtype=q_embs.dtype)
                
        depth = depth.item()
        end_index = 2 ** (depth + 1) - 1
        parent_cutoff = 2 ** depth - 1
        
        q_embs = q_embs[:, :end_index]
        c_embs = c_embs[:, :end_index]
        
        # Filter precomputed indices to only include valid parents for this depth
        valid_mask = self.parent_indices < parent_cutoff
        parent_indices = self.parent_indices[valid_mask]
        left_child_indices = self.left_child_indices[valid_mask]
        right_child_indices = self.right_child_indices[valid_mask]
        
        
        # If no valid parents, return zero loss
        if len(parent_indices) == 0:
            return torch.tensor(0.0, device=q_embs.device, dtype=q_embs.dtype)
        
        # Extract all parents and children at once using precomputed indices
        q_parents = q_embs[:, parent_indices]  # [batch, num_parents]
        q_left = q_embs[:, left_child_indices]  # [batch, num_parents]
        q_right = q_embs[:, right_child_indices]  # [batch, num_parents]
        
        c_parents = c_embs[:, parent_indices]  # [batch, num_parents]
        c_left = c_embs[:, left_child_indices]  # [batch, num_parents]
        c_right = c_embs[:, right_child_indices]  # [batch, num_parents]
        
        # Aggregate children scores (sum siblings)
        q_aggregated = q_left + q_right  # [batch, num_parents]
        c_aggregated = c_left + c_right  # [batch, num_parents]
        
        # Compute TVD: 0.5 * sum(|parent - aggregated|)
        # Sum over all parent nodes, then average over batch
        q_tvd = 0.5 * (q_parents - q_aggregated).abs().sum(dim=-1).mean()  # scalar
        c_tvd = 0.5 * (c_parents - c_aggregated).abs().sum(dim=-1).mean()  # scalar
        
        # Average both regularization terms
        return (q_tvd + c_tvd) / 2
    
class TreeLocalityLoss(torch.nn.Module):
    def __init__(self, max_depth: int, eps: float = 1e-10, distance_type: str = 'linear'):
        """
        Encourages node probabilities to be concentrated in tree-local regions
        by minimizing expected tree distance.
        
        Args:
            max_depth: Maximum depth of the tree
            eps: Small value for numerical stability
            distance_type: 'linear', 'squared', or 'exponential'
        """
        super(TreeLocalityLoss, self).__init__()
        self.max_depth = max_depth
        self.eps = eps
        self.distance_type = distance_type
        
        # Precompute distance matrices for each depth
        self.distance_matrices = self._compute_distance_matrices(max_depth)
    
    def _compute_tree_distance(self, i: int, j: int, depth: int) -> int:
        """
        Compute tree distance between two nodes at the same depth.
        
        Tree distance = depth - depth_of_LCA
        where LCA is found by common prefix in binary paths.
        
        Args:
            i, j: node offsets at the given depth (0 to 2^depth - 1)
            depth: the depth level
        
        Returns:
            Tree distance (number of edges in path between nodes)
        """
        if depth == 0:
            return 0  # Only one node at depth 0
        
        if i == j:
            return 0
        
        # Convert to binary paths (path from root is binary representation of offset)
        path_i = format(i, f'0{depth}b')
        path_j = format(j, f'0{depth}b')
        
        # Count common prefix length
        common_prefix_len = 0
        for bit_i, bit_j in zip(path_i, path_j):
            if bit_i == bit_j:
                common_prefix_len += 1
            else:
                break
        
        # Tree distance = depth - depth_of_LCA
        # Depth of LCA = common_prefix_len
        return depth - common_prefix_len
    
    def _compute_distance_matrices(self, max_depth: int) -> dict:
        """
        Precompute distance matrices for all depths.
        
        Returns:
            {depth: tensor of shape [2^depth, 2^depth]}
        """
        matrices = {}
        
        for d in range(max_depth + 1):
            num_nodes = 2 ** d
            D = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
            
            for i in range(num_nodes):
                for j in range(num_nodes):
                    dist = self._compute_tree_distance(i, j, d)
                    
                    # Apply distance transformation
                    if self.distance_type == 'linear':
                        D[i, j] = float(dist)
                    elif self.distance_type == 'squared':
                        D[i, j] = float(dist ** 2)
                    elif self.distance_type == 'exponential':
                        D[i, j] = float(torch.exp(torch.tensor(dist, dtype=torch.float32)))
                    else:
                        D[i, j] = float(dist)
            
            matrices[d] = D
        
        return matrices
    
    def forward(self, q_embs: torch.Tensor, c_embs: torch.Tensor, depth: torch.Tensor):
        """
        Args:
            q_embs: [batch_size, num_nodes] - node probabilities for queries  
            c_embs: [batch_size, num_nodes] - node probabilities for contexts
            depth: current depth level to evaluate at
        """
        depth_val = depth.item()
        
        q_loss = self._compute_tree_distance_loss(q_embs, depth_val)
        c_loss = self._compute_tree_distance_loss(c_embs, depth_val)
        
        return q_loss + c_loss
    
    def _compute_tree_distance_loss(self, node_probs: torch.Tensor, depth_val: int):
        """
        Compute expected tree distance: E[tree_distance] = p^T @ D @ p
        
        where p is the probability distribution over nodes at depth_val
        and D is the pairwise tree distance matrix.
        
        Args:
            node_probs: [batch_size, num_nodes] - all node probabilities
            depth_val: which depth level to evaluate
            
        Returns:
            Mean expected tree distance across batch
        """
        # Extract probabilities at target depth
        start_idx = 2**depth_val - 1
        end_idx = 2**(depth_val + 1) - 1
        depth_node_probs = node_probs[:, start_idx:end_idx]  # [batch_size, 2^depth]
        
        # Get distance matrix for this depth: D[i,j] = tree_distance(i, j)
        D = self.distance_matrices[depth_val].to(node_probs.device)  # [2^depth, 2^depth]
        
        # Compute p^T @ D @ p for each sample in batch
        # This gives expected tree distance under distribution p
        # 
        # Vectorized computation:
        # temp = p @ D gives [batch_size, num_nodes_at_depth]
        # (temp * p).sum() gives p^T @ D @ p for each sample
        temp = depth_node_probs @ D  # [batch_size, num_nodes_at_depth]
        expected_distances = (temp * depth_node_probs).sum(dim=-1)  # [batch_size]
        
        return expected_distances.mean()
    
class Balance(torch.nn.Module):
    def __init__(self, args_idx_to_eval: int = 0):
        """Negative tre Shannon entropy.

        Args:
            args_idx_to_eval (int, optional): Index of argument passed to __call__ to evaluate balance. Defaults to 0.
        """

        super(Balance, self).__init__()

        self.args_idx = args_idx_to_eval

    def __call__(
        self,
        *args,
        eps: float = 1e-10,
    ) -> torch.Tensor:
        logits = args[self.args_idx]
        probs = logits / logits.sum(1, keepdim=True)

        return (probs * torch.log(probs + eps)).mean(0).sum()


class Sharpness(torch.nn.Module):
    def __init__(self, args_idx_to_eval: int = 0):
        """Negative L2 loss to encourage sharpness.

        Args:
            args_idx_to_eval (int, optional): Index of argument passed to __call__ to evaluate sharpness. Defaults to 0.
        """

        super(Sharpness, self).__init__()

        self.args_idx = args_idx_to_eval

    def __call__(
        self,
        *args,
    ) -> torch.Tensor:
        logits = args[self.args_idx]
        probs = logits / logits.sum(1, keepdim=True)

        return -(probs**2).sum(1).mean()


class CompositeLoss(torch.nn.Module):
    def __init__(
        self,
        term_list: List,
        hp_list: List[float] = None,
    ):
        """Objective function composed of losses and regularization terms.

        Args:
            term_list (List): list of losses and regularization terms to add up.
            hp_list (List[float], optional): list of hyper-parameters, one per term. Defaults to list of 1s.

        """

        super(CompositeLoss, self).__init__()

        self.term_list = term_list

        for term in self.term_list:
            for param_name, param in term.named_parameters(recurse=True):
                self.register_parameter(param_name.replace(".", "-"), param)

        if hp_list is not None:
            assert len(hp_list) == len(
                term_list
            ), "Must pass exactly one hyper-parameter per obj function term."

            self.hp_list = hp_list

        else:
            self.hp_list = torch.ones(len(term_list))

    def __call__(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        res = 0.0
        for hp, term in zip(self.hp_list, self.term_list):
            res += hp * term(*args, **kwargs)

        return res


class TripletLoss(torch.nn.Module):
    def __init__(
        self,
        margin: float = 0.2,
        local_loss: bool = False,
        freeze_tmp: bool = False,
        sim_measure: str = "ce",
        max_tmp: float = 30.0,
        use_learnable_temp: bool = False,
        optimize_prior_levels_too: bool = False,
        encoder_dim: int = 1024,
        margin_values: list = [0.03, 0.07, 0.1],  # List of margin values to sample from
        margin_weights: list = [32, 8, 1],  # Weights for sampling (will be normalized)
        use_dynamic_margin: bool = False,  # Enable dynamic margin selection
    ):
        """Triplet loss for metric learning in multi-GPU settings.
        
        Uses in-batch negatives: for each anchor-positive pair, all other 
        samples in the batch serve as negatives.

        Args:
            margin: Margin value for triplet constraint. Recommended values:
                   - 0.2: FaceNet baseline (most common)
                   - 0.1-0.3: Recent research optimal range
                   - Avoid values > 0.5 (too difficult to optimize)
            local_loss: If True, only use data from current device
            freeze_tmp: If True, temperature is not optimized
            sim_measure: The similarity measure to use (from sim_measure_dict)
            max_tmp: Maximum temperature value
            use_learnable_temp: Whether to use learnable temperature scaling.
                               Traditional triplet loss doesn't use temperature,
                               but it can be useful for scaling distances.
            encoder_dim: Dimension of encoder output (for compatibility)
        """
        super(TripletLoss, self).__init__()

        self.margin = margin
        self.local_loss = local_loss
        self.sim_measure = sim_measure
        self.use_learnable_temp = use_learnable_temp
        self.encoder_dim = encoder_dim
        self.optimize_prior_levels_too = optimize_prior_levels_too
        
        self.use_dynamic_margin = use_dynamic_margin
        if self.use_dynamic_margin:
            assert margin_values is not None and margin_weights is not None, \
                "margin_values and margin_weights must be provided when use_dynamic_margin=True"
            assert len(margin_values) == len(margin_weights), \
                "margin_values and margin_weights must have same length"
            
            self.margin_values = torch.tensor(margin_values, dtype=torch.float32)
            # Normalize weights to probabilities
            weights_tensor = torch.tensor(margin_weights, dtype=torch.float32)
            self.margin_probs = weights_tensor / weights_tensor.sum()
        else:
            self.margin_values = None
            self.margin_probs = None
        
        if self.use_learnable_temp:
            self.temp_param = TempCoef(initial_value=1.0, max_tmp=max_tmp)
            if freeze_tmp:
                self.temp_param.temp_coef.requires_grad = False

    def __call__(
        self,
        q_embs: torch.Tensor,
        c_embs: torch.Tensor,
        depth: torch.Tensor,
    ) -> torch.Tensor:
        """Computes triplet loss using in-batch negatives.
        
        For each query-context pair, all other contexts in the batch 
        are treated as negatives.

        Args:
            q_embs: Query representations [batch_size, dim]
            c_embs: Context representations [batch_size, dim]
            depth: Current depth level (for hierarchical models)
            
        Returns:
            Triplet loss value
        """
        # Extract the relevant slice based on depth (same logic as ContrastiveLoss)
        # Note: Removed optimize_prior_levels_too and optimize_token_loss for simplicity
        # If you need them, add them back as __init__ parameters
        _, d = q_embs.shape
        
        if not self.optimize_prior_levels_too:
            depth_val = depth.item()
            
            start_index = 2 ** depth_val - 1
            end_index = 2 ** (depth_val + 1) - 1
        
            q_embs = q_embs[:, start_index:end_index]
            c_embs = c_embs[:, start_index:end_index]
        
        if self.local_loss:
            q_embs_all = q_embs
            c_embs_all = c_embs
        else:
            # Gather embeddings across devices
            q_embs_all = all_gather(q_embs.contiguous())
            c_embs_all = all_gather(c_embs.contiguous())

        # Compute similarity matrix using your existing sim_measure_dict
        # qc_sim_matrix: [batch_size, num_all] - query to all contexts
        qc_sim_matrix = sim_measure_dict[self.sim_measure](q_embs_all, c_embs_all)
        
        # Apply learnable temperature if enabled
        if self.use_learnable_temp:
            self.temp_param = self.temp_param.to(qc_sim_matrix.device)
            qc_sim_matrix = self.temp_param(qc_sim_matrix)
        
        # Convert similarities to distances
        # Higher similarity = lower distance, so we use negative similarity as distance
        qc_distance_matrix = -qc_sim_matrix
        
        # # Get positive indices (matching query-context pairs on diagonal)
        # positive_idx = torch.arange(qc_distance_matrix.size(0)).long().to(qc_distance_matrix.device)
        # if not self.local_loss:
        #     positive_idx = positive_idx + dist.get_rank() * q_embs.size(0)
        
        # # Distance to positive (diagonal)
        # dist_qp = qc_distance_matrix[torch.arange(q_embs_all.size(0)), positive_idx]
        
        # # For negatives, use all other contexts in the batch
        # # Create mask to exclude the positive (diagonal)
        # mask = torch.ones_like(qc_distance_matrix, dtype=torch.bool)
        # mask[torch.arange(q_embs_all.size(0)), positive_idx] = False
        
        # # Get hardest negative (minimum distance, i.e., maximum similarity)
        # # This is "hard negative mining" within the batch
        # negative_distances = qc_distance_matrix.masked_fill(~mask, float('inf'))
        # dist_qn = negative_distances.min(dim=1)[0]
        
        if not self.local_loss:
            # Only compute loss for local samples on this GPU
            local_batch_size = q_embs.size(0)
            rank_offset = dist.get_rank() * local_batch_size
            
            # Extract local portion of distance matrix (only rows for local samples)
            qc_distance_matrix_local = qc_distance_matrix[rank_offset:rank_offset + local_batch_size, :]
            
            # Positive indices are where local samples match in the full gathered batch
            positive_idx = torch.arange(rank_offset, rank_offset + local_batch_size).long().to(qc_distance_matrix.device)
            
            # Distance to positive (diagonal positions)
            dist_qp = qc_distance_matrix_local[torch.arange(local_batch_size), positive_idx]
            
            # Create mask to exclude positives (use local batch size for rows)
            mask = torch.ones_like(qc_distance_matrix_local, dtype=torch.bool)
            mask[torch.arange(local_batch_size), positive_idx] = False
            
            # Get hardest negative
            negative_distances = qc_distance_matrix_local.masked_fill(~mask, float('inf'))
            dist_qn = negative_distances.min(dim=1)[0]
        else:
            # Local loss: use full matrix
            positive_idx = torch.arange(q_embs_all.size(0)).long().to(qc_distance_matrix.device)
            
            dist_qp = qc_distance_matrix[positive_idx, positive_idx]
            
            mask = torch.ones_like(qc_distance_matrix, dtype=torch.bool)
            mask[positive_idx, positive_idx] = False
            
            negative_distances = qc_distance_matrix.masked_fill(~mask, float('inf'))
            dist_qn = negative_distances.min(dim=1)[0]
        
                
        # Sample margin dynamically if enabled
        if self.use_dynamic_margin:
            # Sample a margin value for this batch
            margin_idx = torch.multinomial(self.margin_probs, 1).item()
            current_margin = self.margin_values[margin_idx].item()
        else:
            current_margin = self.margin
        
        # Triplet loss: max(d(q,p) - d(q,n) + margin, 0)
        losses = F.relu(dist_qp - dist_qn + current_margin)
        
        return losses.mean()


class TripletLossAllNegatives(torch.nn.Module):
    """Alternative: Average over all negatives instead of just hardest."""
    
    def __init__(
        self,
        margin: float = 0.2,
        local_loss: bool = False,
        freeze_tmp: bool = False,
        sim_measure: str = "ce",
        max_tmp: float = 30.0,
        use_learnable_temp: bool = False,
        encoder_dim: int = 1024,
    ):
        super(TripletLossAllNegatives, self).__init__()

        self.margin = margin
        self.local_loss = local_loss
        self.sim_measure = sim_measure
        self.use_learnable_temp = use_learnable_temp
        self.encoder_dim = encoder_dim
        
        if self.use_learnable_temp:
            self.temp_param = TempCoef(initial_value=1.0, max_tmp=max_tmp)
            if freeze_tmp:
                self.temp_param.temp_coef.requires_grad = False

    def __call__(
        self,
        q_embs: torch.Tensor,
        c_embs: torch.Tensor,
        depth: torch.Tensor,
    ) -> torch.Tensor:
        """Computes triplet loss averaged over all negatives.
        
        Args:
            q_embs: Query representations [batch_size, dim]
            c_embs: Context representations [batch_size, dim]
            depth: Current depth level (for hierarchical models)
            
        Returns:
            Triplet loss value
        """
        # Extract the relevant slice based on depth
        _, d = q_embs.shape
        depth_val = depth.item()
        
        start_index = 2 ** depth_val - 1
        end_index = 2 ** (depth_val + 1) - 1
        
        q_embs = q_embs[:, start_index:end_index]
        c_embs = c_embs[:, start_index:end_index]
        
        if self.local_loss:
            q_embs_all = q_embs
            c_embs_all = c_embs
        else:
            q_embs_all = all_gather(q_embs.contiguous())
            c_embs_all = all_gather(c_embs.contiguous())

        # Compute similarity matrix
        qc_sim_matrix = sim_measure_dict[self.sim_measure](q_embs_all, c_embs_all)
        
        if self.use_learnable_temp:
            self.temp_param = self.temp_param.to(qc_sim_matrix.device)
            qc_sim_matrix = self.temp_param(qc_sim_matrix)
        
        # Convert to distances (negative similarity)
        qc_distance_matrix = -qc_sim_matrix
        
        # Get indices
        positive_idx = torch.arange(qc_distance_matrix.size(0)).long().to(qc_distance_matrix.device)
        if not self.local_loss:
            positive_idx = positive_idx + dist.get_rank() * q_embs.size(0)
        
        # Distance to positive
        dist_qp = qc_distance_matrix[torch.arange(q_embs_all.size(0)), positive_idx]
        
        # Average over all negatives
        mask = torch.ones_like(qc_distance_matrix, dtype=torch.bool)
        mask[torch.arange(q_embs_all.size(0)), positive_idx] = False
        
        # Compute triplet loss for all negatives and average
        dist_qp_expanded = dist_qp.unsqueeze(1)  # [batch_size, 1]
        triplet_losses = F.relu(dist_qp_expanded - qc_distance_matrix + self.margin)
        
        # Zero out the positive pairs
        triplet_losses = triplet_losses.masked_fill(~mask, 0.0)
        
        # Average over all negatives
        num_negatives = mask.sum(dim=1).float()
        loss = (triplet_losses.sum(dim=1) / num_negatives).mean()
        
        return loss

sim_measure_dict = {
    "ce": matrix_neg_cross_entropy,
    "tvd": total_variation_distance_matrix,
    "cos_sim": cosine_similarity_matrix,
}
