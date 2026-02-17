"""Unit tests for loss functions."""

import pytest
import torch

from retreever.utils.neural import (
    ContrastiveLoss,
    MultiLabelContrastiveLoss,
    MRLLoss,
    TempCoef,
)


class TestLossFunctions:
    """Test loss functions."""
    
    def setup_method(self):
        """Setup test data."""
        self.batch_size = 4
        self.emb_dim = 128
        
        # Create dummy embeddings
        self.q_embeddings = torch.randn(self.batch_size, self.emb_dim)
        self.c_embeddings = torch.randn(self.batch_size, self.emb_dim)
    
    def test_contrastive_loss_forward(self):
        """Test contrastive loss forward pass."""
        loss_fn = ContrastiveLoss(
            local_loss=True,
            init_tmp=1.0,
            freeze_tmp=True,
        )
        
        loss = loss_fn(self.q_embeddings, self.c_embeddings)
        
        assert loss.shape == ()  # Scalar
        assert loss.item() > 0  # Positive loss
    
    def test_multi_label_contrastive_loss(self):
        """Test multi-label contrastive loss."""
        loss_fn = MultiLabelContrastiveLoss(
            local_loss=True,
            init_tmp=1.0,
            freeze_tmp=True,
        )
        
        # Create labels (batch_size,)
        labels = torch.tensor([0, 1, 0, 1])
        
        loss = loss_fn(self.q_embeddings, self.c_embeddings, labels=labels)
        
        assert loss.shape == ()
        assert loss.item() > 0
    
    def test_mrl_loss(self):
        """Test MRL loss."""
        loss_fn = MRLLoss(
            encoder_dim=self.emb_dim,
            local_loss=True,
            init_tmp=1.0,
            freeze_tmp=True,
        )
        
        loss = loss_fn(self.q_embeddings, self.c_embeddings)
        
        assert loss.shape == ()
        assert loss.item() > 0
    
    def test_temperature_coefficient(self):
        """Test learnable temperature coefficient."""
        temp_coef = TempCoef(init_val=1.0, freeze=False)
        
        # Check initial value
        temp = temp_coef()
        assert temp.item() == pytest.approx(1.0, rel=0.1)
        
        # Check that it's learnable
        assert temp_coef.temp_coef.requires_grad
    
    def test_frozen_temperature(self):
        """Test frozen temperature coefficient."""
        temp_coef = TempCoef(init_val=2.0, freeze=True)
        
        # Check that it's not learnable
        assert not temp_coef.temp_coef.requires_grad
    
    def test_loss_backward(self):
        """Test that loss can be backpropagated."""
        loss_fn = ContrastiveLoss(
            local_loss=True,
            init_tmp=1.0,
            freeze_tmp=True,
        )
        
        # Make embeddings require gradients
        q_emb = self.q_embeddings.clone().requires_grad_(True)
        c_emb = self.c_embeddings.clone().requires_grad_(True)
        
        loss = loss_fn(q_emb, c_emb)
        loss.backward()
        
        # Check that gradients are computed
        assert q_emb.grad is not None
        assert c_emb.grad is not None
    
    def test_similarity_measures(self):
        """Test different similarity measures."""
        for sim_measure in ["cos_sim", "tvd", "cross_entropy"]:
            loss_fn = ContrastiveLoss(
                local_loss=True,
                init_tmp=1.0,
                freeze_tmp=True,
                sim_measure=sim_measure,
            )
            
            loss = loss_fn(self.q_embeddings, self.c_embeddings)
            assert loss.item() > 0
