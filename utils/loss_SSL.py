
import torch.nn as nn


"""
SimSiamLoss Class implements the loss function for SimSiam. 
asymmetric_loss Method computes the similarity-based loss for one prediction-projection 
pair while ensuring that no gradients flow through the projection vector ( z ). 
The original version uses the negative dot product between  p  and  z  
while the simplified version uses negative cosine similarity,
 which is computationally efficient and aligns with the unit-norm constraint.
 """

class SimSiamLoss(nn.Module):
    """
    Implementation of the SimSiam loss function.
    This loss is designed for self-supervised learning by comparing the similarity
    between pairs of projections and predictions from two augmented views of the same image.
 
    Reference:
    SimSiam: Exploring Simple Siamese Representation Learning (https://arxiv.org/abs/2011.10566)
    """
 
    def __init__(self, version='simplified'):
        """
        Initialize the SimSiam loss module.
 
        Args:
            version (str): Specifies the version of the loss.
                           'original' uses the original dot-product-based formulation,
                           'simplified' uses cosine similarity (default).
        """
        super().__init__()
        self.ver = version
 
    def asymmetric_loss(self, p, z):
        """
        Compute the asymmetric loss between the prediction (p) and the projection (z).
        This enforces similarity between the two while detaching the gradient from `z`.
 
        Args:
            p (torch.Tensor): Prediction vector.
            z (torch.Tensor): Projection vector.
 
        Returns:
            torch.Tensor: Computed loss.
        """
        if self.ver == 'original':
            # Detach z to stop gradient flow
            z = z.detach()
 
            # Normalize vectors
            p = nn.functional.normalize(p, dim=1)
            z = nn.functional.normalize(z, dim=1)
 
            # Original formulation: negative dot product
            return -(p * z).sum(dim=1).mean()
 
        elif self.ver == 'simplified':
            # Detach z to stop gradient flow
            z = z.detach()
 
            # Simplified formulation: negative cosine similarity
            return -nn.functional.cosine_similarity(p, z, dim=-1).mean()
 
    def forward(self, z1, z2, p1, p2):
        """
        Compute the SimSiam loss for two pairs of projections and predictions.
 
        Args:
            z1 (torch.Tensor): Projection vector from the first augmented view.
            z2 (torch.Tensor): Projection vector from the second augmented view.
            p1 (torch.Tensor): Prediction vector corresponding to z1.
            p2 (torch.Tensor): Prediction vector corresponding to z2.
 
        Returns:
            torch.Tensor: Averaged SimSiam loss.
        """
        # Compute the loss for each pair (p1, z2) and (p2, z1)
        loss1 = self.asymmetric_loss(p1, z2)
        loss2 = self.asymmetric_loss(p2, z1)
 
        # Average the two losses
        return 0.5 * loss1 + 0.5 * loss2