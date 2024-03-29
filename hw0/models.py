"""
Classes defining user and item latent representations in
factorization models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)
            
    
class ZeroEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.

    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)
            
        
class MultiTaskNet(nn.Module):
    """
    Multitask factorization representation.

    Encodes both users and items as an embedding layer; the likelihood score
    for a user-item pair is given by the dot product of the item
    and user latent vectors. The numerical score is predicted using a small MLP.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    layer_sizes: list
        List of layer sizes to for the regression network.
    sparse: boolean, optional
        Use sparse gradients.
    embedding_sharing: boolean, optional
        Share embedding representations for both tasks.

    """

    def __init__(self, num_users, num_items, embedding_dim=32, layer_sizes=[96, 64], 
                 sparse=False, embedding_sharing=True):

        super().__init__()

        self.embedding_dim = embedding_dim

        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************
        if embedding_sharing:
            self.U = ScaledEmbedding(num_users, embedding_dim)
            self.Q = ScaledEmbedding(num_items, embedding_dim)
        else:
            self.U = nn.ModuleList([
                ScaledEmbedding(num_users, embedding_dim),
                ScaledEmbedding(num_users, embedding_dim),
            ])
            self.Q = nn.ModuleList([
                ScaledEmbedding(num_items, embedding_dim),
                ScaledEmbedding(num_items, embedding_dim),
            ])
        self.A = ZeroEmbedding(num_users, 1)
        self.B = ZeroEmbedding(num_items, 1)
        self.embedding_sharing = embedding_sharing

        self.f_theta = nn.Sequential(
            nn.Linear(layer_sizes[0], layer_sizes[1]),
            nn.ReLU(),
            nn.Linear(layer_sizes[1], 1),
        )
        
        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************
        
    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            A tensor of integer user IDs of shape (batch,)
        item_ids: tensor
            A tensor of integer item IDs of shape (batch,)

        Returns
        -------

        predictions: tensor
            Tensor of user-item interaction predictions of 
            shape (batch,). This corresponds to p_ij in the 
            assignment.
        score: tensor
            Tensor of user-item score predictions of shape 
            (batch,). This corresponds to r_ij in the 
            assignment.
        """
        
        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************
        if self.embedding_sharing:
            uv = self.U(user_ids)
            qv = self.Q(item_ids)
            uq = uv * qv
            predictions = torch.sum(uq, -1) + self.A(user_ids) + self.B(item_ids)
            score = self.f_theta(torch.cat((uv, qv, uq), -1))
        else:
            predictions = torch.sum(self.U[0](user_ids) * self.Q[0](item_ids), -1) + self.A(user_ids) + self.B(item_ids)
            score = self.f_theta(
                torch.cat((self.U[1](user_ids), self.Q[1](item_ids), self.U[1](user_ids) * self.Q[1](item_ids)), 
                -1
                )
            )
        #********************************************************
        #********************************************************
        #********************************************************
        return predictions, score
