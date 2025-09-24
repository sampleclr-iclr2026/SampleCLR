import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    def __init__(self):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = 1.0

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
    
class DiscriminatorHead(nn.Module):
    def __init__(self, input_dim, num_classes, num_layers=2, hidden_size=128, use_normalization=True):
        super(DiscriminatorHead, self).__init__()
        self.grl = GradientReversalLayer()
        self.discriminator = build_mlp(input_dim, num_classes, num_layers, hidden_size, use_normalization)
    
    def forward(self, x, grl_lambda):
        self.grl.lambda_ = grl_lambda
        x_reversed = self.grl(x)
        return self.discriminator(x_reversed)


def build_mlp(input_dim, output_dim, num_layers, hidden_size, use_normalization):
    layers = []
    if num_layers == 1:
        layers.append(nn.Linear(input_dim, output_dim))
    else:
        # First layer
        layers.append(nn.Linear(input_dim, hidden_size))
        if use_normalization:
            # layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            
        layers.append(nn.ReLU())
        # Hidden layers
        for i in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            if use_normalization:
                # layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
        # Output layer
        layers.append(nn.Linear(hidden_size, output_dim))
    return nn.Sequential(*layers)

class ClassifierHead(nn.Module):
    def __init__(self, n_input_features, n_classes, num_layers, hidden_size, use_normalization=True):
        super(ClassifierHead, self).__init__()
        self.mlp = build_mlp(n_input_features, n_classes, num_layers, hidden_size, use_normalization)

    def forward(self, x):
        return self.mlp(x)

class RegressionHead(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_size, use_normalization=True, output_dim=1):
        super(RegressionHead, self).__init__()
        self.mlp = build_mlp(input_dim, output_dim, num_layers, hidden_size, use_normalization)

    def forward(self, x):
        return self.mlp(x)

class OrdinalRegressionHead(nn.Module):
    def __init__(self, input_dim, num_classes, num_layers, hidden_size, use_normalization=True):
        super(OrdinalRegressionHead, self).__init__()
        # For ordinal regression we output (num_classes - 1) logits.
        self.mlp = build_mlp(input_dim, num_classes - 1, num_layers, hidden_size, use_normalization)
        self.num_classes = num_classes

    def forward(self, x):
        logits = self.mlp(x)
        probas = torch.sigmoid(logits)
        return probas

class DynamicNetwork(nn.Module):
    def __init__(self, n_input_features: int, n_output_features: int, num_layers: int, hidden_size: int, activation='relu', normalization='BatchNorm'):
        super(DynamicNetwork, self).__init__()
        blocks = nn.ModuleList()
        if normalization == 'BatchNorm':
            normalization_layer = nn.BatchNorm1d
        elif normalization == 'LayerNorm':
            normalization_layer = nn.LayerNorm

        self.input_layer = nn.Linear(n_input_features, hidden_size)
            # nn.ReLU() if activation == 'relu' else nn.Sigmoid()
        self.input_layer_activation = nn.ReLU() if activation == 'relu' else nn.Sigmoid()

        for _ in range(num_layers - 1):
            layers = [nn.Linear(hidden_size, hidden_size)]
            if normalization is not None:
                layers.append(normalization_layer(hidden_size))
            layers.append(nn.ReLU() if activation == 'relu' else nn.Sigmoid() if activation == 'sigmoid' else None)
            block = nn.Sequential(*layers)
            blocks.append(block)

        self.blocks = blocks
        self.fc = nn.Linear(hidden_size, n_output_features)

    def forward(self, x):
        # Handle 3D input: (B, O, F) -> (B*O, F) for processing, then reshape back
        x = self.get_representations(x)
        x = self.fc(x)

        return x

    def get_representations(self, x, stop_at_layer=None):
        original_shape = x.shape
        if len(original_shape) == 3:
            B, O, F = original_shape
            x = x.view(B * O, F)
        
        x = self.input_layer(x)
        x = self.input_layer_activation(x)

        if stop_at_layer is None:
            stop_at_layer = len(self.blocks)

        for block in self.blocks[:stop_at_layer]:
            x = block(x) + x
        
        # Reshape back to original format if needed
        if len(original_shape) == 3:
            x = x.view(B, O, -1)

        return x

class MultiHeadAggregationNetwork(nn.Module):
    def __init__(
            self, n_heads,
            n_input_features,
            activation='sigmoid',
            pruning_threshold=None,
            use_gumbel=True,
            temperature=1.0, 
            return_weights=False, 
            num_layers=2,
            hidden_size=128, 
            normalization='BatchNorm'
        ):
        super().__init__()
        if num_layers == 1:
            self.head = nn.Linear(n_input_features, n_heads, bias=True)
        else:
            self.head = DynamicNetwork(
                n_input_features=n_input_features,
                n_output_features=n_heads,
                num_layers=num_layers,
                hidden_size=hidden_size,
                activation="relu",
                normalization=normalization
            )
        self.pruning_threshold = pruning_threshold
        self.use_gumbel = use_gumbel
        self.temperature = temperature
        self.return_weights = return_weights
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError("Please select 'sigmoid' or 'softmax' as the activation function.")

    def forward(self, x):
        # x: shape (B, O, F) where O = number of observations (cells), F = number of features
        weights_unscaled = self.head(x)  # shape: (B, O, H)

        # Padded cells have 0s in every feature and thus 0 norm. It does not contribute to the weighted sum,
        # but biases will accumulate for the empty cells. We need to correct for this or simply mask the weigts.
        features_norms = x.norm(dim=2)  # (B, O)
        # Create a mask for padded cells (where feature norm is 0)
        mask = (features_norms != 0).unsqueeze(-1)  # (B, O, 1)
        # Zero out weights for padded cells
        weights_unscaled = weights_unscaled * mask
        
        if self.use_gumbel:
            weights_scaled = torch.nn.functional.gumbel_softmax(weights_unscaled, tau=self.temperature, hard=False, dim=1)
        else:
            weights_scaled = self.activation(weights_unscaled)
        if self.pruning_threshold is not None:
            weights_scaled = (weights_scaled > self.pruning_threshold).float() * weights_scaled
        # Expand the weights so that they can be applied to each feature dimension.
        weights_repeated = weights_scaled.unsqueeze(-1).repeat(1, 1, 1, x.size(2))  # (B, O, H, F)
        x = x.unsqueeze(2)  # (B, O, 1, F)
        result = x * weights_repeated  # (B, O, H, F)
        result_aggregated = result.sum(axis=1)  # (B, H, F)
        result_flattened = result_aggregated.view(x.size(0), -1)  # (B, H*F)
        if self.return_weights:
            return result_flattened, weights_scaled
        else:
            return result_flattened

    def diversity_loss(self, weights):
        """
        Compute a diversity loss that encourages different heads to attend to different cells.
        weights: tensor of shape (B, O, H) - the attention weights.
        For each sample, normalize the weights along the observation dimension for each head,
        compute the cosine similarity between each pair of heads, and then average the off-diagonal values.
        """
        # weights: (B, O, H)
        B, O, H = weights.size()
        loss = 0.0
        for b in range(B):
            # A: (O, H)
            A = weights[b]  # (O, H)
            # Normalize each column (head) with L2 norm.
            A_norm = nn.functional.normalize(A, p=2, dim=0)  # (O, H)
            # Compute similarity matrix: (H, H)
            sim_matrix = torch.matmul(A_norm.t(), A_norm)
            # Remove diagonal elements
            off_diag = sim_matrix - torch.diag(torch.diag(sim_matrix))
            # Average of absolute similarities
            loss += off_diag.abs().mean()
        return loss / B

    
class UncertaintyWeightingLoss(nn.Module):
    def __init__(self, num_losses):
        super(UncertaintyWeightingLoss, self).__init__()
        self.log_vars = nn.Parameter(torch.ones(num_losses) * 0.1)

    def forward(self, losses):
        """
        Args:
            losses: List of loss tensors
        Returns:
            total_loss: Weighted sum of losses
        """
        total_loss = 0
        for i, loss in enumerate(losses):
            log_var = self.log_vars[i]
            precision = torch.exp(-log_var)
            total_loss += precision * loss + log_var
        return total_loss
    
