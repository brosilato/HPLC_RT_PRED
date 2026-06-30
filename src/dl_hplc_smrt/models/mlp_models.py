import abc
from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseModel(nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, x):
        """Every subclass must implement the forward pass."""
        pass

    @abc.abstractmethod
    def get_hyperparams(self) -> dict:
        """
        Abstract method. Must be implemented by sibclasses to return a dictionary of hyperparameters.
        """
        pass 

    def freeze_model(self):
        """
        Freezes the model parameters, preventing them from being updated during training.
        """
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze_model(self):
        """
        Unfreezes the model parameters, allowing them to be updated during training.
        """
        for param in self.parameters():
            param.requires_grad = True

class LinearLayerPerceptron(BaseModel):
    def __init__(
            self, 
            input_dim:int,
            output_dim: int=1,
            dropout_prob: float=0.5
            ) -> None:
        super(LinearLayerPerceptron, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob

        self.fc = nn.Linear(self.input_dim, self.output_dim)
        self.bn = nn.BatchNorm1d(self.output_dim)
        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x

    def get_hyperparams(self):
        hyperparams = {
            "input_dim": self.input_dim, 
            "output_dim": self.output_dim,
            "dropout_prob": self.dropout_prob, 
        }
        return hyperparams

class MultiLayerPerceptron(BaseModel):
    def __init__(
            self, 
            num_features:int,
            linear_layers_dim: list[int]=[32,32],
            dropout_prob: float=0.5,
            activation: Literal['relu', 'elu', 'leaky']="relu"
            ) -> None:
        super(MultiLayerPerceptron, self).__init__()
        
        self.num_features = num_features
        self.linear_layers_dim = linear_layers_dim
        self.dropout_prob = dropout_prob
        self.activation = activation

        if len(linear_layers_dim) < 1:
            raise ValueError("At least one hidden layer is required")
        
        if self.activation=="relu":
            self.activation_layer = nn.ReLU()
        elif self.activation=="elu":
            self.activation_layer = nn.ELU()
        elif self.activation=="leaky":
            self.activation_layer = nn.LeakyReLU()

        self.linear_layers = nn.ModuleList()

        previous_dim = self.num_features
        for linear_dim in self.linear_layers_dim:
            self.linear_layers.append(LinearLayerPerceptron(previous_dim, linear_dim, dropout_prob=self.dropout_prob))
            previous_dim = linear_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Loop through the dynamic layers sequentially
        for linear_layer in self.linear_layers[:-1]:
            x = linear_layer(x)
            x = self.activation(x)

        # Compute final prediction
        x = self.linear_layers[-1](x)
        return x
    
    def get_hyperparams(self):
        hyperparams = {
            "num_features": self.num_features, 
            "linear_layers_dim": self.linear_layers_dim,
            "dropout_prob": self.dropout_prob, 
            "activation": self.activation,
        }
        return hyperparams
    
class ComplexMultilayerPerceptron(BaseModel):
    def __init__(
            self, num_features:int,
            encoder_hidden_layers_dim: list[int]=[32,32],
            processer_hidden_layers_dim: list[int]= [16], 
            output_dim: int=1,
            dropout_prob: float=0.5,
            activation: Literal['relu', 'elu', 'leaky']="relu"
            ) -> None:
        """Simple fully coneected neural network

        Args:
            num_features (int): _description_
            hidden_layers_dim (List[int], optional): _description_. Defaults to [64,32].
            output_dim (int, optional): _description_. Defaults to 1.
            dropout_prob (float, optional): _description_. Defaults to 0.5.
            activation (Literal['relu', 'elu', 'leaky'], optional): _description_. Defaults to "relu".

        Raises:
            ValueError: _description_
        """
        super(MultiLayerPerceptron, self).__init__()

        self.num_features = num_features
        self.encoder_hidden_layers_dim = encoder_hidden_layers_dim
        self.processer_hidden_layers_dim = processer_hidden_layers_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob
        self.activation = activation

        
        if len(self.encoder_hidden_layers_dim) < 1:
            raise ValueError("At least one hidden layer is required for the encoder")
        if len(self.processer_hidden_layers_dim) < 1:
            raise ValueError("At least one hidden layer is required for the decoder")
        
        
        if self.activation=="relu":
            self.activation_layer = nn.ReLU()
        elif self.activation=="elu":
            self.activation_layer = nn.ELU()
        elif self.activation=="leaky":
            self.activation_layer = nn.LeakyReLU()
        
        self.encoder = MultiLayerPerceptron(
            num_features=self.num_features, 
            linear_layers_dim=self.encoder_hidden_layers_dim,
            dropout_prob=self.dropout_prob,
            activation=self.activation
            )
        
        self.processer = MultiLayerPerceptron(
            num_features=self.num_features, 
            linear_layers_dim=self.processer_hidden_layers_dim,
            dropout_prob=self.dropout_prob,
            activation=self.activation
            )
        
        self.output_layer = nn.Linear(
            self.decoder_hidden_layers_dim[-1], 
            self.output_dim
            )     
    
    def forward(self, x):
        # Encode the molecules
        x = self.encoder(x)
        x = self.activation_layer(x)

        # Predict the targets
        x = self.processer(x)
        x = self.activation_layer(x)
        x = self.output_layer(x)
        
        return x
    
    def get_hyperparams(self):
        hyperparams = {
            "num_features": self.num_features, 
            "encoder_hidden_layers_dim": self.encoder_hidden_layers_dim,
            "processer_hidden_layers_dim": self.processer_hidden_layers_dim,
            "output_dim": self.output_dim,
            "dropout_prob": self.dropout_prob, 
            "activation": self.activation,
        }
        return hyperparams
    
    
