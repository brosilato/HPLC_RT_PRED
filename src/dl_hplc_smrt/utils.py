import itertools
from pathlib import Path
from typing import Literal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from sklearn import datasets
from sklearn.base import clone
import tqdm
import torch
from torch.nn.modules.loss import _Loss
from dl_hplc_smrt.models.mlp_models import BaseNet


def inchi_to_smiles(inchi_string: str) -> str | None:
    """Converts InChI strings into SMILES strings one at a time

    Args:
        str (inchi_string): InChI string of the molecule

    Returns:
        str: Corresponding SMILES string of the molecule
    """
    mol = Chem.inchi.MolFromInchi(inchi_string)
    if mol is None:
        return None
    smiles_string = Chem.MolToSmiles(mol)
    return smiles_string

class OptimizeParametersValidation():
    """Uses a train set and validation set to find the best model using a user
    provided model and all combinations of parameters from a parm_dict.

    Args:
            model (_type_): Instance of the model to optimize.
            param_grid (dict[str, Any]): Parameters to try. All possible
                combinations are tested.
            scoring (_type_): The score to meassure the quality of the models.
            keep_lowest (bool, optional): Whether to keep the optimized model
                with the lowest score (is lower better?). Defaults to True.
            keep_highest (bool, optional): Whether to keep the optimized model
                with the highest score (is higher better?). Defaults to False.
    """

    def __init__(
            self,
            model,
            param_grid,
            scoring,
            keep_lowest: bool=True,
            keep_highest: bool=False
            ):
        self.model = model
        self.param_grid = param_grid
        self.scoring = scoring
        self.keep_lowest = keep_lowest
        self.keep_highest = keep_highest

    def fit(self, x_train, y_train, x_val, y_val)-> self:
        """Train a series of models on xtrain and y_train. Then evaluates their
        performance on both the train set and the validation (x_val, y_val)
        set.  

        Args:
            x_train (np.array): Train features set
            y_train (np.array): Train target set
            x_val (np.array): Validation features set
            y_val (np.array): Validation target set

        Returns:
            self: Returns the instance
        """
        self.x_train_ = x_train
        self.y_train_ = y_train
        self.x_val_ = x_val
        self.y_val_ = y_val
        lowest_score = np.inf
        highest_score = -np.inf
        self.lowest_model_ = None
        self.highest_model_ = None


        keys = self.param_grid.keys()
        values = self.param_grid.values()
        self.params_ = [
            dict(zip(keys, value_group)) 
            for value_group in list(itertools.product(*values))
            ]
        
        train_scores = []
        val_scores = []
        for param_group in tqdm.tqdm(self.params_):
            self.model.set_params(**param_group).fit(self.x_train_, self.y_train_)
            train_scores.append(self.scoring(self.y_train_, self.model.predict(self.x_train_)))
            current_score =  self.scoring(self.y_val_, self.model.predict(self.x_val_))
            val_scores.append(current_score)
        
            if current_score < lowest_score:
                self.lowest_params_ = self.model.get_params()
                lowest_score = current_score
                if self.keep_lowest:
                    self.lowest_model_ = clone(self.model)
        
            if current_score > highest_score:
                self.highest_params_ = self.model.get_params()
                highest_score = current_score
                if self.keep_highest:
                    self.highest_model_ = clone(self.model)


        self.results_ = pd.DataFrame(self.params_)
        self.results_["train score"] = train_scores
        self.results_["validation score"] = val_scores
        return self
    
    def get_results(self) -> pd.DataFrame | None:
        """Returns a pandas DataFrame summarizing the results of the
        optimization.

        Returns:
            pd.DataFrame: Results of the optimization.
        """
        try:
            return self.results_
        except NameError:
            print("OptimizeParametersValidation has not yet been succesfully fitted")
            return None


def evaluate_sklearn_model(model, datasets: dict['str', tuple|list], scores: dict[str, callable]) -> pd.DataFrame:
    """Evaluates a sklearn model on multiple datasets and scores.
    
    Args:
        model: The sklearn model to evaluate.
        datasets (dict): A dictionary mapping dataset names to (X, y) tuples.
        scores (dict): A dictionary mapping score names to sklearn-like score functions.

    Returns:
        pd.DataFrame: A DataFrame containing the evaluation results.
    """
    evaluation = { "dataset": list(datasets.keys())}
    for score_name, score_func in scores.items():
        evaluation.update({
            score_name: [score_func(datasets.get(dataset)[1], model.predict(datasets.get(dataset)[0])) for dataset in evaluation["dataset"]]
        })

    return pd.DataFrame(evaluation)

def plot_results(y_targets, y_pred, title=None, limits=(400,1600)) -> plt.Figure:
    """Plots predicted vs target with error tolerance bands.
    
    Creates a scatter plot comparing predicted and target retention times, with
    reference lines showing acceptable error ranges (1, 2, and 3 minutes).
    
    Args:
        y_targets (array-like): Target retention time values.
        y_pred (array-like): Predicted retention time values.
        title (str, optional): Title for the plot. Defaults to None.
        limits (tuple, optional): (min, max) limits for both axes. Defaults to (400, 1600).
    
    Returns:
        matplotlib.figure.Figure: The figure object containing the plot.
    """
    plt.figure(figsize=(10,10))
    sns.scatterplot(x=y_targets, y=y_pred,alpha=0.5)
    plt.ylim(*limits)
    plt.xlim(*limits)
    sns.lineplot(x=[0, 100000], y=[0, 100000], color="cyan", alpha=0.3)
    # Under one minute (60 s)
    sns.lineplot(x=[0, 100060], y=[-60, 100000], color="green", alpha=0.3 )
    sns.lineplot(y=[0, 100060], x=[-60, 100000], color="green", alpha=0.3 )
    # Under two minute (120 s)
    sns.lineplot(x=[0, 100120], y=[-120, 100000], color="yellow", alpha=0.3 )
    sns.lineplot(y=[0, 100120], x=[-120, 100000], color="yellow", alpha=0.3 )
    # Under three minute (180 s)
    sns.lineplot(x=[0, 100180], y=[-180, 100000], color="red", alpha=0.3 )
    sns.lineplot(y=[0, 100180], x=[-180, 100000], color="red", alpha=0.3 )
    plt.title(title)
    plt.xlabel("Target")
    plt.ylabel("Predictions")
    return plt.gcf()

def evaluate_pytorch_regressor(model, dataloaders: dict['str', tuple|list], scores: dict[str, callable]=None, device: Literal['cuda', 'cpu']='cpu') -> pd.DataFrame:
    """Evaluates a PyTorch regression model on a given dataset.

    Args:
        model: The PyTorch model to evaluate.
        dataloaders: A dictionary mapping dataset names to DataLoader instances.
        scores (dict): A dictionary mapping score names to sklearn-like score functions.
        device (Literal['cuda', 'cpu']): The device (CPU or GPU) to perform computations on.

    Returns:
        pd.DataFrame: A DataFrame containing the evaluation results.
    """
    model.eval()
    model.to(device)   
    evaluation = { "dataset": list(dataloaders.keys())}
    with torch.no_grad():
        for dataset in evaluation["dataset"]:
            y_true = []
            y_pred = []
            for X_batch, y_batch in dataloaders[dataset]:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                predictions = model(X_batch)
            
                y_true.append(y_batch.detach().squeeze().cpu().numpy())
                y_pred.append(predictions.detach().squeeze().cpu().numpy())
            y_true = np.hstack(y_true)
            y_pred = np.hstack(y_pred)
            for score_name, score_func in scores.items():
                evaluation[score_name] = evaluation.get(score_name, []) + [score_func(y_true, y_pred)]

    return pd.DataFrame(evaluation)

def train_with_early_stopping(
    model: BaseNet, 
    train_loader: torch.utils.data.DataLoader, 
    val_loader: torch.utils.data.DataLoader, 
    criterion: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    checkpoint_path: str | Path,
    delta: float=0.0,
    eval_criterion: torch.nn.Module | None = None, 
    scheduler: torch.optim.lr_scheduler._LRScheduler=None,
    evals_per_epoch: int=10,
    patience: int=5, 
    epochs: int=100, 
    device: Literal['cuda', 'cpu']="cpu",
    init_epoch: int=0,
    init_best_eval_loss: float=np.inf,
):
    """
    Trains a PyTorch model with early stopping based on validation loss.
    
    Parameters:
        model: The PyTorch neural network to train.
        train_loader: DataLoader for the training dataset.
        val_loader: DataLoader for the validation dataset.
        criterion: The loss function (e.g., nn.CrossEntropyLoss()).
        optimizer: The optimizer (e.g., optim.Adam()).
        scheduler: The learning rate scheduler (e.g., optim.lr_scheduler.StepLR()).
        evals_per_epoch: How often to evaluate the model during each training epoch.
        patience: How many epochs to wait for improvement before stopping.
        epochs: Maximum number of epochs to train.
        device: Device to run training on ('cuda' or 'cpu').
        
    Returns:
        model: The trained model with the best weights restored.
        history: Dictionary containing training and validation loss history.
    """
    if eval_criterion is None:
        eval_criterion = criterion
    model.to(device)
    best_eval_loss = init_best_eval_loss
    best_epoch_eval_loss = best_eval_loss
    patience_counter = 0

    eval_list = [int(len(train_loader)/(evals_per_epoch)*ii) for ii in range(1, evals_per_epoch)] + [len(train_loader)]
    
    history = {'epoch': [], 'mini_batch': [], 'train_loss': [], 'val_loss': [], 'val_eval_loss': []}
    
    for epoch in range(init_epoch, init_epoch + epochs):
        # --- TRAINING PHASE ---
        running_train_loss = 0.0
        mini_batch_counter = 0
        samples_counter = 0
        for inputs, targets in train_loader:
            #best_epoch_loss = best_loss
            model.train()
            inputs, targets = inputs.to(device), targets.to(device)         
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.view_as(outputs))  # Ensure targets are of shape (batch_size, 1)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)
            samples_counter += inputs.size(0)
            mini_batch_counter += 1

            if mini_batch_counter in eval_list:
                model.eval()
                running_val_loss = 0.0
                running_val_eval_loss = 0.0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        eval_loss = eval_criterion(outputs, targets.view_as(outputs))  # Ensure targets are of shape (batch_size, 1)
                        loss = criterion(outputs, targets.view_as(outputs))  # Ensure targets are of shape (batch_size, 1)
                        running_val_eval_loss += eval_loss.item() * inputs.size(0)
                        running_val_loss += loss.item() * inputs.size(0)    
                    val_loss = running_val_loss / len(val_loader.dataset)
                    val_eval_loss = running_val_eval_loss / len(val_loader.dataset)
                    train_loss = running_train_loss / samples_counter
                    history['epoch'].append(epoch)
                    history['mini_batch'].append(mini_batch_counter)
                    history['train_loss'].append(train_loss)
                    history['val_loss'].append(val_loss)
                    history['val_eval_loss'].append(val_eval_loss)

                print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Eval Loss: {val_eval_loss:.4f}")
                # Peek and save model---
                if val_eval_loss < best_eval_loss:
                    print(f"\tValidation evaluation loss improved from {best_eval_loss:.4f} to {val_eval_loss:.4f}. Saving model checkpoint.")
                    best_eval_loss = val_eval_loss
                    model.save_checkpoint(
                        checkpoint_path, 
                        optimizer=optimizer, 
                        scheduler=scheduler, 
                        epoch=epoch, 
                        train_loss=train_loss, 
                        val_loss=val_loss,  
                        comments=f"Best model at epoch {epoch} with val_loss {val_loss:.4f} and val_eval_loss {val_eval_loss:.4f}",                        
                        val_eval_loss=val_eval_loss,
                        )

        new_best_epoch_eval_loss = best_eval_loss
        print(f"best_epoch_eval_loss: {best_epoch_eval_loss:.3f}  || best_eval_loss: {best_eval_loss:.3f}")
        if new_best_epoch_eval_loss < (best_epoch_eval_loss - delta):   
            patience_counter = 0
            # update best epoch evaluation loss
            best_epoch_eval_loss = new_best_epoch_eval_loss
        else:
            patience_counter += 1
        # Should we stop the training early?
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}. Best model saved ({str(checkpoint_path)}) but not loaded.")
            break
        print(f"best_epoch_eval_loss: {best_epoch_eval_loss:.3f}  || best_eval_loss: {best_eval_loss:.3f}")
        print(patience_counter)

        # Update the scheduler if provided
        if scheduler is not None:
            scheduler.step()              
                
    return pd.DataFrame(history)
