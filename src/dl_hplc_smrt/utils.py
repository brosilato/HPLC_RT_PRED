import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from sklearn.base import clone
import tqdm


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


def evaluate_sklearn_model(model, datasets: dict['str', tuple|list], scores: dict[str, callable]):
    """Evaluates a sklearn model on multiple datasets and scores.
    
    Args:
        model: The sklearn model to evaluate.
        datasets (dict): A dictionary mapping dataset names to (X, y) tuples.
        scores (dict): A dictionary mapping score names to score functions.

    Returns:
        pd.DataFrame: A DataFrame containing the evaluation results.
    """
    evaluation = { "dataset": list(datasets.keys())}
    for score_name, score_func in scores.items():
        evaluation.update({
            score_name: [score_func(datasets.get(dataset)[1], model.predict(datasets.get(dataset)[0])) for dataset in evaluation["dataset"]]
        })

    return pd.DataFrame(evaluation)

def plot_results(y_targets, y_pred, title=None, limits=(400,1600)):
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


