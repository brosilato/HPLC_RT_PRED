import os
from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
import yaml
from dl_hplc_smrt.data.data_transformers import SmilesToMolTransformer as STM
from dl_hplc_smrt.data.data_transformers import MolToFingerPrintTransformer as MTFP

FPS_PIPELINE = Pipeline([    
    ("mol_converter", STM()),
    ("fp_transformer", MTFP()),
    ("standard_scaler", StandardScaler()),
    ("variance_thres", VarianceThreshold()),# Remove constant all-zero features    
    ])

class RF_FPS_PIPELINE:
    def __init__(self, config_file: Path):
        """ Loads the configuration file
        """
        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)

        self.model_path = Path(self.config["model"]["output_dir"]) / self.config["model"]["model_name"]

    def build_pipeline(self):
        """Creates a sckit-learn pipeline: preprocessing (fingerprints generation) + molel (Random Forest Regressor)
        """
        pipeline = FPS_PIPELINE.steps.append(("rfr", RFR()))
        
        pipeline_params = self.config["hyperparameters"]
        pipeline.set_params(**pipeline_params)
        return pipeline
    
    def train_and_save(self, X=None, y=None):
        """Trains the full pipeline and saves the artifact to disk."""
        pipeline = self.build_pipeline()
        if X is None:
            self.x_train_path = Path(self.config["datasets"]["train"]["X_path"]) 
            if not os.path.exists(self.x_train_path):
                raise FileNotFoundError(f"No saved feature matrix found at {self.x_train_path}")
            self.X = joblib.load(self.x_train_path)
        else:
            print("Using provided feature matrix for training.")
            self.X = X

        if y is None:
            self.y_train_path = Path(self.config["datasets"]["train"]["y_path"])
            if not os.path.exists(self.y_train_path):
                raise FileNotFoundError(f"No saved target vector found at {self.y_train_path}")
            self.y = joblib.load(self.y_train_path)
        else:
            print("Using provided target vector for training.")
            self.y = y

        print("Training the scikit-learn pipeline...")
        pipeline.fit(self.X, self.y)

        # Ensure output directory exists
        os.makedirs(self.model_path.parent, exist_ok=True)
        
        # Save the complete pipeline object
        joblib.dump(pipeline, self.model_path)
        print(f"Pipeline successfully saved to: {self.model_path}")
        return pipeline

    def load_pipeline(self):
        """Loads the trained pipeline artifact for production inference."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No saved model found at {self.model_path}")
        
        print(f"Loading pipeline from: {self.model_path}")
        return joblib.load(self.model_path)
