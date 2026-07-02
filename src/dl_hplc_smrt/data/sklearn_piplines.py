from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from dl_hplc_smrt.data.data_transformers import SmilesToMolTransformer as STM
from dl_hplc_smrt.data.data_transformers import MolToFingerPrintTransformer as MTFP

fps_pipeline = Pipeline([    
    ("mol_converter", STM()),
    ("fp_transformer", MTFP()),
    ("standard_scaler", StandardScaler()),
    ("variance_thres", VarianceThreshold()),# Remove constant all-zero features    
    ])
