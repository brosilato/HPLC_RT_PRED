import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, rdFingerprintGenerator
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._param_validation import StrOptions, Interval

class SmilesToFingerPrintTransformer(BaseEstimator, TransformerMixin):
    """Scikit-Learn like transformer that generates a molecular fingerprint
    representation from SMILES strings
    """

    _parameter_contraints = {
        "fp_type": [StrOptions({"morgan", "rdkit", "maccs"})],
        "radius": [Interval(Integral, left=1, closed='left')],
        "fp_size": [Interval(Integral, left=1, closed='left')],
    }

    def __init_(self, fp_type: str="morgan", radius:int=3, counts: bool=False, fp_size: int=2048, dense: bool=False):
        self.fp_type = fp_type
        self.radius = radius
        self.counts = counts
        self.fp_size = fp_size
        self.dense = dense

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True # Set based on your needs
        return tags
    
    @staticmethod
    def _mol_generator(smile_string):
        mol = Chem.MolFromSmiles(smile_string)
        if mol is None:
            return Chem.MolFromSmiles('')
        return mol
    
    @staticmethod
    def _csr_from_bits_rdkit(fp):
        idx = np.array(fp.GetOnBits()).astype(np.uint32)
        fp_info = (np.array([1]*len(idx)), idx, np.array([0, len(idx)]))
        return sparse.csr_array(fp_info, dtype=np.int32, shape=(1,len(fp)))
    
    @staticmethod
    def _csr_from_counts_rdkit(fp):
        idx_2_vals = fp.GetNonzeroElements()
        idx = np.array(list(idx_2_vals.keys()))
        vals = np.array(list(idx_2_vals.values()))
        fp_info = (vals, idx, np.array([0, len(idx)]))
        return sparse.csr_array(fp_info, dtype=np.int32, shape=(1,fp.GetLength()))        
    
    def _transform_maccs(self, X):
        if self.dense:
            hstack = np.hstack
            vstack = np.vstack
            array = np.array
        else:
            hstack = sparse.hstack
            vstack = sparse.vstack
            array = sparse.csr_array
        
        transformed = []
        for j in range(X.shape[1]):
            transformed.append(vstack([array(self.fp_converter_(SmilesToFingerPrintTransformer._mol_generator(smile_string)), dtype=np.int32) for smile_string in X[:,j]]))
        if len(transformed) > 1:
            return hstack(transformed)
        return transformed[0]
    
    def _transform_to_dense(self, X):
        transformed = []
        for j in range(X.shape[1]):
            transformed.append(np.vstack([self.fp_converter_(SmilesToFingerPrintTransformer._mol_generator(smile_string)) for smile_string in X[:,j]]))
        if len(transformed) > 1:
            return np.hstack(transformed)
        return transformed [0]
    
    def _transform_to_sparse(self, X):
        if self.counts:
            csr_from_counts_rdkit = SmilesToFingerPrintTransformer._csr_from_counts_rdkit
        else:
            csr_from_counts_rdkit = SmilesToFingerPrintTransformer._csr_from_bits_rdkit

        transformed = []
        for j in range(X.shape[1]):
            transformed.append(sparse.vstack([csr_from_counts_rdkit(self.fp_converter_(SmilesToFingerPrintTransformer._mol_generator(smile_string))) for smile_string in X[:,j]]))
        if len(transformed) > 1:
            return sparse.hstack(transformed)
        return transformed [0]
    
    def fit(self, X, y=None):
        self._validate_params()
        if self.fp_type == "maccs" and self.counts:
            raise ValueError(
                ("SmilesToFingerPrintTransformer can not" 
                 + " generate counts of MACCS fingerprints. Set counts=False"
                 + " or choose fp_type='morgan' or fp_type= 'rdkit'")
            )
 
        self._mol_generator_ = Chem.MolFromSmiles
        if self.fp_type is 'maccs':
            self.fp_converter_ = staticmethod(MACCSkeys.GenMACCSKeys)
            self._transform_ = self._transform_maccs 
        else:
            # Then self.fp_type is 'rdkit' or 'morgan'
            if self.fp_type is 'morgan':
                self._fp_generator_ = (rdFingerprintGenerator.
                                       GetMorganGenerator(
                                           radius=self.radius, 
                                           fpSize=self.fp_size
                                           )
                                        )
            else:
                # Then self.fp_type is 'rdkit'
                self._fp_generator_ = (rdFingerprintGenerator.
                                       GetRDKitFPGenerator(
                                           maxPath=self.radius, 
                                           fpSize=self.fp_size
                                           )
                                        )
            # Let's define the fingerprint generator based on counts/bits
            if self.counts:
                self.fp_converter_ = self._fp_generator_.GetCountFingerprint
            else:
                # Then not self.counts (bits) 
                self.fp_converter_ = self._fp_generator_.GetFingerprint
            # Let's define te proper transformer function based on dense/sparse
            if self.dense:
                self._transform_ = self._transform_to_dense
            else:
                # Then not self.dense (sparse)
                self._transform_ = self._transform_to_sparse
            
        return self
    
    def transform(self, X, y=None):
        self.check_is_fitted()
        return (self._transform_(X), y)
        




