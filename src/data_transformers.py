import numpy as np
from numbers import Integral 
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, rdFingerprintGenerator
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._param_validation import StrOptions, Interval
import sklearn.utils.validation

class SmilesToFingerPrintTransformer(BaseEstimator, TransformerMixin):
    """Fingerprint (FP) generation

    Scikit-Learn like transformer that generates a molecular fingerprint
    representation from SMILES strings. This transformer is indeed an adapter
    for the open-source library RDKit. Three main types of fingerprints can
    be genreted: Morgan (circular) fingerprints, RDKit (path) fingerprints 
    and MACCS (substructure keys) fingerprints.

    For Morgan and RDKit fingerprints, the size of the fingerprint can be
    selected, while for MACCS fps the size is fix and any selection will be
    ignored.

    For Morgan snd RDKit fingerprints, one can select whether counts or bits
    are output. MACCS only produces bitvectors and asking for MACCS counts
    will rise an exception.

    For all fingerprints, the output can be requested to be a numpy array
    or a scipy sparse array.

    The generation of mol objects from smiles with RDKit, a necessary step,
    can often help. This issue is handled in a not satisfactory, funky way by
    generating an all 0 fp.

    Parameters
    ----------
    fp_type : {'morgan', 'rdkit', 'maccs'}, default='morgan'
        Type of fingerprint to output.
            'morgan': 
                Circular, Morgan fingerprints.
            'rdkit': 
                Path-based, RDKit fingerprints.
            'maccs': 
                Substructure key, MACCS fingerprints.
    
    radius : int, default=3
        When computing Morgan fps, this is the maximum radius of the molecular
        features included ni the fingerprints. When computing RDKit
        fingerprints, it corresponds to the maximum size of the paths
        considered. When computing MACCS fingerprints, this value is ignored.

    counts : bool, default=False
        If true, fingerprint will contain integer counts of how many times the
        moieties associated to each fingerpring appear in each molecule. If
        False, a one-hot encoding for fingerprint will be used. MACCS fps do
        not provide counts and setting this parameter to True will result in an
        exception being rise.

    fp_size : int, default=2048
        The size of the fingerprint (number of counts/bits). For MACCS fps, this
        parameter is ignored since the always produce 167-sized vectors. 

    counts : dense, default=False
        If True, the output of the transformation (the fingerprint matrix) will
        be a numpy array. If false, the ouput will be a sparse scipy csr_array.

    Attributes
    ----------
    num_columns_ : int 
        Expected number of columns of the arrays to be processed by transform.
        They should have the same noumber of feature columns (SMILES columns)
        as the array used during fitting.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.   

    failed_mol_ : list[str]
        List of molecular string whose conversion to mol object failed. 
    """

    _parameter_constraints = {
        "fp_type": [StrOptions({"morgan", "rdkit", "maccs"})],
        "radius": [Interval(Integral, left=1, right=None, closed='left')],
        "fp_size": [Interval(Integral, left=1, right=None, closed='left')],
    }

    def __init__(self, fp_type: str="morgan", radius: int=3, counts: bool=False, fp_size: int=2048, dense: bool=False):
        self.fp_type = fp_type
        self.radius = radius
        self.counts = counts
        self.fp_size = fp_size
        self.dense = dense
      
    def _mol_generator(self, smiles_string):
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            self.failed_mol_.append(smiles_string)
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
            transformed.append(vstack([array(self.fp_converter_(self._mol_generator(smile_string)), dtype=np.int32) for smile_string in X[:,j]]))
        if len(transformed) > 1:
            return hstack(transformed)
        return transformed[0]
    
    def _transform_to_dense(self, X):
        transformed = []
        for j in range(X.shape[1]):
            transformed.append(np.vstack([self.fp_converter_(self._mol_generator(smile_string)) for smile_string in X[:,j]]))
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
            transformed.append(sparse.vstack([csr_from_counts_rdkit(self.fp_converter_(self._mol_generator(smile_string))) for smile_string in X[:,j]]))
        if len(transformed) > 1:
            return sparse.hstack(transformed)
        return transformed [0]
    
    def fit(self, X, y=None):
        """Fit the model with X.

        Args:
            X (array_like of shape (n_samples, n_features)):Training data,
            where `n_samples` (number of SMILES per column) is the number of
            samples and `n_features` is the number of features (columns
            containing SMILES).
            y (array-like, optional): Matrix of the targets/labels. Ignored.
            Defaults to None.

        Raises:
            ValueError: If MACCS fingerprints are requested (fp_type = 'maccs')
            together with counts (counts=True).

        Returns:
            self: Object. The fitted instance.
        """
        self._validate_params()
        self.num_columns_ = X.shape[1]
        if self.fp_type == "maccs" and self.counts:
            raise ValueError(
                ("SmilesToFingerPrintTransformer can not" 
                 + " generate counts of MACCS fingerprints. Set counts=False"
                 + " or choose fp_type='morgan' or fp_type= 'rdkit'")
            )
 
        if self.fp_type == 'maccs':
            self.fp_converter_ = staticmethod(MACCSkeys.GenMACCSKeys)
            self._transform_ = self._transform_maccs 
        else:
            # Then self.fp_type is 'rdkit' or 'morgan'
            if self.fp_type == 'morgan':
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
            if self.counts and self.dense:
                self.fp_converter_ = self._fp_generator_.GetCountFingerprintAsNumPy
                self._transform_ = self._transform_to_dense
            elif self.counts and not self.dense:
                self.fp_converter_ = self._fp_generator_.GetCountFingerprint
                self._transform_ = self._transform_to_sparse
            elif not self.counts and self.dense:
                self.fp_converter_ = self._fp_generator_.GetFingerprintAsNumPy
                self._transform_ = self._transform_to_dense
            elif not self.counts and not self.dense:
                self.fp_converter_ = self._fp_generator_.GetFingerprint
                self._transform_ = self._transform_to_sparse

        return self
    
    def transform(self, X, y=None):
        """Computes the fingerprints of the passed molecules.

        Args:
            X (array_like of shape (n_samples, n_features)):Training data,
            where `n_samples` (number of SMILES per column) is the number of
            samples and `n_features` is the number of features (columns
            containing SMILES).
            y (array-like, optional): Matrix of the targets/labels. Ignored.
            Defaults to None.

        Returns:
            np.array | scipy.sparse.csr_array: Matrix with the fingerprints.
        """
        sklearn.utils.validation.check_is_fitted(self)
        if X.shape[1] != self.num_columns_:
            raise ValueError(
                f"The feature matrix has an invalid shape. "
                f"Expected {self.num_columns_} columns, but got {X.shape[1]}."
            ) 
        self.failed_mol_ = list()
        return self._transform_(X)
    
    
    
    
        




