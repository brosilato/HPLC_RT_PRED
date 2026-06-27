import pytest
from contextlib import nullcontext
import numpy as np
import numpy.testing as npt
from scipy import sparse
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from data_transformers import SmilesToFingerPrintTransformer as STFP


x_smiles_2col = np.array([["CC1=CC=C(C=C1)O","CCC(C)CC(C)CC"],
            ["CC1=CC(=CO1)C", "CCC(C)(C)CC"],
            ["C1=CC=C(C=C1)F","CC1CCC(CC1)C"],])

x_smiles_1col = np.array([["CC1=CC=C(C=C1)O"],
            ["CC1=CC(=CO1)C"],
            ["C1=CC=C(C=C1)F"],])

x_smiles_1col_falty = np.array([["CC1=CC=C(C=C1)O"],
                                ["ThisIsNotaProperSMILES"],
                                ["C1=CC=C(C=C1)F"],
                                ["NeitherItIsThisOne"]])

@pytest.fixture(params=[x_smiles_1col, x_smiles_2col, x_smiles_1col_falty])
def feature_matrix(request):
    return request.param

@pytest.fixture()
def STFP_transformer():
    return STFP()

@pytest.fixture()
def maccs_fp_generator():
    return MACCSkeys.GenMACCSKeys
    

class TestSmilesToFingerPrintTransformer:
    @pytest.mark.parametrize(
    "transformer, expected_exception",
    [
        (STFP(), pytest.raises(ValueError)),     # Not fitted (exception expected)
        (STFP().fit(x_smiles_2col), nullcontext()),                 # Fitted (no exception expected)        
    ], ids=["not_fitted_triggers_exception","fitted_so_no_exception"]
)
    def test_fit_before_transform(self, transformer, expected_exception):
        """This method tests that an exception is riased if using the transform
        method without fitting the method first. And that such exception is not
        raised if the transformer has been fitted"""
        with expected_exception:
            transformer.transform(x_smiles_2col)

    def test_maccs_counts_not_available(self):
        """This method tests that an exception is riased when MACCS fps are
        requested together with counts instead of bits"""
        with pytest.raises(ValueError):
            STFP(fp_type="maccs", counts=True).fit(x_smiles_2col)
    
    @pytest.mark.parametrize(
        "options_dict, output_type",
        [
            ({"fp_type": "maccs", "dense": True}, np.ndarray),
            ({"fp_type": "rdkit", "dense": True}, np.ndarray),
            ({"fp_type": "morgan", "dense": True}, np.ndarray),
            ({"fp_type": "maccs", "dense": False}, sparse.csr_array),
            ({"fp_type": "rdkit", "dense": False}, sparse.csr_array),
            ({"fp_type": "morgan", "dense": False}, sparse.csr_array),
        ], ids=["maccs_dense_fps", "rdkit_dense_fps", "morgan_dense_fps",
                "maccs_sparse_fps", "rdkit_sparse_fps", "morgan_sparse_fps"])
    def test_output_type(self, STFP_transformer, options_dict, output_type, feature_matrix):
        """ This method tests that the output type (sparse/dense) is consistent
        with the requested output using the option `dense`"""
        STFP_transformer.set_params(**options_dict)
        produced_output = STFP_transformer.fit_transform(feature_matrix)
        assert isinstance(produced_output, output_type)
    
    @pytest.mark.parametrize(
        "options_dict",
        [
            ({"fp_type": "morgan", "dense": True, "fp_size": 32}),
            ({"fp_type": "rdkit", "dense": True, "fp_size": 32}),
            ({"fp_type": "morgan", "dense": True, "fp_size": 512}),
            ({"fp_type": "rdkit", "dense": True, "counts": True, "fp_size": 512}),
            ({"fp_type": "morgan", "dense": False, "counts": True, "fp_size": 32}),
            ({"fp_type": "rdkit", "dense": False, "fp_size": 32}),
            ({"fp_type": "morgan", "dense": False, "fp_size": 512}),
            ({"fp_type": "rdkit", "dense": False, "fp_size": 512}),
        ], ids=["morgan_size_32_dense", "rdkit_size_32_dense", "morgan_size_512_dense", "rdkit_size_512_dense_counts",
                "morgan_size_32_sparse_counts", "rdkit_size_32_sparse", "morgan_size_512_sparse", "rdkit_size_512_sparse",])
    def test_morgan_rdkit_output_dims(self, STFP_transformer, options_dict, feature_matrix):
        """ This method tests that the transformed feature matrices have the
        expected size, when Morgan or RDKit fps are requested"""
        STFP_transformer.set_params(**options_dict)
        produced_output = STFP_transformer.fit_transform(feature_matrix)
        expected_dimensions = (feature_matrix.shape[0], feature_matrix.shape[1]*options_dict['fp_size'])
        #assert npt.assert_allclose(produced_output.shape, expected_dimensions)
        assert produced_output.shape == expected_dimensions
        
    @pytest.mark.parametrize(
        "options_dict",
        [
            ({"fp_type": "maccs", "dense": True, "fp_size": 32}),
            ({"fp_type": "maccs", "dense": True, "fp_size": 512}),
            ({"fp_type": "maccs", "dense": False, "fp_size": 32}),
            ({"fp_type": "maccs", "dense": False, "fp_size": 512}),
        ], ids=["maccs_size_32_dense", "maccs_size_512_dense",
                "maccs_size_32_sparse", "maccs_size_512_sparse",])
    def test_maccs_output_dims(self, STFP_transformer, options_dict, feature_matrix):
        """ This method tests that the transformed feature matrices have the
        expected size, when MACSS fps are requested. Noted that for MACCS fps
        the parameter `fp_size` is silently ignored and the produced fps are
        always of size 167"""
        STFP_transformer.set_params(**options_dict)
        produced_output = STFP_transformer.fit_transform(feature_matrix)
        expected_dimensions = (feature_matrix.shape[0], feature_matrix.shape[1]*167)
        #assert npt.assert_allclose(produced_output.shape, expected_dimensions)
        assert produced_output.shape == expected_dimensions
    
    @pytest.mark.parametrize(
            "options_dict",
            [
            ({"fp_type": "maccs", "dense": False}),
            ({"fp_type": "maccs", "dense": True}),
            ], ids=["sparse", "dense"]
    )    
    def test_maccs_transformer_selected(self, STFP_transformer, options_dict):
        """This test checks that the right transform function
        (_transform_maccs) is selected"""
        my_transformer = STFP_transformer.set_params(**options_dict).fit(x_smiles_1col)
        assert my_transformer._transform_ == my_transformer._transform_maccs

    @pytest.mark.parametrize(
            "options_dict",
            [
            ({"fp_type": "maccs", "dense": False}),
            ({"fp_type": "maccs", "dense": True}),
            ], ids=["sparse", "dense"]
    )    
    @pytest.mark.parametrize(
        "features",
        [
            np.array([["CC1=CC=C(C=C1)O"]]),
            np.array([["C1=CC=C(C=C1)F"]]),
        ], ids=["one column of SMILES", "two columns of SMILES"]
    )
    def test_transformer_reproduces_rdkit_for_maccs_fps(self, STFP_transformer, maccs_fp_generator, options_dict, features):
        my_transformer = STFP_transformer.set_params(**options_dict).fit(features)
        transformer_fp = my_transformer.transform(features)[0]
        if not options_dict["dense"]:
            transformer_fp = transformer_fp.toarray()
        rdkit_fp = maccs_fp_generator(Chem.MolFromSmiles(features[0][0])).ToList()
        npt.assert_array_equal(transformer_fp, rdkit_fp)

    @pytest.mark.parametrize(
            "options_dict, fp_generator",
            [
            ({"fp_type": "morgan", "dense": False, "fp_size":16, "counts": False, "radius": 3}, 
             rdFingerprintGenerator.GetMorganGenerator),
            ({"fp_type": "morgan", "dense": True, "fp_size":32, "counts": False, "radius": 4}, 
             rdFingerprintGenerator.GetMorganGenerator),
             ({"fp_type": "morgan", "dense": False, "fp_size":16, "counts": True, "radius": 7}, 
             rdFingerprintGenerator.GetMorganGenerator),
            ({"fp_type": "morgan", "dense": True, "fp_size":32, "counts": True, "radius": 2}, 
             rdFingerprintGenerator.GetMorganGenerator),
            ], ids=["morgan sparse bits", "morgan dense bits", "morgan sparse counts", "morgan dense counts"]
    )    
    @pytest.mark.parametrize(
        "features",
        [
            np.array([["CC1=CC=C(C=C1)O"]]),
            np.array([["C1=CC=C(C=C1)F"]]),
        ], ids=["molecule 1", "molecule 2"]
    )
    def test_transformer_reproduces_morgan_fps(self, STFP_transformer, fp_generator, options_dict, features):
        my_transformer = STFP_transformer.set_params(**options_dict).fit(features)
        transformer_fp = my_transformer.transform(features)[0]
        if not options_dict["dense"]:
            transformer_fp = transformer_fp.toarray()
        if options_dict["counts"]:
            rdkit_fp = fp_generator(fpSize=options_dict["fp_size"], radius=options_dict["radius"]).GetCountFingerprintAsNumPy(Chem.MolFromSmiles(features[0][0]))
        else:
            rdkit_fp = fp_generator(fpSize=options_dict["fp_size"], radius=options_dict["radius"]).GetFingerprintAsNumPy(Chem.MolFromSmiles(features[0][0]))
        npt.assert_array_equal(transformer_fp, rdkit_fp)
        
    @pytest.mark.parametrize(
            "options_dict, fp_generator",
            [
            ({"fp_type": "rdkit", "dense": False, "fp_size":8, "counts": False, "radius": 2}, 
             rdFingerprintGenerator.GetRDKitFPGenerator),
            ({"fp_type": "rdkit", "dense": True, "fp_size":64, "counts": False, "radius": 1}, 
             rdFingerprintGenerator.GetRDKitFPGenerator),
             ({"fp_type": "rdkit", "dense": False, "fp_size":8, "counts": True, "radius": 5}, 
             rdFingerprintGenerator.GetRDKitFPGenerator),
            ({"fp_type": "rdkit", "dense": True, "fp_size":64, "counts": True, "radius": 6}, 
             rdFingerprintGenerator.GetRDKitFPGenerator),
            ], ids=["rdkit sparse bits", "rdkit dense bits", "rdkit sparse counts", "rdkit dense counts"]
    )    
    @pytest.mark.parametrize(
        "features",
        [
            np.array([["CC1=CC=C(C=C1)O"]]),
            np.array([["C1=CC=C(C=C1)F"]]),
        ], ids=["molecule 1", "molecule 2"]
    )
    def test_transformer_reproduces_rdkit_fps(self, STFP_transformer, fp_generator, options_dict, features):
        my_transformer = STFP_transformer.set_params(**options_dict).fit(features)
        transformer_fp = my_transformer.transform(features)[0]
        if not options_dict["dense"]:
            transformer_fp = transformer_fp.toarray()
        if options_dict["counts"]:
            rdkit_fp = fp_generator(fpSize=options_dict["fp_size"], maxPath=options_dict["radius"]).GetCountFingerprintAsNumPy(Chem.MolFromSmiles(features[0][0]))
        else:
            rdkit_fp = fp_generator(fpSize=options_dict["fp_size"], maxPath=options_dict["radius"]).GetFingerprintAsNumPy(Chem.MolFromSmiles(features[0][0]))
        npt.assert_array_equal(transformer_fp, rdkit_fp)
    



