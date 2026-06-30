import pytest
from contextlib import nullcontext
import numpy as np
import numpy.testing as npt
from scipy import sparse
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from dl_hplc_smrt.data_transformers import SmilesToMolTransformer as STM
from dl_hplc_smrt.data_transformers import MolToFingerPrintTransformer as MTFP


x_smiles_2col = np.array([["CC1=CC=C(C=C1)O","CCC(C)CC(C)CC"],
            ["CC1=CC(=CO1)C", "CCC(C)(C)CC"],
            ["C1=CC=C(C=C1)F","CC1CCC(CC1)C"],])

x_smiles_1col = np.array([["CC1=CC=C(C=C1)O"],
            ["CC1=CC(=CO1)C"],
            ["C1=CC=C(C=C1)F"],])

x_smiles_1col_faulty = np.array([["CC1=CC=C(C=C1)O"],
                                ["ThisIsNotaProperSMILES"],
                                ["C1=CC=C(C=C1)F"],
                                ["NeitherItIsThisOne"]])

x_mol_2col = np.array([[Chem.MolFromSmiles("CC1=CC=C(C=C1)O"),Chem.MolFromSmiles("CCC(C)CC(C)CC")],
            [Chem.MolFromSmiles("CC1=CC(=CO1)C"), Chem.MolFromSmiles("CCC(C)(C)CC")],
            [Chem.MolFromSmiles("C1=CC=C(C=C1)F"),Chem.MolFromSmiles("CC1CCC(CC1)C")],])

x_mol_1col = np.array([[Chem.MolFromSmiles("CC1=CC=C(C=C1)O")],
            [Chem.MolFromSmiles("CC1=CC(=CO1)C")],
            [Chem.MolFromSmiles("C1=CC=C(C=C1)F")],])

x_mol_1col_faulty = np.array([[Chem.MolFromSmiles("CC1=CC=C(C=C1)O")],
                                [Chem.MolFromSmiles("")],
                                [Chem.MolFromSmiles("C1=CC=C(C=C1)F")],
                                [Chem.MolFromSmiles("")]])

@pytest.fixture(params=[x_smiles_1col, x_smiles_2col, x_smiles_1col_faulty])
def smiles_matrix(request):
    return request.param

@pytest.fixture(params=[x_mol_1col, x_mol_2col, x_mol_1col_faulty])
def mol_matrix(request):
    return request.param

@pytest.fixture()
def STM_transformer():
    return STM()

@pytest.fixture()
def MTFP_transformer():
    return MTFP()

@pytest.fixture()
def maccs_fp_generator():
    return MACCSkeys.GenMACCSKeys
    

class TestSmilesToMolTransformer:
    @pytest.mark.parametrize(
    "transformer, expected_exception",
    [
        (STM(), pytest.raises(ValueError)),     # Not fitted (exception expected)
        (STM().fit(x_smiles_2col), nullcontext()),                 # Fitted (no exception expected)        
    ], ids=["not_fitted_triggers_exception","fitted_so_no_exception"]
)
    def test_fit_before_transform(self, transformer, expected_exception):
        """This method tests that an exception is riased if using the transform
        method without fitting the method first. And that such exception is not
        raised if the transformer has been fitted"""
        with expected_exception:
            transformer.transform(x_smiles_2col)
    
    def test_output_type(self, STM_transformer, smiles_matrix):
        """ This method tests that the output type (sparse/dense) is consistent
        with the requested output using the option `dense`"""
        produced_output = STM_transformer.fit_transform(smiles_matrix)
        assert isinstance(produced_output, np.ndarray)
    
    def test_output_dims(self, STM_transformer, smiles_matrix):
        """ This method tests that the transformed feature matrices have the
        expected size, when Morgan or RDKit fps are requested"""
        produced_output = STM_transformer.fit_transform(smiles_matrix)
        expected_dimensions = (smiles_matrix.shape[0], smiles_matrix.shape[1])
        assert produced_output.shape == expected_dimensions

    @pytest.mark.parametrize(
            "smiles_string",
            [
                "CC1=CC=C(C=C1)O",
                "ThisIsNotaProperSMILES",
                "C1=CC=C(C=C1)F",
                "NeitherItIsThisOne"
        ]
    )
    def test_reproduces_rdkit(self, STM_transformer, smiles_string):
        """This test checks that we get mol objects when we pass SMILES strings
        """
        produced_output = STM_transformer.fit_transform(np.array([[smiles_string]]))
        assert isinstance(produced_output[0][0], Chem.rdchem.Mol)

class TestMolToFingerPrintTransformer:
    @pytest.mark.parametrize(
    "transformer, expected_exception",
    [
        (MTFP(), pytest.raises(ValueError)),     # Not fitted (exception expected)
        (MTFP().fit(x_mol_2col), nullcontext()),                 # Fitted (no exception expected)        
    ], ids=["not_fitted_triggers_exception","fitted_so_no_exception"]
)
    def test_fit_before_transform(self, transformer, expected_exception):
        """This method tests that an exception is riased if using the transform
        method without fitting the method first. And that such exception is not
        raised if the transformer has been fitted"""
        with expected_exception:
            transformer.transform(x_mol_2col)

    def test_maccs_counts_not_available(self):
        """This method tests that an exception is riased when MACCS fps are
        requested together with counts instead of bits"""
        with pytest.raises(ValueError):
            MTFP(fp_type="maccs", counts=True).fit(x_mol_2col)
    
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
    def test_output_type(self, MTFP_transformer, options_dict, output_type, mol_matrix):
        """ This method tests that the output type (sparse/dense) is consistent
        with the requested output using the option `dense`"""
        MTFP_transformer.set_params(**options_dict)
        produced_output = MTFP_transformer.fit_transform(mol_matrix)
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
    def test_morgan_rdkit_output_dims(self, MTFP_transformer, options_dict, mol_matrix):
        """ This method tests that the transformed feature matrices have the
        expected size, when Morgan or RDKit fps are requested"""
        MTFP_transformer.set_params(**options_dict)
        produced_output = MTFP_transformer.fit_transform(mol_matrix)
        expected_dimensions = (mol_matrix.shape[0], mol_matrix.shape[1]*options_dict['fp_size'])
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
    def test_maccs_output_dims(self, MTFP_transformer, options_dict, mol_matrix):
        """ This method tests that the transformed feature matrices have the
        expected size, when MACSS fps are requested. Noted that for MACCS fps
        the parameter `fp_size` is silently ignored and the produced fps are
        always of size 167"""
        MTFP_transformer.set_params(**options_dict)
        produced_output = MTFP_transformer.fit_transform(mol_matrix)
        expected_dimensions = (mol_matrix.shape[0], mol_matrix.shape[1]*167)
        #assert npt.assert_allclose(produced_output.shape, expected_dimensions)
        assert produced_output.shape == expected_dimensions
    
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
            np.array([[Chem.MolFromSmiles("CC1=CC=C(C=C1)O")]]),
            np.array([[Chem.MolFromSmiles("C1=CC=C(C=C1)F")]]),
        ], ids=["molecule 1", "molecule 2"]
    )
    def test_reproduces_rdkit_for_maccs_fps(self, MTFP_transformer, maccs_fp_generator, options_dict, features):
        my_transformer = MTFP_transformer.set_params(**options_dict).fit(features)
        transformer_fp = my_transformer.transform(features)[0]
        if not options_dict["dense"]:
            transformer_fp = transformer_fp.toarray()
        rdkit_fp = maccs_fp_generator(features[0][0]).ToList()
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
            np.array([[Chem.MolFromSmiles("CC1=CC=C(C=C1)O")]]),
            np.array([[Chem.MolFromSmiles("C1=CC=C(C=C1)F")]]),
        ], ids=["molecule 1", "molecule 2"]
    )
    def test_reproduces_morgan_fps(self, MTFP_transformer, fp_generator, options_dict, features):
        my_transformer = MTFP_transformer.set_params(**options_dict).fit(features)
        transformer_fp = my_transformer.transform(features)[0]
        if not options_dict["dense"]:
            transformer_fp = transformer_fp.toarray()
        if options_dict["counts"]:
            rdkit_fp = fp_generator(fpSize=options_dict["fp_size"], radius=options_dict["radius"]).GetCountFingerprintAsNumPy(features[0][0])
        else:
            rdkit_fp = fp_generator(fpSize=options_dict["fp_size"], radius=options_dict["radius"]).GetFingerprintAsNumPy(features[0][0])
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
            np.array([[Chem.MolFromSmiles("CC1=CC=C(C=C1)O")]]),
            np.array([[Chem.MolFromSmiles("C1=CC=C(C=C1)F")]]),
        ], ids=["molecule 1", "molecule 2"]
    )
    def test_transformer_reproduces_rdkit_fps(self, MTFP_transformer, fp_generator, options_dict, features):
        my_transformer = MTFP_transformer.set_params(**options_dict).fit(features)
        transformer_fp = my_transformer.transform(features)[0]
        if not options_dict["dense"]:
            transformer_fp = transformer_fp.toarray()
        if options_dict["counts"]:
            rdkit_fp = fp_generator(fpSize=options_dict["fp_size"], maxPath=options_dict["radius"]).GetCountFingerprintAsNumPy(features[0][0])
        else:
            rdkit_fp = fp_generator(fpSize=options_dict["fp_size"], maxPath=options_dict["radius"]).GetFingerprintAsNumPy(features[0][0])
        npt.assert_array_equal(transformer_fp, rdkit_fp)
    



