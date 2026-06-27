from rdkit import Chem

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