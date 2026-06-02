"""
vetorizador_fingerprint.py
Geração de vetores moleculares usando fingerprints Morgan (ECFP4) do RDKit.
Vetores binários L2-normalizados para busca por cosseno no ChromaDB.
"""

import logging
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

logger = logging.getLogger(__name__)

RAIO_MORGAN = 2
NBITS_MORGAN = 2048


class VetorizadorFingerprint:
    """
    Fingerprints Morgan ECFP4 normalizados para busca vetorial por cosseno.
    Não requer modelo neural — geração puramente via RDKit.
    """

    def __init__(self, raio: int = RAIO_MORGAN, nbits: int = NBITS_MORGAN) -> None:
        self.raio = raio
        self.nbits = nbits

    def vetorizar_molecula(self, smiles_canonico: str) -> Optional[list[float]]:
        try:
            mol = Chem.MolFromSmiles(smiles_canonico)
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.raio, nBits=self.nbits)
            arr = np.zeros(self.nbits, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            return arr.tolist()
        except Exception as e:
            logger.error("Erro ao gerar fingerprint para '%s': %s", smiles_canonico, e)
            return None

    def vetorizar_lote(
        self,
        lista_smiles: list[str],
        tamanho_lote: int = 512,
    ) -> list[Optional[list[float]]]:
        return [self.vetorizar_molecula(s) for s in lista_smiles]
