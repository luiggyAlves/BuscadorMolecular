"""
preparador_smiles.py
Funções de validação e canonicalização de strings SMILES usando RDKit.
SMILES inválidos são descartados e registrados via logging.
"""

import logging
from typing import Optional

from rdkit import Chem
from rdkit import RDLogger

# Suprime warnings internos do RDKit no console
RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)


def canonicalizar_smiles(smiles_entrada: str) -> Optional[str]:
    """
    Valida e converte um SMILES para sua forma canônica via RDKit.

    Parâmetros:
        smiles_entrada: String SMILES arbitrária (pode ser não canônica ou inválida).

    Retorna:
        String SMILES canônica se válida, None caso contrário.
    """
    if not smiles_entrada or not smiles_entrada.strip():
        return None

    molecula_rdkit = Chem.MolFromSmiles(smiles_entrada.strip())
    if molecula_rdkit is None:
        return None

    smiles_canonico = Chem.MolToSmiles(molecula_rdkit, canonical=True)
    return smiles_canonico if smiles_canonico else None


def validar_e_canonicalizar_lista(
    lista_smiles_brutos: list[str],
) -> tuple[list[str], list[str]]:
    """
    Processa uma lista de SMILES brutos: valida, canonicaliza e separa
    os válidos dos inválidos.

    Parâmetros:
        lista_smiles_brutos: Lista de strings SMILES vindas da fonte de dados.

    Retorna:
        Tupla (lista_smiles_validos, lista_smiles_rejeitados).
        - lista_smiles_validos:   SMILES canônicos prontos para vetorização.
        - lista_smiles_rejeitados: SMILES originais que falharam na validação.
    """
    lista_smiles_validos: list[str] = []
    lista_smiles_rejeitados: list[str] = []

    for smiles_bruto in lista_smiles_brutos:
        smiles_canonico = canonicalizar_smiles(smiles_bruto)
        if smiles_canonico is not None:
            lista_smiles_validos.append(smiles_canonico)
        else:
            lista_smiles_rejeitados.append(smiles_bruto)
            logger.warning("SMILES inválido descartado: '%s'", smiles_bruto)

    total = len(lista_smiles_brutos)
    total_validos = len(lista_smiles_validos)
    total_rejeitados = len(lista_smiles_rejeitados)
    logger.info(
        "Validação concluída: %d/%d válidos, %d rejeitados.",
        total_validos,
        total,
        total_rejeitados,
    )

    return lista_smiles_validos, lista_smiles_rejeitados


def validar_smiles_unico(smiles_entrada: str) -> str:
    """
    Valida e canonicaliza um SMILES único. Lança ValueError se inválido.
    Conveniente para uso em CLIs onde falha deve ser explícita.

    Parâmetros:
        smiles_entrada: String SMILES fornecida pelo usuário.

    Retorna:
        SMILES canônico.

    Lança:
        ValueError: Se o SMILES não for reconhecido pelo RDKit.
    """
    smiles_canonico = canonicalizar_smiles(smiles_entrada)
    if smiles_canonico is None:
        raise ValueError(
            f"SMILES inválido ou não reconhecido pelo RDKit: '{smiles_entrada}'"
        )
    return smiles_canonico
