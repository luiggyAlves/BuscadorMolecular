"""
carregador_coconut.py
Lê a base COCONUT (Collection of Open Natural Products) a partir de
arquivos CSV, SDF ou SMI e extrai SMILES e identificadores das moléculas.
"""

import logging
import os
from typing import Optional

import pandas as pd
from rdkit import Chem

logger = logging.getLogger(__name__)

# Variantes comuns de nomes de colunas SMILES em datasets moleculares
NOMES_COLUNA_SMILES_POSSIVEIS = [
    "smiles", "SMILES", "Smiles", "canonical_smiles",
    "canonicalSmiles", "smi", "SMI",
]

# Variantes comuns de nomes de colunas de identificador COCONUT
NOMES_COLUNA_ID_POSSIVEIS = [
    "coconut_id", "coconutID", "id", "ID", "identifier",
    "name", "Name", "compound_id", "compoundID",
]


def _detectar_coluna_smiles(colunas_disponiveis: list[str]) -> Optional[str]:
    """Detecta o nome da coluna SMILES no DataFrame."""
    for nome_candidato in NOMES_COLUNA_SMILES_POSSIVEIS:
        if nome_candidato in colunas_disponiveis:
            return nome_candidato
    return None


def _detectar_coluna_id(colunas_disponiveis: list[str]) -> Optional[str]:
    """Detecta o nome da coluna de identificador no DataFrame."""
    for nome_candidato in NOMES_COLUNA_ID_POSSIVEIS:
        if nome_candidato in colunas_disponiveis:
            return nome_candidato
    return None


def _carregar_de_csv(caminho_arquivo_csv: str) -> list[dict]:
    """
    Carrega moléculas a partir de arquivo CSV com detecção automática
    de colunas de SMILES e identificador.

    Parâmetros:
        caminho_arquivo_csv: Caminho para o arquivo .csv da COCONUT.

    Retorna:
        Lista de dicts {"smiles_bruto": ..., "id_coconut": ...}.
    """
    dataframe_coconut = pd.read_csv(caminho_arquivo_csv, low_memory=False)
    colunas = dataframe_coconut.columns.tolist()

    nome_coluna_smiles = _detectar_coluna_smiles(colunas)
    if nome_coluna_smiles is None:
        raise ValueError(
            f"Nenhuma coluna de SMILES reconhecida no CSV. "
            f"Colunas encontradas: {colunas}"
        )

    nome_coluna_id = _detectar_coluna_id(colunas)
    logger.info(
        "CSV: coluna SMILES='%s', coluna ID='%s'.",
        nome_coluna_smiles,
        nome_coluna_id,
    )

    lista_moleculas: list[dict] = []
    for indice_linha, linha in dataframe_coconut.iterrows():
        smiles_bruto = str(linha[nome_coluna_smiles]).strip()
        if smiles_bruto in ("nan", "", "None"):
            continue

        id_coconut = (
            str(linha[nome_coluna_id]).strip()
            if nome_coluna_id
            else f"MOL_{indice_linha}"
        )

        lista_moleculas.append(
            {"smiles_bruto": smiles_bruto, "id_coconut": id_coconut}
        )

    return lista_moleculas


def _carregar_de_sdf(caminho_arquivo_sdf: str) -> list[dict]:
    """
    Carrega moléculas a partir de arquivo SDF usando RDKit SDMolSupplier.
    O identificador é extraído da propriedade '_Name' ou índice sequencial.

    Parâmetros:
        caminho_arquivo_sdf: Caminho para o arquivo .sdf da COCONUT.

    Retorna:
        Lista de dicts {"smiles_bruto": ..., "id_coconut": ...}.
    """
    fornecedor_sdf = Chem.SDMolSupplier(caminho_arquivo_sdf, sanitize=True)
    lista_moleculas: list[dict] = []

    for indice_sdf, molecula_sdf in enumerate(fornecedor_sdf):
        if molecula_sdf is None:
            logger.warning("Molécula SDF índice %d inválida, ignorada.", indice_sdf)
            continue

        smiles_bruto = Chem.MolToSmiles(molecula_sdf, canonical=False)
        if not smiles_bruto:
            continue

        # Tenta obter ID das propriedades do SDF
        if molecula_sdf.HasProp("coconut_id"):
            id_coconut = molecula_sdf.GetProp("coconut_id")
        elif molecula_sdf.HasProp("_Name") and molecula_sdf.GetProp("_Name").strip():
            id_coconut = molecula_sdf.GetProp("_Name").strip()
        else:
            id_coconut = f"SDF_{indice_sdf}"

        lista_moleculas.append(
            {"smiles_bruto": smiles_bruto, "id_coconut": id_coconut}
        )

    return lista_moleculas


def _carregar_de_smi(caminho_arquivo_smi: str) -> list[dict]:
    """
    Carrega moléculas a partir de arquivo .smi (um SMILES por linha,
    opcionalmente com identificador separado por espaço/tab).

    Formato esperado por linha:
        SMILES [separador] ID_OPCIONAL

    Parâmetros:
        caminho_arquivo_smi: Caminho para o arquivo .smi da COCONUT.

    Retorna:
        Lista de dicts {"smiles_bruto": ..., "id_coconut": ...}.
    """
    lista_moleculas: list[dict] = []

    with open(caminho_arquivo_smi, "r", encoding="utf-8") as arquivo_smi:
        for indice_linha, linha_bruta in enumerate(arquivo_smi):
            linha = linha_bruta.strip()
            if not linha or linha.startswith("#"):
                continue

            partes_linha = linha.split()
            smiles_bruto = partes_linha[0]
            id_coconut = partes_linha[1] if len(partes_linha) > 1 else f"SMI_{indice_linha}"

            lista_moleculas.append(
                {"smiles_bruto": smiles_bruto, "id_coconut": id_coconut}
            )

    return lista_moleculas


def carregar_moleculas_coconut(caminho_arquivo: str) -> list[dict]:
    """
    Carrega todas as moléculas da base COCONUT a partir de arquivo CSV, SDF ou SMI.
    Detecta o formato automaticamente pela extensão do arquivo.

    Parâmetros:
        caminho_arquivo: Caminho para o arquivo de dados da COCONUT.
                         Extensões suportadas: .csv, .sdf, .smi, .smiles

    Retorna:
        Lista de dicionários com campos:
            - smiles_bruto: String SMILES original (ainda não canonicalizada).
            - id_coconut:   Identificador único da molécula na COCONUT.

    Lança:
        FileNotFoundError: Se o arquivo não existir.
        ValueError: Se o formato de arquivo não for suportado.
    """
    if not os.path.isfile(caminho_arquivo):
        raise FileNotFoundError(f"Arquivo COCONUT não encontrado: '{caminho_arquivo}'")

    extensao_arquivo = os.path.splitext(caminho_arquivo)[1].lower()
    logger.info("Carregando COCONUT de '%s' (formato: %s).", caminho_arquivo, extensao_arquivo)

    if extensao_arquivo == ".csv":
        lista_moleculas = _carregar_de_csv(caminho_arquivo)
    elif extensao_arquivo == ".sdf":
        lista_moleculas = _carregar_de_sdf(caminho_arquivo)
    elif extensao_arquivo in (".smi", ".smiles"):
        lista_moleculas = _carregar_de_smi(caminho_arquivo)
    else:
        raise ValueError(
            f"Formato de arquivo não suportado: '{extensao_arquivo}'. "
            "Use .csv, .sdf ou .smi/.smiles"
        )

    logger.info(
        "COCONUT carregada: %d moléculas encontradas em '%s'.",
        len(lista_moleculas),
        caminho_arquivo,
    )
    return lista_moleculas
