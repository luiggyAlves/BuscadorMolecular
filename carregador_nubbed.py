"""
carregador_nubbed.py
Lê a base NuBBED (Nuclei of Bioassays, Biosynthesis and Biodiversity)
a partir de arquivo SDF e extrai SMILES e identificadores das moléculas.

Campos lidos do SDF:
    - <identifier>      → ID único da molécula (ex: CNP0173930.1)
    - <canonical_smiles> → SMILES canônico pré-computado
    Fallback: SMILES gerado diretamente do molblock via RDKit.
"""

import logging
import os

from rdkit import Chem

logger = logging.getLogger(__name__)


def carregar_moleculas_nubbed(caminho_arquivo_sdf: str) -> list[dict]:
    """
    Carrega todas as moléculas da base NuBBED a partir de arquivo SDF.

    Extrai o identificador da propriedade <identifier> e o SMILES da
    propriedade <canonical_smiles>. Quando <canonical_smiles> estiver
    ausente, o SMILES é gerado diretamente do molblock pelo RDKit.
    Moléculas que o RDKit não consegue parsear são ignoradas com log.

    Parâmetros:
        caminho_arquivo_sdf: Caminho para o arquivo .sdf da NuBBED.

    Retorna:
        Lista de dicionários com campos:
            - smiles_bruto: SMILES da molécula (ainda não revalidado).
            - id_nubbed:    Identificador único da molécula na NuBBED.

    Lança:
        FileNotFoundError: Se o arquivo não existir.
        ValueError: Se a extensão não for .sdf.
    """
    if not os.path.isfile(caminho_arquivo_sdf):
        raise FileNotFoundError(
            f"Arquivo NuBBED não encontrado: '{caminho_arquivo_sdf}'"
        )

    extensao = os.path.splitext(caminho_arquivo_sdf)[1].lower()
    if extensao != ".sdf":
        raise ValueError(
            f"Formato inválido: '{extensao}'. O carregador NuBBED aceita apenas .sdf"
        )

    logger.info("Carregando NuBBED de '%s'...", caminho_arquivo_sdf)

    fornecedor_sdf = Chem.SDMolSupplier(caminho_arquivo_sdf, sanitize=True)

    lista_moleculas: list[dict] = []
    total_ignoradas = 0

    for indice_sdf, molecula_sdf in enumerate(fornecedor_sdf):
        if molecula_sdf is None:
            logger.debug("Entrada SDF índice %d inválida (RDKit retornou None).", indice_sdf)
            total_ignoradas += 1
            continue

        # ── Extrair identificador ──────────────────────────────────────────
        if molecula_sdf.HasProp("identifier"):
            id_nubbed = molecula_sdf.GetProp("identifier").strip()
        elif molecula_sdf.HasProp("_Name") and molecula_sdf.GetProp("_Name").strip():
            id_nubbed = molecula_sdf.GetProp("_Name").strip()
        else:
            id_nubbed = f"NUBBED_{indice_sdf}"

        # ── Extrair SMILES ────────────────────────────────────────────────
        # Prefere o SMILES pré-computado do campo <canonical_smiles> do SDF
        if molecula_sdf.HasProp("canonical_smiles"):
            smiles_bruto = molecula_sdf.GetProp("canonical_smiles").strip()
        else:
            smiles_bruto = Chem.MolToSmiles(molecula_sdf, canonical=False)

        if not smiles_bruto:
            logger.warning("SMILES vazio para ID '%s', molécula ignorada.", id_nubbed)
            total_ignoradas += 1
            continue

        lista_moleculas.append(
            {"smiles_bruto": smiles_bruto, "id_nubbed": id_nubbed}
        )

    total_carregado = len(lista_moleculas)
    logger.info(
        "NuBBED carregada: %d moléculas válidas, %d ignoradas.",
        total_carregado,
        total_ignoradas,
    )
    return lista_moleculas
