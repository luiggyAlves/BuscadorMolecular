"""
popular_banco.py
Script executável que orquestra o pipeline completo de população do
banco vetorial ChromaDB com as moléculas da base NuBBED.

Fluxo:
    1. Carregar moléculas NuBBED do arquivo SDF.
    2. Validar e canonicalizar SMILES com RDKit.
    3. Verificar quais IDs já foram inseridos (idempotência/retomabilidade).
    4. Vetorizar em lotes com MolFormer-XL.
    5. Inserir embeddings no ChromaDB.

Uso:
    python popular_banco.py \\
        --caminho_nubbed nubbedb-05-2026.sdf \\
        --caminho_banco ./banco_vetorial \\
        --tamanho_lote 64
"""

import argparse
import logging
import sys
from typing import Optional

from tqdm import tqdm

from carregador_nubbed import carregar_moleculas_nubbed
from gerenciador_banco_vetorial import GerenciadorBancoVetorial
from preparador_smiles import canonicalizar_smiles, validar_e_canonicalizar_lista
from vetorizador_molformer import VetorizadorMolFormer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

NOME_COLECAO_NUBBED = "moleculas_nubbed"


def construir_parser_argumentos() -> argparse.ArgumentParser:
    """Constrói e retorna o parser de argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Popula o banco vetorial ChromaDB com embeddings da base NuBBED.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--caminho_nubbed",
        type=str,
        required=True,
        help="Caminho para o arquivo SDF da base NuBBED.",
    )
    parser.add_argument(
        "--caminho_banco",
        type=str,
        default="./banco_vetorial",
        help="Diretório onde o ChromaDB será persistido.",
    )
    parser.add_argument(
        "--tamanho_lote",
        type=int,
        default=64,
        help="Número de moléculas por lote de vetorização.",
    )
    parser.add_argument(
        "--dispositivo",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Dispositivo para inferência (padrão: detecção automática).",
    )
    return parser


def popular_banco_com_nubbed(
    caminho_arquivo_nubbed: str,
    caminho_banco_vetorial: str,
    tamanho_lote_processamento: int = 64,
    dispositivo_computacao: Optional[str] = None,
) -> None:
    """
    Orquestra o pipeline completo de carregamento, validação, vetorização
    e inserção das moléculas NuBBED no banco vetorial ChromaDB.

    Parâmetros:
        caminho_arquivo_nubbed:    Arquivo SDF da NuBBED.
        caminho_banco_vetorial:    Diretório de persistência do ChromaDB.
        tamanho_lote_processamento: Moléculas por lote de vetorização.
        dispositivo_computacao:    'cuda', 'cpu' ou None (automático).
    """
    # ── Etapa 1: Carregar base NuBBED ──────────────────────────────────────
    logger.info("=== ETAPA 1/5: Carregando base NuBBED ===")
    lista_moleculas_brutas = carregar_moleculas_nubbed(caminho_arquivo_nubbed)
    total_carregado = len(lista_moleculas_brutas)
    logger.info("Total carregado: %d moléculas.", total_carregado)

    # ── Etapa 2: Validar e canonicalizar SMILES ────────────────────────────
    logger.info("=== ETAPA 2/5: Validando e canonicalizando SMILES ===")
    lista_smiles_brutos = [mol["smiles_bruto"] for mol in lista_moleculas_brutas]
    lista_ids_originais = [mol["id_nubbed"] for mol in lista_moleculas_brutas]

    _, lista_smiles_rejeitados = validar_e_canonicalizar_lista(lista_smiles_brutos)

    # Reconstrói lista pareada (smiles canônico ↔ id_nubbed) apenas dos válidos
    lista_pares_validos: list[tuple[str, str]] = []
    for smiles_original, id_original in zip(lista_smiles_brutos, lista_ids_originais):
        smiles_canonico = canonicalizar_smiles(smiles_original)
        if smiles_canonico is not None:
            lista_pares_validos.append((smiles_canonico, id_original))

    total_validos = len(lista_pares_validos)
    total_rejeitados = len(lista_smiles_rejeitados)
    logger.info(
        "SMILES válidos: %d | Rejeitados: %d", total_validos, total_rejeitados
    )

    # ── Etapa 3: Verificar IDs já inseridos (idempotência) ─────────────────
    logger.info("=== ETAPA 3/5: Verificando entradas já inseridas ===")
    gerenciador_banco = GerenciadorBancoVetorial(
        caminho_banco_vetorial,
        nome_colecao=NOME_COLECAO_NUBBED,
    )
    conjunto_ids_ja_inseridos = gerenciador_banco.obter_ids_existentes()
    logger.info("IDs já no banco: %d", len(conjunto_ids_ja_inseridos))

    lista_pares_pendentes = [
        (smiles, id_nubbed)
        for smiles, id_nubbed in lista_pares_validos
        if id_nubbed not in conjunto_ids_ja_inseridos
    ]
    total_pendentes = len(lista_pares_pendentes)
    total_ignorados = total_validos - total_pendentes
    logger.info(
        "Pendentes para inserção: %d | Já inseridos (ignorados): %d",
        total_pendentes,
        total_ignorados,
    )

    if total_pendentes == 0:
        logger.info("Banco já está completo. Nenhuma inserção necessária.")
        return

    # ── Etapa 4: Carregar modelo e vetorizar em lotes ──────────────────────
    logger.info("=== ETAPA 4/5: Vetorizando moléculas com MolFormer-XL ===")
    vetorizador_molecular = VetorizadorMolFormer(dispositivo=dispositivo_computacao)

    lista_smiles_pendentes = [par[0] for par in lista_pares_pendentes]
    lista_ids_pendentes = [par[1] for par in lista_pares_pendentes]

    # ── Etapa 5: Inserir no ChromaDB com barra de progresso ───────────────
    logger.info("=== ETAPA 5/5: Inserindo embeddings no ChromaDB ===")

    total_inserido_sucesso = 0
    total_falha_vetorizacao = 0

    barra_progresso = tqdm(
        total=total_pendentes,
        desc="Populando banco NuBBED",
        unit="mol",
        dynamic_ncols=True,
    )

    for indice_lote in range(0, total_pendentes, tamanho_lote_processamento):
        smiles_lote_atual = lista_smiles_pendentes[
            indice_lote: indice_lote + tamanho_lote_processamento
        ]
        ids_lote_atual = lista_ids_pendentes[
            indice_lote: indice_lote + tamanho_lote_processamento
        ]

        embeddings_lote = vetorizador_molecular.vetorizar_lote(
            smiles_lote_atual, tamanho_lote=tamanho_lote_processamento
        )

        smiles_inserir: list[str] = []
        ids_inserir: list[str] = []
        embeddings_inserir: list[list[float]] = []

        for smiles_mol, id_mol, embedding_mol in zip(
            smiles_lote_atual, ids_lote_atual, embeddings_lote
        ):
            if embedding_mol is not None:
                smiles_inserir.append(smiles_mol)
                ids_inserir.append(id_mol)
                embeddings_inserir.append(embedding_mol)
            else:
                total_falha_vetorizacao += 1
                logger.warning("Embedding falhou para ID '%s', molécula ignorada.", id_mol)

        if ids_inserir:
            gerenciador_banco.inserir_lote(
                lista_ids_insercao=ids_inserir,
                lista_embeddings_moleculares=embeddings_inserir,
                lista_smiles_canonicos=smiles_inserir,
                lista_ids_molecula=ids_inserir,
            )
            total_inserido_sucesso += len(ids_inserir)

        barra_progresso.update(len(smiles_lote_atual))

    barra_progresso.close()

    # ── Resumo final ───────────────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("RESUMO DA POPULAÇÃO DO BANCO VETORIAL (NuBBED)")
    logger.info("  Moléculas carregadas do SDF:      %d", total_carregado)
    logger.info("  SMILES inválidos descartados:     %d", total_rejeitados)
    logger.info("  Já presentes no banco (ignorados):%d", total_ignorados)
    logger.info("  Falhas na vetorização:            %d", total_falha_vetorizacao)
    logger.info("  Inseridas com sucesso:            %d", total_inserido_sucesso)
    logger.info(
        "  Total no banco agora:             %d",
        gerenciador_banco.total_moleculas_indexadas(),
    )
    logger.info("=" * 55)


def main() -> None:
    """Ponto de entrada principal do script de população."""
    parser = construir_parser_argumentos()
    argumentos = parser.parse_args()

    try:
        popular_banco_com_nubbed(
            caminho_arquivo_nubbed=argumentos.caminho_nubbed,
            caminho_banco_vetorial=argumentos.caminho_banco,
            tamanho_lote_processamento=argumentos.tamanho_lote,
            dispositivo_computacao=argumentos.dispositivo,
        )
    except FileNotFoundError as erro_arquivo:
        logger.error("Arquivo não encontrado: %s", erro_arquivo)
        sys.exit(1)
    except ValueError as erro_valor:
        logger.error("Erro de configuração: %s", erro_valor)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrompido pelo usuário. Progresso salvo no banco ChromaDB.")
        sys.exit(0)


if __name__ == "__main__":
    main()
