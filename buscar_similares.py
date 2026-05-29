"""
buscar_similares.py
CLI executável para busca por similaridade molecular.
Recebe SMILES query e K, retorna as K moléculas mais similares do banco.

Uso:
    python buscar_similares.py \\
        --smiles_consulta "CC(=O)Oc1ccccc1C(=O)O" \\
        --quantidade_resultados 10 \\
        --caminho_banco ./banco_vetorial
"""

import argparse
import logging
import sys

from gerenciador_banco_vetorial import GerenciadorBancoVetorial
from preparador_smiles import validar_smiles_unico
from vetorizador_molformer import VetorizadorMolFormer

NOME_COLECAO_NUBBED = "moleculas_nubbed"

# Configura logging silencioso para CLI (apenas erros críticos)
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def construir_parser_busca() -> argparse.ArgumentParser:
    """Constrói e retorna o parser de argumentos para a CLI de busca."""
    parser = argparse.ArgumentParser(
        description="Busca moléculas similares no banco vetorial usando MolFormer-XL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--smiles_consulta",
        type=str,
        required=True,
        help="SMILES da molécula query para busca de similaridade.",
    )
    parser.add_argument(
        "--quantidade_resultados",
        type=int,
        default=10,
        help="Número K de moléculas similares a retornar.",
    )
    parser.add_argument(
        "--caminho_banco",
        type=str,
        default="./banco_vetorial",
        help="Diretório do ChromaDB persistente.",
    )
    parser.add_argument(
        "--dispositivo",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Dispositivo para inferência (padrão: detecção automática).",
    )
    return parser


def formatar_tabela_resultados(
    lista_moleculas_similares: list[dict],
    smiles_consulta_canonico: str,
    quantidade_solicitada: int,
) -> str:
    """
    Formata os resultados de busca em tabela legível para exibição no terminal.

    Parâmetros:
        lista_moleculas_similares: Lista de dicts retornada pelo GerenciadorBancoVetorial.
        smiles_consulta_canonico:  SMILES canônico da molécula query.
        quantidade_solicitada:     K solicitado pelo usuário.

    Retorna:
        String formatada com a tabela de resultados.
    """
    linhas_saida: list[str] = []

    linhas_saida.append("")
    linhas_saida.append("═" * 80)
    linhas_saida.append("  BUSCA POR SIMILARIDADE MOLECULAR — MolFormer-XL + ChromaDB")
    linhas_saida.append("═" * 80)
    linhas_saida.append(f"  Molécula query (SMILES): {smiles_consulta_canonico}")
    linhas_saida.append(f"  Resultados solicitados:  {quantidade_solicitada}")
    linhas_saida.append(f"  Resultados encontrados:  {len(lista_moleculas_similares)}")
    linhas_saida.append("─" * 80)

    if not lista_moleculas_similares:
        linhas_saida.append("  Nenhuma molécula encontrada no banco vetorial.")
        linhas_saida.append("═" * 80)
        return "\n".join(linhas_saida)

    # Cabeçalho da tabela
    cabecalho_tabela = (
        f"  {'Pos':>3}  {'ID NuBBED':<20}  {'Similaridade':>12}  {'SMILES'}"
    )
    linhas_saida.append(cabecalho_tabela)
    linhas_saida.append("─" * 80)

    for resultado_molecular in lista_moleculas_similares:
        posicao = resultado_molecular["posicao"]
        id_nubbed = resultado_molecular["id_molecula"]
        smiles_resultado = resultado_molecular["smiles_canonico"]
        score_similaridade = resultado_molecular["similaridade"]

        # Trunca SMILES longos para não quebrar a tabela
        largura_maxima_smiles = 38
        smiles_exibicao = (
            smiles_resultado[:largura_maxima_smiles] + "..."
            if len(smiles_resultado) > largura_maxima_smiles
            else smiles_resultado
        )

        linha_resultado = (
            f"  {posicao:>3}.  {id_nubbed:<20}  {score_similaridade:>12.6f}  "
            f"{smiles_exibicao}"
        )
        linhas_saida.append(linha_resultado)

    linhas_saida.append("═" * 80)
    linhas_saida.append("")

    return "\n".join(linhas_saida)


def executar_busca_por_similaridade(
    smiles_entrada_usuario: str,
    quantidade_resultados: int,
    caminho_banco_vetorial: str,
    dispositivo_computacao: str = None,
) -> None:
    """
    Executa o fluxo completo de busca por similaridade molecular:
    validar SMILES → gerar embedding → consultar ChromaDB → exibir resultados.

    Parâmetros:
        smiles_entrada_usuario:  SMILES fornecido pelo usuário.
        quantidade_resultados:   Número K de vizinhos a retornar.
        caminho_banco_vetorial:  Diretório do ChromaDB persistente.
        dispositivo_computacao:  'cuda', 'cpu' ou None (automático).
    """
    # ── Etapa 1: Validar e canonicalizar SMILES query ─────────────────────
    print(f"\nValidando SMILES: '{smiles_entrada_usuario}' ...", flush=True)
    smiles_consulta_canonico = validar_smiles_unico(smiles_entrada_usuario)
    print(f"SMILES canônico: '{smiles_consulta_canonico}'", flush=True)

    # ── Etapa 2: Carregar modelo e gerar embedding da query ───────────────
    print("Carregando MolFormer-XL e gerando embedding...", flush=True)
    vetorizador_molecular = VetorizadorMolFormer(dispositivo=dispositivo_computacao)
    embedding_molecula_consulta = vetorizador_molecular.vetorizar_molecula(
        smiles_consulta_canonico
    )

    if embedding_molecula_consulta is None:
        raise RuntimeError(
            f"Não foi possível gerar embedding para: '{smiles_consulta_canonico}'"
        )

    # ── Etapa 3: Conectar ao banco e buscar vizinhos mais próximos ─────────
    print("Consultando banco vetorial ChromaDB...", flush=True)
    gerenciador_banco = GerenciadorBancoVetorial(
        caminho_banco_vetorial,
        nome_colecao=NOME_COLECAO_NUBBED,
    )
    total_indexado = gerenciador_banco.total_moleculas_indexadas()

    if total_indexado == 0:
        print(
            "\nERRO: Banco vetorial vazio. Execute 'popular_banco.py' primeiro.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    lista_moleculas_similares = gerenciador_banco.buscar_moleculas_similares(
        embedding_consulta=embedding_molecula_consulta,
        quantidade_resultados=quantidade_resultados,
    )

    # ── Etapa 4: Exibir resultados formatados ─────────────────────────────
    tabela_formatada = formatar_tabela_resultados(
        lista_moleculas_similares=lista_moleculas_similares,
        smiles_consulta_canonico=smiles_consulta_canonico,
        quantidade_solicitada=quantidade_resultados,
    )
    print(tabela_formatada)


def main() -> None:
    """Ponto de entrada principal da CLI de busca por similaridade."""
    parser = construir_parser_busca()
    argumentos = parser.parse_args()

    try:
        executar_busca_por_similaridade(
            smiles_entrada_usuario=argumentos.smiles_consulta,
            quantidade_resultados=argumentos.quantidade_resultados,
            caminho_banco_vetorial=argumentos.caminho_banco,
            dispositivo_computacao=argumentos.dispositivo,
        )
    except ValueError as erro_smiles:
        print(f"\nERRO: {erro_smiles}\n", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as erro_banco:
        print(
            f"\nERRO: Banco vetorial não encontrado em '{argumentos.caminho_banco}'. "
            "Execute 'popular_banco.py' primeiro.\n",
            file=sys.stderr,
        )
        sys.exit(1)
    except RuntimeError as erro_execucao:
        print(f"\nERRO: {erro_execucao}\n", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nBusca interrompida pelo usuário.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
