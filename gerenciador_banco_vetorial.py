"""
gerenciador_banco_vetorial.py
Encapsula a conexão com ChromaDB persistente, configuração da coleção
com distância de cosseno, inserção em lotes e consulta de vizinhos.
"""

import logging
from typing import Optional

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

NOME_PADRAO_COLECAO = "moleculas_coconut"


class GerenciadorBancoVetorial:
    """
    Gerencia o banco vetorial ChromaDB para armazenamento e consulta
    de embeddings moleculares com métrica de cosseno.
    """

    def __init__(
        self,
        caminho_persistencia: str,
        nome_colecao: str = NOME_PADRAO_COLECAO,
    ) -> None:
        """
        Inicializa cliente ChromaDB persistente e obtém/cria a coleção
        configurada com espaço de cosseno.

        Parâmetros:
            caminho_persistencia: Diretório onde o ChromaDB será salvo em disco.
            nome_colecao:         Nome da coleção (padrão: 'moleculas_coconut').
        """
        self.caminho_persistencia = caminho_persistencia
        self.nome_colecao = nome_colecao

        logger.info(
            "Conectando ao ChromaDB em '%s', coleção '%s'.",
            caminho_persistencia,
            nome_colecao,
        )

        self.cliente_chroma = chromadb.PersistentClient(
            path=caminho_persistencia,
            settings=Settings(anonymized_telemetry=False),
        )

        # Cria ou recupera coleção com distância de cosseno
        self.colecao_moleculas = self.cliente_chroma.get_or_create_collection(
            name=nome_colecao,
            metadata={"hnsw:space": "cosine"},
        )

        total_existente = self.colecao_moleculas.count()
        logger.info(
            "Coleção '%s' inicializada com %d moléculas já presentes.",
            nome_colecao,
            total_existente,
        )

    def obter_ids_existentes(self) -> set[str]:
        """
        Retorna o conjunto de IDs de moléculas já inseridas na coleção.
        Usado para garantir idempotência na população do banco.

        Retorna:
            Conjunto (set) de strings com os IDs existentes.
        """
        total_moleculas = self.colecao_moleculas.count()
        if total_moleculas == 0:
            return set()

        # Recupera todos os IDs em uma única chamada
        resposta_ids = self.colecao_moleculas.get(include=[])
        conjunto_ids_existentes = set(resposta_ids["ids"])
        logger.debug(
            "Encontrados %d IDs já inseridos no banco.",
            len(conjunto_ids_existentes),
        )
        return conjunto_ids_existentes

    def inserir_lote(
        self,
        lista_ids_insercao: list[str],
        lista_embeddings_moleculares: list[list[float]],
        lista_smiles_canonicos: list[str],
        lista_ids_molecula: list[str],
    ) -> None:
        """
        Insere um lote de moléculas no ChromaDB com seus embeddings e metadados.

        Parâmetros:
            lista_ids_insercao:          IDs únicos para cada entrada no ChromaDB.
            lista_embeddings_moleculares: Embeddings L2-normalizados de cada molécula.
            lista_smiles_canonicos:       SMILES canônicos para armazenar nos metadados.
            lista_ids_molecula:           Identificadores originais da base de dados.
        """
        if not lista_ids_insercao:
            return

        lista_metadados_moleculares = [
            {"smiles_canonico": smiles, "id_molecula": id_mol}
            for smiles, id_mol in zip(lista_smiles_canonicos, lista_ids_molecula)
        ]

        self.colecao_moleculas.add(
            ids=lista_ids_insercao,
            embeddings=lista_embeddings_moleculares,
            metadatas=lista_metadados_moleculares,
        )
        logger.debug("Inseridas %d moléculas no ChromaDB.", len(lista_ids_insercao))

    def buscar_moleculas_similares(
        self,
        embedding_consulta: list[float],
        quantidade_resultados: int = 10,
    ) -> list[dict]:
        """
        Busca as K moléculas mais similares ao embedding de consulta usando
        distância de cosseno. Converte distância para score de similaridade.

        Parâmetros:
            embedding_consulta:   Embedding L2-normalizado da molécula query.
            quantidade_resultados: Número K de vizinhos mais próximos a retornar.

        Retorna:
            Lista de dicionários com campos:
                - posicao:          Posição no ranking (1-indexado).
                - id_coconut:       Identificador COCONUT da molécula.
                - smiles_canonico:  SMILES canônico da molécula encontrada.
                - similaridade:     Score de similaridade (1 - distância_cosseno).
        """
        total_disponivel = self.colecao_moleculas.count()
        quantidade_real = min(quantidade_resultados, total_disponivel)

        if quantidade_real == 0:
            logger.warning("Banco vetorial vazio. Nenhum resultado retornado.")
            return []

        resposta_consulta = self.colecao_moleculas.query(
            query_embeddings=[embedding_consulta],
            n_results=quantidade_real,
            include=["metadatas", "distances"],
        )

        lista_distancias = resposta_consulta["distances"][0]
        lista_metadados = resposta_consulta["metadatas"][0]

        lista_moleculas_similares: list[dict] = []
        for posicao_ranking, (distancia_cosseno, metadados_molecula) in enumerate(
            zip(lista_distancias, lista_metadados), start=1
        ):
            score_similaridade = 1.0 - float(distancia_cosseno)
            lista_moleculas_similares.append(
                {
                    "posicao": posicao_ranking,
                    "id_molecula": metadados_molecula.get("id_molecula", "N/A"),
                    "smiles_canonico": metadados_molecula.get("smiles_canonico", "N/A"),
                    "similaridade": round(score_similaridade, 6),
                }
            )

        return lista_moleculas_similares

    def total_moleculas_indexadas(self) -> int:
        """Retorna o número total de moléculas presentes na coleção."""
        return self.colecao_moleculas.count()
