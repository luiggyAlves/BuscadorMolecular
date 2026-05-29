"""
vetorizador_molformer.py
Encapsula o carregamento do modelo MolFormer-XL (IBM) e a geração de
embeddings moleculares a partir de strings SMILES.
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

NOME_MODELO_MOLFORMER = "ibm-research/MoLFormer-XL-both-10pct"


class VetorizadorMolFormer:
    """
    Encapsula o modelo MolFormer-XL da IBM para geração de embeddings
    moleculares a partir de SMILES. Suporta GPU quando disponível.
    """

    def __init__(self, dispositivo: Optional[str] = None) -> None:
        """
        Inicializa o tokenizador e o modelo MolFormer-XL.

        Parâmetros:
            dispositivo: 'cuda', 'cpu' ou None (detecção automática).
        """
        if dispositivo is None:
            self.dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.dispositivo = dispositivo

        logger.info("Carregando MolFormer-XL em dispositivo: %s", self.dispositivo)

        self.tokenizador_molecular = AutoTokenizer.from_pretrained(
            NOME_MODELO_MOLFORMER,
            trust_remote_code=True,
        )
        self.modelo_molformer = AutoModel.from_pretrained(
            NOME_MODELO_MOLFORMER,
            trust_remote_code=True,
            deterministic_eval=True,
        )
        self.modelo_molformer.to(self.dispositivo)
        self.modelo_molformer.eval()
        logger.info("Modelo MolFormer-XL carregado com sucesso.")

    def _mean_pooling(
        self,
        saida_modelo: torch.Tensor,
        mascara_atencao: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aplica mean pooling sobre os hidden states da última camada,
        respeitando a attention_mask para ignorar tokens de padding.

        Parâmetros:
            saida_modelo:   Tensor (batch, seq_len, hidden_size) com last_hidden_state.
            mascara_atencao: Tensor (batch, seq_len) com 1 para tokens reais, 0 para padding.

        Retorna:
            Tensor (batch, hidden_size) com embedding médio por sequência.
        """
        # Expande máscara para mesma dimensão dos hidden states
        mascara_expandida = mascara_atencao.unsqueeze(-1).expand(saida_modelo.size()).float()
        soma_estados = torch.sum(saida_modelo * mascara_expandida, dim=1)
        contagem_tokens_validos = torch.clamp(mascara_expandida.sum(dim=1), min=1e-9)
        return soma_estados / contagem_tokens_validos

    def _normalizar_l2(self, vetor_embedding: torch.Tensor) -> torch.Tensor:
        """
        Normaliza embeddings pela norma L2 para que a distância de cosseno
        seja equivalente ao produto escalar.

        Parâmetros:
            vetor_embedding: Tensor (batch, hidden_size) ou (hidden_size,).

        Retorna:
            Tensor normalizado de mesma forma.
        """
        return F.normalize(vetor_embedding, p=2, dim=-1)

    def vetorizar_molecula(self, smiles_canonico: str) -> Optional[list[float]]:
        """
        Gera o embedding normalizado de uma única molécula SMILES.

        Parâmetros:
            smiles_canonico: String SMILES canônica e validada.

        Retorna:
            Lista de floats representando o embedding, ou None em caso de erro.
        """
        try:
            tokens_molecula = self.tokenizador_molecular(
                smiles_canonico,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            tokens_molecula = {
                chave: tensor.to(self.dispositivo)
                for chave, tensor in tokens_molecula.items()
            }

            with torch.no_grad():
                saida_bruta = self.modelo_molformer(**tokens_molecula)

            embedding_pooled = self._mean_pooling(
                saida_bruta.last_hidden_state,
                tokens_molecula["attention_mask"],
            )
            embedding_normalizado = self._normalizar_l2(embedding_pooled)
            return embedding_normalizado.squeeze(0).cpu().tolist()

        except Exception as excecao_vetorizacao:
            logger.error(
                "Erro ao vetorizar SMILES '%s': %s",
                smiles_canonico,
                excecao_vetorizacao,
            )
            return None

    def vetorizar_lote(
        self,
        lista_smiles: list[str],
        tamanho_lote: int = 64,
    ) -> list[Optional[list[float]]]:
        """
        Gera embeddings para um lote de moléculas SMILES.
        Processa em sub-lotes para controle de memória. Moléculas que causam
        erros recebem None na posição correspondente sem interromper o lote.

        Parâmetros:
            lista_smiles:  Lista de strings SMILES canônicas.
            tamanho_lote:  Tamanho do sub-lote para inferência.

        Retorna:
            Lista de embeddings (list[float]) ou None por posição.
        """
        lista_embeddings_resultado: list[Optional[list[float]]] = []

        for indice_inicio in range(0, len(lista_smiles), tamanho_lote):
            sublote_smiles = lista_smiles[indice_inicio: indice_inicio + tamanho_lote]

            try:
                tokens_sublote = self.tokenizador_molecular(
                    sublote_smiles,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                tokens_sublote = {
                    chave: tensor.to(self.dispositivo)
                    for chave, tensor in tokens_sublote.items()
                }

                with torch.no_grad():
                    saida_sublote = self.modelo_molformer(**tokens_sublote)

                embeddings_sublote = self._mean_pooling(
                    saida_sublote.last_hidden_state,
                    tokens_sublote["attention_mask"],
                )
                embeddings_normalizados = self._normalizar_l2(embeddings_sublote)
                embeddings_cpu = embeddings_normalizados.cpu().tolist()
                lista_embeddings_resultado.extend(embeddings_cpu)

            except Exception as excecao_lote:
                logger.warning(
                    "Falha no sub-lote [%d:%d]. Reprocessando molécula a molécula. Erro: %s",
                    indice_inicio,
                    indice_inicio + tamanho_lote,
                    excecao_lote,
                )
                # Fallback: tenta molécula por molécula para isolar a falha
                for smiles_individual in sublote_smiles:
                    embedding_individual = self.vetorizar_molecula(smiles_individual)
                    lista_embeddings_resultado.append(embedding_individual)

        return lista_embeddings_resultado
