"""
vetorizador_chemberta.py
Geração de embeddings moleculares usando ChemBERTa-2 (DeepChem).
Mean pooling + normalização L2, mesma interface do VetorizadorMolFormer.
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

NOME_MODELO_CHEMBERTA = "DeepChem/ChemBERTa-77M-MLM"


class VetorizadorChemBERTa:
    """Embeddings moleculares via ChemBERTa-2 com mean pooling L2-normalizado."""

    def __init__(self, dispositivo: Optional[str] = None) -> None:
        if dispositivo is None:
            self.dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.dispositivo = dispositivo

        logger.info("Carregando ChemBERTa-2 em dispositivo: %s", self.dispositivo)
        self.tokenizador = AutoTokenizer.from_pretrained(NOME_MODELO_CHEMBERTA)
        self.modelo = AutoModel.from_pretrained(NOME_MODELO_CHEMBERTA)
        self.modelo.to(self.dispositivo)
        self.modelo.eval()
        logger.info("ChemBERTa-2 carregado com sucesso.")

    def _mean_pooling(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mascara = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        soma = torch.sum(last_hidden_state * mascara, dim=1)
        contagem = torch.clamp(mascara.sum(dim=1), min=1e-9)
        return soma / contagem

    def vetorizar_molecula(self, smiles_canonico: str) -> Optional[list[float]]:
        try:
            tokens = self.tokenizador(
                smiles_canonico,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            tokens = {k: v.to(self.dispositivo) for k, v in tokens.items()}
            with torch.no_grad():
                saida = self.modelo(**tokens)
            emb = self._mean_pooling(saida.last_hidden_state, tokens["attention_mask"])
            emb = F.normalize(emb, p=2, dim=-1)
            return emb.squeeze(0).cpu().tolist()
        except Exception as e:
            logger.error("Erro ao vetorizar '%s': %s", smiles_canonico, e)
            return None

    def vetorizar_lote(
        self,
        lista_smiles: list[str],
        tamanho_lote: int = 32,
    ) -> list[Optional[list[float]]]:
        resultados: list[Optional[list[float]]] = []
        for ini in range(0, len(lista_smiles), tamanho_lote):
            sub = lista_smiles[ini : ini + tamanho_lote]
            try:
                tokens = self.tokenizador(
                    sub,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                tokens = {k: v.to(self.dispositivo) for k, v in tokens.items()}
                with torch.no_grad():
                    saida = self.modelo(**tokens)
                embs = self._mean_pooling(saida.last_hidden_state, tokens["attention_mask"])
                embs = F.normalize(embs, p=2, dim=-1)
                resultados.extend(embs.cpu().tolist())
            except Exception as e:
                logger.warning("Falha sub-lote, reprocessando individual. Erro: %s", e)
                for s in sub:
                    resultados.append(self.vetorizar_molecula(s))
        return resultados