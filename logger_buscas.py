import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

CAMINHO_DB = Path("./logs/buscas.db")


def _inicializar_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS buscas (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT    NOT NULL,
            metodo    TEXT    NOT NULL,
            query     TEXT    NOT NULL
        );
        CREATE TABLE IF NOT EXISTS retornos (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            busca_id              INTEGER NOT NULL REFERENCES buscas(id),
            posicao               INTEGER NOT NULL,
            smile_retorno         TEXT    NOT NULL,
            percentual_relevancia REAL    NOT NULL
        );
    """)
    conn.commit()


@contextmanager
def _conexao():
    CAMINHO_DB.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(CAMINHO_DB)
    conn.row_factory = sqlite3.Row
    try:
        _inicializar_db(conn)
        yield conn
    finally:
        conn.close()


def registrar_busca(metodo: str, query: str, resultados: list[dict]) -> None:
    with _conexao() as conn:
        cur = conn.execute(
            "INSERT INTO buscas (timestamp, metodo, query) VALUES (?, ?, ?)",
            (datetime.now().isoformat(), metodo, query),
        )
        busca_id = cur.lastrowid
        conn.executemany(
            "INSERT INTO retornos (busca_id, posicao, smile_retorno, percentual_relevancia) "
            "VALUES (?, ?, ?, ?)",
            [
                (busca_id, i + 1, r["smiles_canonico"], r["similaridade"])
                for i, r in enumerate(resultados[:20])
            ],
        )
        conn.commit()
    logger.info("Busca registrada: método=%s query=%s", metodo, query)
