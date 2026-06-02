#!/usr/bin/env python
"""
exportar_logs.py — CLI para exportar logs de busca do BuscadorMolecular.

Uso:
    python exportar_logs.py [opcoes]

Exemplos:
    python exportar_logs.py
    python exportar_logs.py --formato csv --saida logs.csv
    python exportar_logs.py --metodo "MolFormer-XL"
    python exportar_logs.py --metodo "ChemBERTa-2" --desde 2026-01-01
    python exportar_logs.py --ultimas 50 --saida recentes.json
"""

import argparse
import csv
import io
import json
import sqlite3
import sys
from pathlib import Path

CAMINHO_DB = Path("./logs/buscas.db")


def _conectar() -> sqlite3.Connection:
    if not CAMINHO_DB.exists():
        print(f"[ERRO] Banco nao encontrado: {CAMINHO_DB}", file=sys.stderr)
        print("Execute o BuscadorMolecular e realize ao menos uma busca.", file=sys.stderr)
        sys.exit(1)
    conn = sqlite3.connect(CAMINHO_DB)
    conn.row_factory = sqlite3.Row
    return conn


def _buscar_registros(
    conn: sqlite3.Connection,
    metodo: str | None,
    desde: str | None,
    ultimas: int | None,
) -> list[dict]:
    filtros: list[str] = []
    params: list = []

    if metodo:
        filtros.append("metodo = ?")
        params.append(metodo)
    if desde:
        filtros.append("timestamp >= ?")
        params.append(desde)

    where = ("WHERE " + " AND ".join(filtros)) if filtros else ""
    limit = f"LIMIT {ultimas}" if ultimas else ""

    buscas = conn.execute(
        f"SELECT id, timestamp, metodo, query FROM buscas {where} ORDER BY id DESC {limit}",
        params,
    ).fetchall()

    registros = []
    for b in buscas:
        retornos = conn.execute(
            "SELECT posicao, smile_retorno, percentual_relevancia "
            "FROM retornos WHERE busca_id = ? ORDER BY posicao",
            (b["id"],),
        ).fetchall()
        registros.append({
            "timestamp": b["timestamp"],
            "metodo_busca": b["metodo"],
            "query": b["query"],
            "retornos": [
                {
                    "smile_retorno": r["smile_retorno"],
                    "percentual_relevancia": r["percentual_relevancia"],
                }
                for r in retornos
            ],
        })
    return registros


def _exportar_json(registros: list[dict], saida: str | None) -> None:
    texto = json.dumps(registros, ensure_ascii=False, indent=2)
    if saida:
        Path(saida).write_text(texto, encoding="utf-8")
        print(f"OK {len(registros)} buscas exportadas -> {saida}")
    else:
        print(texto)


def _exportar_csv(registros: list[dict], saida: str | None) -> None:
    campos = [
        "timestamp", "metodo_busca", "query",
        "posicao", "smile_retorno", "percentual_relevancia",
    ]
    linhas = [
        {
            "timestamp": reg["timestamp"],
            "metodo_busca": reg["metodo_busca"],
            "query": reg["query"],
            "posicao": idx + 1,
            "smile_retorno": ret["smile_retorno"],
            "percentual_relevancia": ret["percentual_relevancia"],
        }
        for reg in registros
        for idx, ret in enumerate(reg["retornos"])
    ]

    if saida:
        with open(saida, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=campos)
            writer.writeheader()
            writer.writerows(linhas)
        print(f"OK {len(registros)} buscas ({len(linhas)} linhas) exportadas -> {saida}")
    else:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=campos)
        writer.writeheader()
        writer.writerows(linhas)
        print(buf.getvalue())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exporta logs de busca do BuscadorMolecular.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--formato", choices=["json", "csv"], default="json",
        help="Formato de saida: json (padrao) ou csv",
    )
    parser.add_argument(
        "--saida", metavar="ARQUIVO",
        help="Arquivo de destino (padrao: stdout)",
    )
    parser.add_argument(
        "--metodo", metavar="METODO",
        help="Filtrar por metodo: 'MolFormer-XL', 'ChemBERTa-2' ou 'Fingerprints (RDKit)'",
    )
    parser.add_argument(
        "--desde", metavar="YYYY-MM-DD",
        help="Filtrar buscas a partir de uma data (ex: 2026-01-15)",
    )
    parser.add_argument(
        "--ultimas", metavar="N", type=int,
        help="Exportar apenas as N buscas mais recentes",
    )

    parser.add_argument(
        "--limpar", action="store_true",
        help="Apaga todos os logs do banco (irreversivel)",
    )

    args = parser.parse_args()

    if args.limpar:
        if not CAMINHO_DB.exists():
            print("Nenhum banco de logs encontrado. Nada a apagar.")
            sys.exit(0)
        conn = sqlite3.connect(CAMINHO_DB)
        conn.execute("DELETE FROM retornos")
        conn.execute("DELETE FROM buscas")
        conn.commit()
        conn.execute("VACUUM")
        conn.close()
        print("OK Todos os logs foram apagados.")
        sys.exit(0)

    conn = _conectar()

    try:
        registros = _buscar_registros(conn, args.metodo, args.desde, args.ultimas)
    finally:
        conn.close()

    if not registros:
        print("Nenhum registro encontrado com os filtros fornecidos.", file=sys.stderr)
        sys.exit(0)

    if args.formato == "json":
        _exportar_json(registros, args.saida)
    else:
        _exportar_csv(registros, args.saida)


if __name__ == "__main__":
    main()
