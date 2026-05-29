"""
interface_grafica.py
Interface web para busca por similaridade molecular.
Inspirada no COCONUT Natural Products Database.

Uso:
    streamlit run interface_grafica.py
"""

import io
import logging

import streamlit as st
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw

from gerenciador_banco_vetorial import GerenciadorBancoVetorial
from preparador_smiles import canonicalizar_smiles
from vetorizador_molformer import VetorizadorMolFormer

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.WARNING)

CAMINHO_BANCO = "./banco_vetorial"
NOME_COLECAO = "moleculas_nubbed"

EXEMPLOS = {
    "Aspirina": "CC(=O)Oc1ccccc1C(=O)O",
    "Cafeína": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
    "Quercetina": "O=c1c(oc2cc(O)cc(O)c2c1=O)-c1ccc(O)c(O)c1",
    "Resveratrol": "Oc1ccc(/C=C/c2cc(O)cc(O)c2)cc1",
}

# ── Configuração da página ────────────────────────────────────────────────────

st.set_page_config(
    page_title="BuscadorMolecular — NuBBED",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* Fundo */
[data-testid="stAppViewContainer"] > .main { background-color: #f4f6f9; }
[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }

/* Hero */
.bm-hero {
    background: linear-gradient(135deg, #0d3d22 0%, #1e7a47 100%);
    color: white;
    text-align: center;
    padding: 2.8rem 1.5rem 2.4rem;
    margin: -4rem -4rem 2rem -4rem;
}
.bm-hero h1 { font-size: 2.4rem; font-weight: 800; margin-bottom: 0.4rem; letter-spacing: -0.5px; }
.bm-hero p  { font-size: 1rem; opacity: 0.85; margin: 0; }
.bm-hero .bm-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    color: white;
    padding: 0.25rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    margin-top: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}

/* Painel de busca */
.bm-search-panel {
    background: white;
    border-radius: 14px;
    padding: 1.5rem 1.5rem 1.2rem;
    box-shadow: 0 2px 14px rgba(0,0,0,0.07);
    margin-bottom: 1.5rem;
}

/* Botão buscar */
div[data-testid="stButton"] > button[kind="primary"] {
    background: #1e7a47 !important;
    border-color: #1e7a47 !important;
    font-weight: 700;
    font-size: 1rem;
    border-radius: 8px;
    height: 2.7rem;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #155c35 !important;
    border-color: #155c35 !important;
}

/* Barra de resumo de resultados */
.bm-result-bar {
    background: #e6f4ec;
    border-left: 5px solid #1e7a47;
    padding: 0.7rem 1rem;
    border-radius: 0 10px 10px 0;
    margin-bottom: 1.5rem;
    font-size: 0.95rem;
    color: #0d3d22;
    font-weight: 600;
}

/* Cabeçalho de resultado individual */
.bm-card-rank  { font-size: 0.72rem; color: #888; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; }
.bm-card-id    { font-size: 1rem; font-weight: 700; color: #0d3d22; margin: 0.15rem 0 0.4rem; }
.bm-sim-badge  {
    display: inline-block;
    background: #e6f4ec;
    color: #1e7a47;
    font-weight: 700;
    font-size: 0.85rem;
    padding: 0.2rem 0.65rem;
    border-radius: 20px;
    margin-bottom: 0.3rem;
}
.bm-smiles {
    font-family: 'Courier New', monospace;
    font-size: 0.68rem;
    color: #555;
    background: #f8f9fb;
    border-radius: 6px;
    padding: 0.4rem 0.5rem;
    word-break: break-all;
    margin-top: 0.4rem;
    border: 1px solid #e2e6ea;
}

/* Barra de similaridade */
[data-testid="stProgressBar"] > div > div { background: linear-gradient(90deg, #1e7a47, #3ab26e) !important; }
</style>
""", unsafe_allow_html=True)


# ── Cache de recursos pesados ─────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _carregar_vetorizador(dispositivo: str | None) -> VetorizadorMolFormer:
    return VetorizadorMolFormer(dispositivo=dispositivo)


@st.cache_resource(show_spinner=False)
def _carregar_banco() -> GerenciadorBancoVetorial:
    return GerenciadorBancoVetorial(CAMINHO_BANCO, nome_colecao=NOME_COLECAO)


def _mol_para_png(smiles: str, largura: int = 300, altura: int = 200) -> bytes | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=(largura, altura))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Estado de sessão ──────────────────────────────────────────────────────────

if "smiles_query" not in st.session_state:
    st.session_state["smiles_query"] = ""


# ── Hero ──────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="bm-hero">
    <h1>⚗️ BuscadorMolecular</h1>
    <p>Busca por similaridade molecular na base NuBBED</p>
    <span class="bm-badge">MolFormer-XL · ChromaDB · RDKit</span>
</div>
""", unsafe_allow_html=True)


# ── Painel de busca ───────────────────────────────────────────────────────────

st.markdown('<div class="bm-search-panel">', unsafe_allow_html=True)

col_input, col_btn = st.columns([6, 1])
with col_input:
    smiles_digitado = st.text_input(
        "smiles",
        value=st.session_state["smiles_query"],
        placeholder="Cole ou digite o SMILES da molécula — ex: CC(=O)Oc1ccccc1C(=O)O",
        label_visibility="collapsed",
    )
with col_btn:
    st.markdown("<div style='margin-top:1.7rem'>", unsafe_allow_html=True)
    buscar = st.button("🔍 Buscar", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

col_k, col_dev, col_space = st.columns([2, 2, 4])
with col_k:
    k = st.slider("Resultados (K)", min_value=1, max_value=50, value=10)
with col_dev:
    opcao_dev = st.selectbox("Dispositivo", ["Automático", "CPU", "CUDA"])
    dispositivo = {"Automático": None, "CPU": "cpu", "CUDA": "cuda"}[opcao_dev]

st.markdown("**Exemplos rápidos:**")
cols_ex = st.columns(len(EXEMPLOS))
for i, (nome, smi) in enumerate(EXEMPLOS.items()):
    if cols_ex[i].button(nome, key=f"ex_{nome}"):
        st.session_state["smiles_query"] = smi
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)  # fecha bm-search-panel


# ── Execução da busca ─────────────────────────────────────────────────────────

smiles_efetivo = smiles_digitado.strip()

if buscar:
    if not smiles_efetivo:
        st.warning("Digite ou cole um SMILES para buscar.")
        st.stop()

    smiles_canonico = canonicalizar_smiles(smiles_efetivo)
    if smiles_canonico is None:
        st.error("SMILES inválido — verifique a notação e tente novamente.")
        st.stop()

    with st.spinner("Carregando MolFormer-XL..."):
        vetorizador = _carregar_vetorizador(dispositivo)

    with st.spinner("Consultando banco vetorial ChromaDB..."):
        banco = _carregar_banco()
        total_banco = banco.total_moleculas_indexadas()

        if total_banco == 0:
            st.error("Banco vetorial vazio. Execute `python popular_banco.py` primeiro.")
            st.stop()

        embedding = vetorizador.vetorizar_molecula(smiles_canonico)
        if embedding is None:
            st.error(f"Não foi possível gerar embedding para `{smiles_canonico}`.")
            st.stop()

        resultados = banco.buscar_moleculas_similares(embedding, k)

    # ── Molécula query ─────────────────────────────────────────────────────────
    st.markdown("#### Molécula query")
    with st.container(border=True):
        col_qimg, col_qinfo = st.columns([1, 3])
        img_q = _mol_para_png(smiles_canonico, 320, 220)
        if img_q:
            col_qimg.image(img_q)
        with col_qinfo:
            st.markdown("**SMILES canônico**")
            st.code(smiles_canonico, language=None)
            st.caption(
                f"Banco: **{total_banco:,}** moléculas indexadas  ·  "
                f"Resultados solicitados: **{k}**  ·  "
                f"Encontrados: **{len(resultados)}**"
            )

    st.markdown("---")

    # ── Resumo ─────────────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="bm-result-bar">🔎 {len(resultados)} moléculas similares encontradas</div>',
        unsafe_allow_html=True,
    )

    # ── Grade de resultados (3 colunas) ────────────────────────────────────────
    NCOLS = 3
    for fila in range(0, len(resultados), NCOLS):
        colunas = st.columns(NCOLS)
        for j, col in enumerate(colunas):
            idx = fila + j
            if idx >= len(resultados):
                break
            r = resultados[idx]
            with col:
                with st.container(border=True):
                    img = _mol_para_png(r["smiles_canonico"], 300, 200)
                    if img:
                        st.image(img, use_container_width=True)
                    st.markdown(
                        f'<div class="bm-card-rank">#{r["posicao"]}</div>'
                        f'<div class="bm-card-id">{r["id_molecula"]}</div>'
                        f'<div class="bm-sim-badge">Similaridade: {r["similaridade"] * 100:.2f}%</div>'
                        f'<div class="bm-smiles">{r["smiles_canonico"]}</div>',
                        unsafe_allow_html=True,
                    )
                    st.progress(float(r["similaridade"]))
