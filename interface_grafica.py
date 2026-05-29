"""
interface_grafica.py
Interface web para busca por similaridade molecular.
Design inspirado no COCONUT Natural Products Database.

Ao iniciar, verifica se o banco vetorial está populado.
Se estiver vazio, executa o pipeline de população com barra de progresso
antes de liberar a interface de busca.

Uso:
    streamlit run interface_grafica.py
"""

import base64
import glob
import io
import logging

import streamlit as st
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw

from carregador_nubbed import carregar_moleculas_nubbed
from gerenciador_banco_vetorial import GerenciadorBancoVetorial
from preparador_smiles import canonicalizar_smiles
from vetorizador_molformer import VetorizadorMolFormer

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.WARNING)

CAMINHO_BANCO   = "./banco_vetorial"
NOME_COLECAO    = "moleculas_nubbed"
CAMINHO_SDF     = "nubbedb-05-2026.sdf"
TAMANHO_LOTE    = 64

EXEMPLOS = {
    "Aspirina":    "CC(=O)Oc1ccccc1C(=O)O",
    "Cafeína":     "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
    "Quercetina":  "O=c1c(oc2cc(O)cc(O)c2c1=O)-c1ccc(O)c(O)c1",
    "Resveratrol": "Oc1ccc(/C=C/c2cc(O)cc(O)c2)cc1",
}

# ── Configuração da página ────────────────────────────────────────────────────

st.set_page_config(
    page_title="BuscadorMolecular — NuBBED",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
footer { visibility: hidden; }
[data-testid="stHeader"] { display: none; }
.block-container {
    max-width: 1200px !important;
    padding: 2.5rem 3rem 4rem !important;
    margin: 0 auto !important;
}
.bm-title {
    font-size: 2rem; font-weight: 700; color: #1a1a1a;
    margin: 0 0 0.35rem; line-height: 1.2;
}
.bm-subtitle {
    font-size: 0.9rem; color: #555; margin: 0 0 1.8rem;
    max-width: 520px; line-height: 1.6;
}
[data-testid="stTextInput"] input {
    border: 2px solid #1a1a1a !important;
    border-radius: 8px !important;
    background: #fff !important;
    color: #1a1a1a !important;
    font-size: 1rem !important;
    padding: 0.6rem 1rem !important;
    box-shadow: none !important;
}
[data-testid="stTextInput"] input:focus { border-color: #1e7a47 !important; }
[data-testid="stTextInput"] input::placeholder { color: #aaa !important; }
button[kind="primary"] {
    background: #1e7a47 !important;
    border: none !important;
    border-radius: 8px !important;
    color: #fff !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}
button[kind="primary"]:hover { background: #155c35 !important; }
button[kind="secondary"] {
    background: #f4f4f4 !important;
    border: 1px solid #ddd !important;
    border-radius: 20px !important;
    color: #444 !important;
    font-size: 0.8rem !important;
    padding: 0.2rem 0.9rem !important;
}
button[kind="secondary"]:hover { background: #e8e8e8 !important; border-color: #bbb !important; }
[data-testid="stSlider"] label,
[data-testid="stSelectbox"] label { font-size: 0.85rem !important; }
.bm-count { font-size: 0.88rem; color: #555; margin: 0 0 1.2rem; }
.bm-count strong { color: #1a1a1a; }
.mol-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
}
.mol-card {
    border: 1px solid #e0e0e0; border-radius: 12px;
    overflow: hidden; background: #fff;
    transition: box-shadow 0.15s ease;
}
.mol-card:hover { box-shadow: 0 4px 18px rgba(0,0,0,0.10); }
.mol-card-img {
    background: #f5f5f5;
    display: flex; align-items: center; justify-content: center;
    height: 175px; overflow: hidden;
}
.mol-card-img img { width: 100%; height: 100%; object-fit: contain; padding: 0.6rem; }
.mol-card-body { padding: 0.75rem 0.9rem 0.9rem; border-top: 1px solid #efefef; }
.mol-stars   { color: #f5a623; font-size: 0.82rem; letter-spacing: 2px; margin-bottom: 0.3rem; }
.mol-card-id { font-size: 0.78rem; color: #444; font-weight: 600; margin-bottom: 0.15rem; }
.mol-card-sim { font-size: 0.85rem; color: #1e7a47; font-weight: 700; margin-bottom: 0.25rem; }
.mol-card-smiles {
    font-family: 'Courier New', monospace; font-size: 0.64rem;
    color: #999; word-break: break-all; line-height: 1.4;
}
hr { border-color: #ebebeb !important; margin: 1.2rem 0 !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ── Utilitários ───────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _carregar_vetorizador(dispositivo: str | None) -> VetorizadorMolFormer:
    return VetorizadorMolFormer(dispositivo=dispositivo)


@st.cache_resource(show_spinner=False)
def _carregar_banco() -> GerenciadorBancoVetorial:
    return GerenciadorBancoVetorial(CAMINHO_BANCO, nome_colecao=NOME_COLECAO)


def _encontrar_sdf() -> str | None:
    if glob.glob(CAMINHO_SDF):
        return CAMINHO_SDF
    candidatos = glob.glob("*.sdf")
    return candidatos[0] if candidatos else None


def _mol_para_png(smiles: str, largura: int = 280, altura: int = 175) -> bytes | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=(largura * 2, altura * 2))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _b64(png: bytes) -> str:
    return base64.b64encode(png).decode()


def _stars(sim: float) -> str:
    filled = round(sim * 5)
    return "★" * filled + "☆" * (5 - filled)


def _render_cards(resultados: list[dict]) -> str:
    cards = []
    for r in resultados:
        png = _mol_para_png(r["smiles_canonico"])
        img_tag = (
            f'<img src="data:image/png;base64,{_b64(png)}" alt="estrutura">'
            if png else ""
        )
        smiles_curto = (
            r["smiles_canonico"][:52] + "…"
            if len(r["smiles_canonico"]) > 52
            else r["smiles_canonico"]
        )
        cards.append(f"""
<div class="mol-card">
  <div class="mol-card-img">{img_tag}</div>
  <div class="mol-card-body">
    <div class="mol-stars">{_stars(r["similaridade"])}</div>
    <div class="mol-card-id">{r["id_molecula"]}</div>
    <div class="mol-card-sim">{r["similaridade"] * 100:.1f}% de similaridade</div>
    <div class="mol-card-smiles">{smiles_curto}</div>
  </div>
</div>""")
    return '<div class="mol-grid">' + "".join(cards) + "</div>"


# ── Tela de inicialização do banco ────────────────────────────────────────────

def _tela_inicializacao(caminho_sdf: str) -> None:
    """Popula o banco vetorial com feedback visual em tempo real."""

    st.markdown("### Inicializando banco vetorial")
    st.caption(f"Arquivo de entrada: `{caminho_sdf}`")
    st.markdown("---")

    # ── Etapas 1–3: carregamento e validação ──────────────────────────────────
    with st.status("Preparando moléculas...", expanded=True) as status_prep:
        st.write("📂 Carregando base NuBBED do arquivo SDF...")
        moleculas_brutas = carregar_moleculas_nubbed(caminho_sdf)
        total_sdf = len(moleculas_brutas)
        st.write(f"✔ {total_sdf:,} moléculas carregadas do SDF")

        st.write("🧪 Validando e canonicalizando SMILES via RDKit...")
        pares_validos: list[tuple[str, str]] = []
        for mol in moleculas_brutas:
            can = canonicalizar_smiles(mol["smiles_bruto"])
            if can:
                pares_validos.append((can, mol["id_nubbed"]))
        total_validos = len(pares_validos)
        st.write(f"✔ {total_validos:,} SMILES válidos ({total_sdf - total_validos} descartados)")

        st.write("🗄️ Verificando entradas já presentes no banco...")
        banco = GerenciadorBancoVetorial(CAMINHO_BANCO, nome_colecao=NOME_COLECAO)
        ids_existentes = banco.obter_ids_existentes()
        pendentes = [(s, i) for s, i in pares_validos if i not in ids_existentes]
        total_pendentes = len(pendentes)

        if total_pendentes == 0:
            st.write("✅ Banco já está completo — nenhuma inserção necessária.")
            status_prep.update(label="Banco vetorial pronto!", state="complete")
            return

        st.write(
            f"✔ {len(ids_existentes):,} já inseridas · "
            f"**{total_pendentes:,} moléculas pendentes**"
        )
        status_prep.update(label="Preparação concluída", state="complete")

    # ── Etapa 4: vetorização e inserção com barra de progresso ────────────────
    st.markdown(f"**Vetorizando {total_pendentes:,} moléculas com MolFormer-XL...**")
    barra       = st.progress(0.0)
    texto_lote  = st.empty()

    vetorizador = VetorizadorMolFormer(dispositivo=None)
    smiles_list = [p[0] for p in pendentes]
    ids_list    = [p[1] for p in pendentes]
    total_inserido = 0

    for ini in range(0, total_pendentes, TAMANHO_LOTE):
        lote_smiles = smiles_list[ini : ini + TAMANHO_LOTE]
        lote_ids    = ids_list[ini : ini + TAMANHO_LOTE]

        embeddings = vetorizador.vetorizar_lote(lote_smiles, tamanho_lote=TAMANHO_LOTE)

        smiles_ok, ids_ok, embs_ok = [], [], []
        for s, id_, emb in zip(lote_smiles, lote_ids, embeddings):
            if emb is not None:
                smiles_ok.append(s)
                ids_ok.append(id_)
                embs_ok.append(emb)

        if ids_ok:
            banco.inserir_lote(ids_ok, embs_ok, smiles_ok, ids_ok)
            total_inserido += len(ids_ok)

        processadas = min(ini + TAMANHO_LOTE, total_pendentes)
        progresso   = processadas / total_pendentes
        barra.progress(progresso)
        texto_lote.caption(
            f"{processadas:,} / {total_pendentes:,} moléculas vetorizadas  ·  "
            f"{total_inserido:,} inseridas no banco"
        )

    barra.progress(1.0)
    texto_lote.empty()
    st.success(
        f"✅ Banco populado com sucesso! "
        f"**{total_inserido:,}** moléculas inseridas. "
        f"Total no banco: **{banco.total_moleculas_indexadas():,}**"
    )
    st.button("Abrir interface de busca →", type="primary", key="btn_avancar")


# ── Tela de busca ─────────────────────────────────────────────────────────────

def _tela_busca() -> None:
    """Interface principal de busca por similaridade."""

    if "smiles_query" not in st.session_state:
        st.session_state["smiles_query"] = ""

    st.markdown("""
<div class="bm-title">Buscar compostos</div>
<div class="bm-subtitle">
  Explore a base NuBBED de produtos naturais e descubra compostos similares usando embeddings do MolFormer-XL.
</div>
""", unsafe_allow_html=True)

    col_input, col_btn = st.columns([8, 1])
    with col_input:
        smiles_digitado = st.text_input(
            "smiles",
            value=st.session_state["smiles_query"],
            placeholder="Cole ou digite o SMILES da molécula query",
            label_visibility="collapsed",
        )
    with col_btn:
        st.markdown("<div style='padding-top:1.75rem'>", unsafe_allow_html=True)
        buscar = st.button("🔍 Buscar", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    k = st.slider("Resultados (K)", min_value=1, max_value=50, value=10)
    dispositivo = None

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

        with st.spinner("Consultando banco vetorial..."):
            banco = _carregar_banco()
            embedding   = vetorizador.vetorizar_molecula(smiles_canonico)

        if embedding is None:
            st.error(f"Não foi possível gerar embedding para `{smiles_canonico}`.")
            st.stop()

        resultados = banco.buscar_moleculas_similares(embedding, k)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="bm-count">Mostrando <strong>1 a {len(resultados)}</strong> '
            f'de <strong>{len(resultados)}</strong> resultados</div>',
            unsafe_allow_html=True,
        )
        st.markdown(_render_cards(resultados), unsafe_allow_html=True)


# ── Roteamento principal ──────────────────────────────────────────────────────

def main() -> None:
    banco_temp = GerenciadorBancoVetorial(CAMINHO_BANCO, nome_colecao=NOME_COLECAO)
    banco_vazio = banco_temp.total_moleculas_indexadas() == 0

    if banco_vazio:
        caminho_sdf = _encontrar_sdf()
        if caminho_sdf is None:
            st.error(
                "Nenhum arquivo `.sdf` encontrado na pasta do projeto. "
                f"Coloque o arquivo `{CAMINHO_SDF}` na mesma pasta que `interface_grafica.py`."
            )
            st.stop()
        _tela_inicializacao(caminho_sdf)
    else:
        _tela_busca()


main()
