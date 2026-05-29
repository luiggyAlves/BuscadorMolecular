# BuscadorMolecular

Ferramenta de busca por similaridade molecular usando embeddings do **MolFormer-XL** (IBM) e **ChromaDB** como banco de dados vetorial.

Dado um SMILES query, retorna as K moléculas mais similares presentes na base **NuBBED**, ranqueadas por score de similaridade de cosseno.

---

## Arquitetura

```
vetorizador_molformer.py      ← MolFormer-XL: geração de embeddings
preparador_smiles.py          ← RDKit: validação e canonicalização
gerenciador_banco_vetorial.py ← ChromaDB: inserção e consulta vetorial
carregador_nubbed.py          ← Leitura do SDF da NuBBED
popular_banco.py              ← Script: popula o banco (executável)
buscar_similares.py           ← CLI: busca por query no terminal
interface_grafica.py          ← Interface web: busca visual no navegador
```

---

## Pré-requisitos

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) ou Anaconda instalado
- (Opcional) GPU NVIDIA com CUDA para vetorização mais rápida
- Arquivo SDF da base NuBBED na pasta do projeto

---

## Passo a passo completo

### Passo 1 — Criar ambiente conda com RDKit

> **Importante:** use conda para instalar o RDKit. A instalação via `pip` no ambiente base do Windows pode ser bloqueada pela política de Application Control.

```powershell
conda create -n buscador python=3.10 rdkit -c conda-forge -y
```

### Passo 2 — Ativar o ambiente

```powershell
conda activate buscador
```

### Passo 3 — Instalar demais dependências

```powershell
pip install torch "transformers==4.38.0" einops chromadb pandas tqdm numpy streamlit Pillow
```

> **Versão do transformers:** use obrigatoriamente `transformers==4.38.0`. Versões 4.40+ removeram o módulo `transformers.onnx` que o `configuration_molformer.py` do MolFormer-XL ainda importa, causando `ModuleNotFoundError`.

> **GPU (opcional):** para instalar PyTorch com suporte CUDA, consulte [pytorch.org/get-started](https://pytorch.org/get-started/locally/) e instale a versão compatível com sua versão de CUDA **antes** do comando acima.

### Passo 4 — Entrar na pasta do projeto

```powershell
cd C:\Users\Luiggy\BuscadorMolecular
```

### Passo 5 — Popular o banco vetorial

```powershell
python popular_banco.py --caminho_nubbed nubbedb-05-2026.sdf --caminho_banco ./banco_vetorial
```

O script exibe barra de progresso. Se for interrompido, rode o mesmo comando novamente — ele retoma de onde parou sem reprocessar o que já foi inserido.

### Passo 6a — Buscar via interface gráfica (recomendado)

```powershell
streamlit run interface_grafica.py
```

O navegador abre automaticamente em `http://localhost:8501`. Cole ou digite o SMILES no campo de busca e clique em **Buscar**. Os resultados aparecem como cards com imagem estrutural, ID NuBBED e score de similaridade.

### Passo 6b — Buscar via linha de comando (CLI)

```powershell
python buscar_similares.py --smiles_consulta "CC(=O)Oc1ccccc1C(=O)O" --quantidade_resultados 10 --caminho_banco ./banco_vetorial
```

---

## Parâmetros do popular_banco.py

| Parâmetro | Descrição | Padrão |
|---|---|---|
| `--caminho_nubbed` | Arquivo SDF da NuBBED | obrigatório |
| `--caminho_banco` | Diretório de persistência ChromaDB | `./banco_vetorial` |
| `--tamanho_lote` | Moléculas por batch de vetorização | `64` |
| `--dispositivo` | `cuda` ou `cpu` (auto-detectado se omitido) | automático |

## Interface gráfica (interface_grafica.py)

Aplicação web construída com **Streamlit**, inspirada no [COCONUT Natural Products Database](https://coconut.naturalproducts.net).

**Funcionalidades:**

- Campo de busca por SMILES com exemplos rápidos pré-carregados (Aspirina, Cafeína, Quercetina, Resveratrol)
- Slider para definir o número de resultados K (1–50)
- Seletor de dispositivo: Automático / CPU / CUDA
- Preview da molécula query com imagem estrutural gerada pelo RDKit
- Grade de resultados em 3 colunas com:
  - Imagem estrutural de cada molécula
  - ID NuBBED e posição no ranking
  - Badge de similaridade em percentual
  - Barra de progresso proporcional ao score
  - SMILES completo em fonte monospace
- Carregamento do modelo e do banco em cache — buscas subsequentes são instantâneas

---

## Parâmetros do buscar_similares.py

| Parâmetro | Descrição | Padrão |
|---|---|---|
| `--smiles_consulta` | SMILES da molécula query | obrigatório |
| `--quantidade_resultados` | K vizinhos mais próximos | `10` |
| `--caminho_banco` | Diretório do ChromaDB | `./banco_vetorial` |
| `--dispositivo` | `cuda` ou `cpu` | automático |

---

## Estimativa de tempo (popular_banco.py)

| Hardware | Tempo estimado |
|---|---|
| GPU (RTX 3080) | ~30–60 min |
| CPU (16 cores) | ~2–4 horas |

---

## Exemplo de saída da busca

```
Validando SMILES: 'CC(=O)Oc1ccccc1C(=O)O' ...
SMILES canônico: 'CC(=O)Oc1ccccc1C(=O)O'
Carregando MolFormer-XL e gerando embedding...
Consultando banco vetorial ChromaDB...

════════════════════════════════════════════════════════════════════════════════
  BUSCA POR SIMILARIDADE MOLECULAR — MolFormer-XL + ChromaDB
════════════════════════════════════════════════════════════════════════════════
  Molécula query (SMILES): CC(=O)Oc1ccccc1C(=O)O
  Resultados solicitados:  10
  Resultados encontrados:  10
────────────────────────────────────────────────────────────────────────────────
  Pos  ID NuBBED              Similaridade  SMILES
────────────────────────────────────────────────────────────────────────────────
    1.  CNP0173930.1            0.998412  CC(=O)Oc1ccccc1C(=O)O
    2.  CNP0001456.1            0.981033  CC(=O)Oc1ccccc1C(=O)OC
    3.  CNP0002789.2            0.974201  OC(=O)c1ccccc1O
   ...
════════════════════════════════════════════════════════════════════════════════
```

---

## Detalhes técnicos

- **Modelo**: `ibm-research/MoLFormer-XL-both-10pct` via HuggingFace Transformers
- **Pooling**: Mean pooling sobre `last_hidden_state`, mascarado por `attention_mask`
- **Normalização**: L2 antes de inserir e antes de consultar
- **Métrica**: Distância de cosseno (`"hnsw:space": "cosine"` no ChromaDB)
- **Score**: `similaridade = 1 - distância_cosseno` (0 = oposto, 1 = idêntico)
- **Idempotência**: IDs já presentes no banco são ignorados automaticamente

---

## Estrutura de arquivos gerados

```
BuscadorMolecular/
├── banco_vetorial/               ← ChromaDB persistente (gerado pelo popular_banco.py)
│   └── chroma.sqlite3
├── nubbedb-05-2026.sdf           ← Arquivo de entrada da NuBBED
├── vetorizador_molformer.py
├── preparador_smiles.py
├── gerenciador_banco_vetorial.py
├── carregador_nubbed.py
├── popular_banco.py
├── buscar_similares.py
├── interface_grafica.py
├── requirements.txt
└── README.md
```
