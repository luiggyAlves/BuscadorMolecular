# BuscadorMolecular

Ferramenta de busca por similaridade molecular sobre a base **NuBBED** (Nuclei of Bioassays, Biosynthesis and Biodiversity). Dado um SMILES query, retorna as moléculas mais similares ranqueadas por score de cosseno.

Suporta três métodos de vetorização intercambiáveis, selecionados pelo usuário antes da inicialização do banco.

---

## Métodos de vetorização

| Método | Modelo | Dimensão | Fonte |
|---|---|---|---|
| **MolFormer-XL** | `ibm-research/MoLFormer-XL-both-10pct` | 768 | IBM via HuggingFace |
| **ChemBERTa-2** | `DeepChem/ChemBERTa-77M-MLM` | 768 | DeepChem via HuggingFace |
| **Fingerprints (RDKit)** | Morgan ECFP4 | 2048 | RDKit (local) |

Todos os vetores são L2-normalizados. A busca usa distância de cosseno via ChromaDB.

---

## Arquitetura

```
interface_grafica.py          ← Interface web Streamlit
logger_buscas.py              ← Registro de buscas em SQLite
exportar_logs.py              ← CLI para exportar/limpar logs
gerenciador_banco_vetorial.py ← ChromaDB: inserção e consulta vetorial
vetorizador_molformer.py      ← Embeddings MolFormer-XL
vetorizador_chemberta.py      ← Embeddings ChemBERTa-2
vetorizador_fingerprint.py    ← Fingerprints Morgan (RDKit)
preparador_smiles.py          ← Validação e canonicalização de SMILES
carregador_nubbed.py          ← Leitura do SDF da NuBBED
carregador_coconut.py         ← Leitura de CSV da COCONUT
popular_banco.py              ← CLI para popular o banco via terminal
buscar_similares.py           ← CLI para buscar via terminal
```

---

## Pré-requisitos

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) ou Anaconda
- Arquivo SDF da NuBBED na pasta do projeto (`nubbedb-05-2026.sdf`)
- GPU NVIDIA com CUDA 12.1 (opcional — CPU funciona, mas é mais lento)

---

## Instalação

### 1. Criar o ambiente conda

```powershell
conda env create -f environment.yml
```

### 2. Ativar o ambiente

```powershell
conda activate buscadormolecular
```

### 3. Verificar instalação

```powershell
python -c "import rdkit, torch, transformers, chromadb, streamlit; print('OK')"
```

> **Windows — RDKit bloqueado por política de segurança**
>
> Se ocorrer `ImportError: DLL load failed while importing rdmolfiles`, execute:
> ```powershell
> Get-ChildItem -Path "C:\Users\<usuario>\miniconda3\envs\buscadormolecular" -Recurse -Filter "*.pyd" | Unblock-File
> ```
> Isso remove a marca "baixado da internet" dos arquivos do ambiente.

### Atualizar após mudanças no environment.yml

```powershell
conda env update -f environment.yml --prune
```

---

## Executar a interface

```powershell
conda activate buscadormolecular
streamlit run interface_grafica.py
```

O navegador abre em `http://localhost:8501`.

---

## Fluxo de uso da interface

```
1. Selecionar método de vetorização (MolFormer-XL / ChemBERTa-2 / Fingerprints)
        ↓
2. Clicar em "▶ Iniciar vetorização"
        ↓
3. Aguardar a população do banco (barra de progresso)
        ↓
4. Interface de busca liberada — método travado para a sessão
        ↓
5. Digitar SMILES ou clicar em molécula de exemplo
        ↓
6. Definir K (número de resultados) e clicar em "Buscar"
        ↓
7. Resultados em cards com estrutura, ID NuBBED e % de similaridade
```

**Comportamento por sessão:**
- O banco vetorial é **esvaziado automaticamente** a cada nova sessão
- O método de vetorização **não pode ser alterado** após o banco ser populado
- A vetorização **não inicia automaticamente** — requer clique do usuário

---

## Interface de busca

- **Moléculas de exemplo:** Aspirina, Cafeína, Quercetina, Resveratrol — clique para preencher o campo
- **Slider K:** define quantos resultados exibir (1–50)
- **Cards de resultado:** imagem estrutural, ID NuBBED, score de similaridade em %, SMILES

---

## Logs de busca

Todas as buscas são registradas automaticamente em `logs/buscas.db` (SQLite). Os logs **não são expostos na interface** — são acessíveis apenas via terminal.

Cada registro contém:

```json
{
  "timestamp": "2026-06-02T14:30:00",
  "metodo_busca": "MolFormer-XL",
  "query": "CC(=O)Oc1ccccc1C(=O)O",
  "retornos": [
    { "smile_retorno": "...", "percentual_relevancia": 0.97 }
  ]
}
```

São armazenados os **20 primeiros retornos** de cada busca, independente do K exibido na tela.

### Comandos de log

```powershell
# Exportar todos os logs (JSON no terminal)
python exportar_logs.py

# Salvar em arquivo JSON
python exportar_logs.py --saida logs.json

# Exportar como CSV
python exportar_logs.py --formato csv --saida logs.csv

# Filtrar por método
python exportar_logs.py --metodo "MolFormer-XL"
python exportar_logs.py --metodo "ChemBERTa-2"
python exportar_logs.py --metodo "Fingerprints (RDKit)"

# Filtrar por data
python exportar_logs.py --desde 2026-06-01

# Últimas N buscas
python exportar_logs.py --ultimas 50

# Limpar todos os logs
python exportar_logs.py --limpar
```

> A pasta `logs/` é criada automaticamente na primeira busca. Pode ser apagada sem problema — será recriada.

---

## Estrutura do projeto

```
BuscadorMolecular/
├── interface_grafica.py
├── logger_buscas.py
├── exportar_logs.py
├── gerenciador_banco_vetorial.py
├── vetorizador_molformer.py
├── vetorizador_chemberta.py
├── vetorizador_fingerprint.py
├── preparador_smiles.py
├── carregador_nubbed.py
├── carregador_coconut.py
├── popular_banco.py
├── buscar_similares.py
├── environment.yml
├── requirements.txt
├── nubbedb-05-2026.sdf        ← não versionado
├── banco_vetorial/            ← não versionado (gerado em runtime)
└── logs/                      ← não versionado (gerado em runtime)
```

---

## .gitignore recomendado

```
banco_vetorial/
logs/
*.sdf
__pycache__/
*.pyc
.streamlit/secrets.toml
```
