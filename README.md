# IN1097 — Atividade 2: meta-características específicas do domínio

Breno Cavalcanti · IN1097 Tópicos Avançados em Agentes Inteligentes 2

Recomendação de pipelines para detecção de discurso de ódio: 50 corpora (14 fontes,
5 idiomas) × 10 pipelines (TF-IDF/SVD esparsos contra MiniLM/hateBERT/mBERT densos),
meta-alvo = ranking por macro-F1. A matriz `P` e o protocolo são os da Atividade 1;
esta atividade só estende `X` com 46 meta-características novas e avalia o efeito.

## Reproduzir as análises (a partir do clone)

As saídas da extração estão versionadas em `data_hs/`, então as duas análises rodam
direto:

```bash
pip install -r requirements.txt
python hs_atv2_analysis.py   # associação (Spearman + FDR), importância por algoritmo, redundância, custo
python hs_atv2_compare.py    # LODO e LOSO com 6 versões de X, Wilcoxon, Friedman + Nemenyi, diagramas CD
```

`hs_atv2_analysis.py` precisa rodar antes de `hs_atv2_compare.py`, porque escreve
`data_hs/X_ext_filtered.csv`. As duas aceitam `--quick` para um teste rápido.

## Reextrair as meta-características

```bash
python build_datasets.py          # baixa e amostra os 50 corpora -> data_hs/datasets/*.parquet
python hs_build_metadataset.py --embed-python /caminho/python-com-cuda   # embeddings (GPU), X.csv e P.csv (Atividade 1)
python hs_build_extended_x.py     # ~7 min; 46 novas colunas + custo por grupo
```

`hs_build_extended_x.py` baixa o léxico HurtLex v1.2 (CC BY-SA 4.0) para
`data_hs/lexica/`.

## Dados e direitos autorais

Este repositório **não redistribui nenhum texto** dos corpora. Os 50 datasets vêm de
14 fontes de terceiros, cada uma com sua licença ou termos de uso (várias contêm
posts de redes sociais cuja redistribuição é restrita pelos termos das plataformas).
Por isso só são versionados **dados derivados e agregados por dataset**: meta-características
(`X*.csv`), macro-F1 dos pipelines (`P.csv`, `P_folds.csv`), metadados (fonte, idioma,
tamanho, proporção da minoria) e resultados. Nenhum desses arquivos permite reconstruir
os textos. Os corpora amostrados (`data_hs/datasets/*.parquet`), os embeddings
(`data_hs/embeddings/*.npy`), as cópias locais das fontes (`fontes_locais/`) e o léxico
HurtLex ficam fora do versionamento (`.gitignore`).

Para reconstruir os corpora, cada usuário deve obter as fontes diretamente, aceitando os
termos de cada uma:

| fonte | como `build_datasets.py` obtém |
|---|---|
| Davidson et al. (2017) | Hugging Face `tdavidson/hate_speech_offensive` |
| ETHOS | Hugging Face `iamollas/ethos` |
| HateCheck | Hugging Face `Paul/hatecheck` |
| HateXplain | Hugging Face `Hate-speech-CNERG/hatexplain` |
| HatEval (SemEval-2019 T5) | Hugging Face `valeriobasile/HatEval` |
| HateBR | Hugging Face `ruanchaves/hatebr` |
| MLMA | Hugging Face `nedjmaou/MLMA_hate_speech` |
| Measuring Hate Speech | Hugging Face `ucberkeley-dlab/measuring-hate-speech` |
| SBIC | Hugging Face `allenai/social_bias_frames` e cópia local (abaixo) |
| TweetEval | Hugging Face `cardiffnlp/tweet_eval` |
| toxic_conversations | Hugging Face `SetFit/toxic_conversations` |
| Implicit Hate Corpus (IHC) | cópia local, obtida junto aos autores |
| HateWiC | cópia local, obtida junto aos autores |

As fontes locais são lidas de `$TCC_ROOT` (padrão `./fontes_locais/`) com esta estrutura:

```
fontes_locais/
├── implicit-hate-corpus/implicit_hate_v1_stg{1,2,3}_posts.tsv
├── sbic/sbic_terms_clean_{train,dev,test}_offall.csv
└── hatewic/HateWiC_IndividualAnnos_with_def.csv
```

Uma fonte ausente é pulada com aviso; nesse caso o meta-dataset reconstruído terá menos
linhas que o versionado. As análises da Atividade 2 não precisam dos corpora: rodam só
com os CSVs de `data_hs/`.

## Conteúdo

| caminho | conteúdo |
|---|---|
| `data_hs/X.csv`, `P.csv`, `P_folds.csv` | `X_base` (50 × 36), desempenho dos 10 pipelines e folds (Atividade 1) |
| `data_hs/X_new.csv` | 50 × 46 novas (prefixos `tok_ hl_ ch_ lm_ cx_ ih_ mt_ lc_`) |
| `data_hs/X_extended.csv` | 50 × 82, base + novas |
| `data_hs/X_ext_filtered.csv` | 50 × 38, um representante por grupo de redundância |
| `data_hs/meta_feature_catalog.csv` | origem, grupo, família e referência de cada coluna |
| `data_hs/extraction_cost.csv` | segundos de extração por dataset e grupo (embeddings em cache) |
| `data_hs/datasets_manifest.csv` | fonte, idioma e tarefa de cada corpus (grupos do LOSO) |
| `results_atv2/analysis/` | análises 1 e 2: associação, importância por algoritmo, redundância, custo |
| `results_atv2/compare/{lodo,loso}/` | análise 3: Spearman, curvas de perda, AUC_loss, Wilcoxon, CD |

Versões de `X` comparadas: `base` (36), `ext` (82), `filt` (38), `sel` (36 escolhidas
por RF dentro de cada rodada, só no treino), `new` (46 novas) e `cheap` (16 novas
baratas: `tok_`, `lm_`, `lc_`).

| script | papel |
|---|---|
| `hs_domain_meta_features.py` | implementação das meta-características novas |
| `hs_build_extended_x.py` | extração de `X_new` e registro de custo |
| `hs_atv2_analysis.py` | análises 2.1.1 e 2.1.2 |
| `hs_atv2_compare.py` | análise 2.1.3 |
| `approaches.py`, `harris.py`, `evaluate.py`, `run_experiments.py`, `hs_leave_one_source_out.py` | recomendadores e protocolo da Atividade 1 |
| `build_datasets.py`, `hs_build_metadataset.py`, `hs_algorithms.py`, `hs_embed_worker.py`, `hs_meta_features.py` | construção de `P` e `X_base` (Atividade 1) |

## Origem do código

Parte do código deste repositório é reaproveitada de trabalhos anteriores meus
sobre detecção de discurso de ódio, meu TCC e minha dissertação de mestrado (em construção),
adaptada aqui para o formato de meta-dataset exigido pela atividade.

## Uso de IA generativa

Este trabalho foi desenvolvido com assistência de IA generativa, utilizada como ferramenta de apoio na escrita e revisão de código, na elaboração da documentação e no aprimoramento da redação do relatório.

Todas as decisões relacionadas ao desenvolvimento do trabalho foram tomadas por mim, incluindo o enquadramento do problema, a definição do domínio de aplicação, a seleção dos datasets e algoritmos, a escolha das métricas de avaliação, das meta-características e dos procedimentos experimentais, bem como a análise e interpretação dos resultados.

A IA também foi utilizada como suporte durante a revisão de trechos de código e de texto, enquanto a validação das implementações, dos resultados obtidos e das afirmações apresentadas permaneceu sob minha responsabilidade.

Todos os números reportados foram produzidos pelo código deste repositório e são reproduzíveis pelas etapas 3 e 4 descritas acima.
# IN1097-atv2
