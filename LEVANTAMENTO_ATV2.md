# Atividade 2 — Levantamento de meta-características para detecção de discurso de ódio

**IN1097 — Tópicos Avançados em Agentes Inteligentes 2 · Parte 1 (apresentação 17/09)**

## O problema e por que as famílias clássicas não bastam

Seleção de pipeline para um novo corpus de discurso de ódio: 50 datasets (14 fontes,
5 idiomas) × 10 pipelines que contrastam **representação esparsa-lexical** (TF-IDF +
LR/SVM/NB/SVD) com **densa-neural** (MiniLM, hateBERT, mBERT congelados). Meta-alvo:
ranking por macro-F1.

A Atividade 1 deixou três lacunas que as famílias clássicas não cobrem:

1. **O idioma inverte o paradigma vencedor**, mas X só enxergava isso por flags binárias
   (`is_english`) e por um léxico de profanidade inglês que virou detector de idioma
   disfarçado. Falta uma medida *contínua* de quão bem o vocabulário do corpus casa com
   o encoder — que é o que realmente importa para o pipeline denso.
2. **Explicitude do ódio.** Corpora de ódio implícito (IHC) e explícito (Davidson)
   diferem exatamente no que o TF-IDF consegue explorar: palavras-gatilho. Medidas
   estatísticas e de teoria da informação sobre atributos numéricos não veem isso.
3. **Dureza depende da representação.** Complexidade medida em um único espaço não diz
   *qual* representação separa melhor as classes — a pergunta central do painel.

Famílias clássicas em dados textuais também têm problema operacional: medidas
estatísticas (assimetria, curtose, correlação) sobre 20 mil colunas TF-IDF esparsas ou
384 dimensões de embedding não têm interpretação e custam caro.

## Quadro de meta-características

Legenda de família (taxonomia da aula + recortes não cobertos): **S** simples,
**E** estatística, **TI** teoria da informação, **MB** baseada em modelo, **L**
landmarker, **C** complexidade, **D** específica do domínio. Custo: complexidade por
dataset com *n* ≤ 3000 documentos.

### Extraídas

| # | Meta-característica (colunas) | Família | Referência | Definição operacional | O que captura | Por que no meu problema | Custo / implementação |
|---|---|---|---|---|---|---|---|
| 1 | Fertilidade do tokenizador, proporção de palavras continuadas, taxa de `[UNK]` — hateBERT e mBERT (`tok_*`) | D | Rust et al. (2021) | Média de *wordpieces* por palavra; fração de palavras quebradas em ≥2 peças; fração com `[UNK]`; diferença de fertilidade hateBERT − mBERT | Casamento entre o vocabulário do corpus e o do encoder pré-treinado | É o mecanismo por trás da queda dos encoders fora do inglês — contínuo, em vez de `is_english`; também reage a gíria, ofuscação ("f*ck") e hashtags | O(tokens); tokenizadores HF. Implementação própria |
| 2 | Taxa de documentos e tokens com termo HurtLex, gap entre classes, entropia de categorias (`hl_*`) | D | Bassignana et al. (2018); Wiegand et al. (2019) | Casamento com o léxico HurtLex no idioma do corpus; `hl_class_gap` = máx − mín por classe da fração de documentos com termo ofensivo; entropia das 17 categorias do léxico | Explicitude do abuso e o viés de palavras-gatilho do dataset | Corrige o léxico inglês-only da Atividade 1 (HurtLex tem en/es/pt/fr/ar); ódio implícito × explícito determina se TF-IDF basta | O(tokens); léxico público (CC BY-SA). Implementação própria |
| 3 | Divergência de Jensen–Shannon entre classes e classe→corpus, Jaccard de vocabulário entre classes, inclinação de Zipf (`ch_*`) | D / TI | Pinto & Rosso (2007); Madrid et al. (2019) | JS médio entre distribuições de unigramas por classe; JS médio de cada classe para o corpus (SVB); Jaccard médio dos vocabulários de classe (RH); inclinação log-freq × log-rank | Dureza do corpus textual: quanto as classes compartilham vocabulário | Classes de ódio frequentemente usam o mesmo vocabulário do não-ódio (contra-discurso, citação); vê o que `mean_mutual_info` sobre 1000 termos dilui | O(tokens). Implementação própria |
| 4 | Landmarkers de atalho lexical: *stump* e árvore de profundidade 3 sobre bag-of-words (`lm_stump_bow_f1`, `lm_tree3_bow_f1`) | L | Pfahringer et al. (2000); Wiegand et al. (2019) | Macro-F1 em 5-fold de árvore de 1 e 3 níveis sobre presença binária de termos | Se poucas palavras resolvem a tarefa (viés de tópico/palavra-chave) | Datasets com viés de palavra-chave favorecem TF-IDF; landmarker é barato e fora do painel | 5 × ajuste de árvore; scikit-learn |
| 5 | Landmarkers relativos: centróide mais próximo em TF-IDF, MiniLM, hateBERT e diferença MiniLM − TF-IDF (`lm_nc_*`) | L | Fürnkranz & Petrak (2001) | Macro-F1 em 5-fold do `NearestCentroid` em cada representação | Qual representação separa melhor com o classificador mais simples possível | É a pergunta do painel (esparso × denso) feita com um aprendiz que não está no painel | Barato; scikit-learn |
| 6 | F1 de Fisher, N1, N2, LSC, T2–T4, densidade, coeficiente de agrupamento e *hubs* da rede ε-NN (`cx_*`) | C | Ho & Basu (2002); Lorena et al. (2019); Garcia et al. (2015) | Definições de Lorena et al. (2019) sobre MiniLM → padronização → PCA(50); rede ε-NN (ε = 0,15) sem arestas entre classes | Sobreposição, fronteira e estrutura de vizinhança das classes | Complementa `knn1_disagreement`/`silhouette` com a família completa; medidas de rede são a direção sugerida no enunciado | O(n²) (MST, rede). `pymfe`/`problexity` existem; reimplementado para controlar o espaço |
| 7 | kDN médio em MiniLM, hateBERT, mBERT e TF-IDF; fração de instâncias duras; gap TF-IDF − MiniLM (`ih_*`) | C (dureza de instância agregada) | Smith et al. (2014) | kDN(x) = fração dos 5 vizinhos (cosseno) com rótulo diferente; média no dataset | Dureza em nível de instância, **por representação** | A diferença de dureza entre espaços prevê qual paradigma vence — o que uma medida de espaço único não diz | O(n²) por espaço. Implementação própria |
| 8 | Tipologia da classe minoritária: safe, borderline, rare, outlier (MiniLM e TF-IDF) (`mt_*`) | C / desbalanceamento | Napierała & Stefanowski (2016) | Para cada exemplo da classe minoritária, nº de vizinhos da mesma classe entre 5: 4–5 safe, 2–3 borderline, 1 rare, 0 outlier | Como a classe minoritária (tipicamente ódio) está distribuída no espaço | Ódio implícito tende a ser borderline/rare; `minority_pct` só mede quantidade, não dificuldade | Reaproveita os vizinhos de #7 |
| 9 | Inclinação da curva de aprendizado (MiniLM, TF-IDF) e gap médio entre elas (`lc_*`) | L (curva como objeto) | Leite & Brazdil (2005) | Macro-F1 do `NearestCentroid` com 25/50/100/200 exemplos (3 repetições), inclinação em log₂(n) | Quão rápido cada representação aprende | O cap de 3000 instâncias favorece quem aprende com pouco dado (limitação vi da Atividade 1) | 24 ajustes rápidos; implementação própria |

### Levantadas e deixadas de fora

| Meta-característica | Referência | Motivo |
|---|---|---|
| Dataset Cartography (confiança e variabilidade ao longo do fine-tuning; fração de exemplos ambíguos/difíceis) | Swayamdipta et al. (2020) | **Custo**: exige fine-tuning de um encoder por dataset registrando a dinâmica de treino (50 × épocas em GPU), mais caro que construir P inteira. Painel não tem modelo fine-tunado |
| Meta-características aprendidas por codificador de conjunto (Dataset2Vec) | Jomaa et al. (2021) | **Incompatibilidade**: 50 meta-instâncias não bastam para treinar o codificador; pensado para dados tabulares |
| Viés de autor / sobreposição de usuários entre treino e teste | Arango et al. (2019) | **Dados indisponíveis**: IDs de usuário ausentes na maioria das fontes |
| Dureza de instância completa (IH com pool de classificadores, pyhard) | Smith et al. (2014); Paiva et al. (2022) | **Circularidade e custo**: o pool de classificadores reproduz o próprio painel, vazando o meta-alvo em X. Mantido só o kDN |
| Escores de toxicidade da Perspective API (média, dispersão por classe) | Lees et al. (2022) | **Indisponibilidade/termos**: API externa com cota e termos que restringem envio de corpora de terceiros; majoritariamente inglês |
| Concordância entre anotadores (α de Krippendorff) | Poletto et al. (2021) | **Dados indisponíveis**: só algumas fontes publicam anotações individuais |
| Similaridade entre categorias de datasets distintos | Fortuna et al. (2021) | **Incompatibilidade com LOO**: é propriedade de pares de datasets, depende de quais estão no treino; reservado para trabalho futuro |
| Famílias estatísticas clássicas (assimetria, curtose, correlação entre atributos) | Alcobaça et al. (2020) — pymfe | **Sem interpretação** sobre TF-IDF esparso / embeddings; alto custo sobre 20 mil colunas |

## Referências

- Alcobaça, E. et al. (2020). MFE: Towards reproducible meta-feature extraction. *JMLR* 21(111).
- Arango, A., Pérez, J., Poblete, B. (2019). Hate speech detection is not as easy as you may think. *SIGIR*.
- Bassignana, E., Basile, V., Patti, V. (2018). Hurtlex: A multilingual lexicon of words to hurt. *CLiC-it*.
- Fortuna, P., Soler-Company, J., Wanner, L. (2021). How well do hate speech, toxic, abusive and offensive language classification models generalize across datasets? *Information Processing & Management* 58(3).
- Fürnkranz, J., Petrak, J. (2001). An evaluation of landmarking variants. *ECML/PKDD Workshop on Integrating Aspects of Data Mining, Decision Support and Meta-Learning*.
- Garcia, L. P. F., de Carvalho, A. C. P. L. F., Lorena, A. C. (2015). Effect of label noise in the complexity of classification problems. *Neurocomputing* 160.
- Ho, T. K., Basu, M. (2002). Complexity measures of supervised classification problems. *IEEE TPAMI* 24(3).
- Jomaa, H. S., Schmidt-Thieme, L., Grabocka, J. (2021). Dataset2Vec: Learning dataset meta-features. *Data Mining and Knowledge Discovery* 35.
- Lees, A. et al. (2022). A new generation of Perspective API: Efficient multilingual character-level transformers. *KDD*.
- Leite, R., Brazdil, P. (2005). Predicting relative performance of classifiers from samples. *ICML*.
- Lorena, A. C., Garcia, L. P. F., Lehmann, J., Souto, M. C. P., Ho, T. K. (2019). How complex is your classification problem? A survey on measuring classification complexity. *ACM Computing Surveys* 52(5).
- Madrid, J. G., Escalante, H. J., Morales, E. F. (2019). Meta-learning of text classification tasks. *CIARP*.
- Napierała, K., Stefanowski, J. (2016). Types of minority class examples and their influence on learning classifiers from imbalanced data. *Journal of Intelligent Information Systems* 46(3).
- Paiva, P. Y. A. et al. (2022). Relating instance hardness to classification performance in a dataset: a visual approach. *Machine Learning* 111.
- Pfahringer, B., Bensusan, H., Giraud-Carrier, C. (2000). Meta-learning by landmarking various learning algorithms. *ICML*.
- Pinto, D., Rosso, P. (2007). On the relative hardness of clustering corpora. *TSD*.
- Poletto, F. et al. (2021). Resources and benchmark corpora for hate speech detection: a systematic review. *Language Resources and Evaluation* 55.
- Rust, P., Pfeiffer, J., Vulić, I., Ruder, S., Gurevych, I. (2021). How good is your tokenizer? On the monolingual performance of multilingual language models. *ACL*.
- Smith, M. R., Martinez, T., Giraud-Carrier, C. (2014). An instance level analysis of data complexity. *Machine Learning* 95(2).
- Swayamdipta, S. et al. (2020). Dataset cartography: Mapping and diagnosing datasets with training dynamics. *EMNLP*.
- Wiegand, M., Ruppenhofer, J., Kleinbauer, T. (2019). Detection of abusive language: the problem of biased datasets. *NAACL-HLT*.
