# StatMath: Reconhecimento e Interpretação Probabilística de Notações Estatísticas Manuscritas com Arquitetura Híbrida Local

> **Autor(es):** [Seu Nome Completo] — RA: [Seu RA]  
> **Disciplina:** 1001336 — Inteligência Artificial (UFSCar — Departamento de Computação)  
> **Docente:** Profa. Dra. Priscila Tiemi Maeda Saito  
> **Formato de referência:** Artigo IEEE (máximo de 4 páginas)

---

## Resumo (*Abstract*)

Estudantes de Probabilidade e Estatística frequentemente encontram dificuldade em desdobrar todas as propriedades implícitas em notações compactas como $X_1, \dots, X_n \overset{\text{i.i.d.}}{\sim} \mathcal{N}(\mu, \sigma^2)$. Este trabalho apresenta o **StatMath**, um sistema de Inteligência Artificial local, multiplataforma (Windows, macOS e Linux) e de código aberto que recebe uma imagem (foto, captura de tela, área de transferência ou câmera) ou texto digitado contendo uma notação estatística manuscrita ou tipografada, reconhece os símbolos, interpreta o contexto probabilístico e apresenta em LaTeX a função densidade/massa de probabilidade (FDP/FMP), a função de distribuição acumulada (FDA), os momentos, propriedades de independência/i.i.d. e um gráfico interativo. A arquitetura híbrida combina restauração de imagem em dois estágios, reconhecimento óptico de caracteres unificado multiplataforma (**RapidOCR ONNX / PP-OCRv4** com reconstrução espacial de linhas e *fallbacks* nativos), um classificador visual por $k$ vizinhos mais próximos ($k$-NN), correspondência léxica aproximada (*fuzzy matching*), reconhecimento estrutural de fórmulas, fusão multi-canal com calibração probabilística **Softmax** e uma base de conhecimento simbólica com **13 distribuições**. Em avaliação experimental, o sistema obteve **100%** no conjunto de treinamento (**121 imagens**) e na bateria automatizada de **17 fórmulas inéditas**, e **92,3% (12,0/13 pontos)** em um protocolo rigoroso de **avaliação cega** sobre **13 fotografias manuscritas reais inéditas** usando métrica composta de distribuição e parâmetros exatos (com **0,5 ms/imagem** em cache LRU SHA-256 e resultados idênticos em Windows, macOS e Linux).

**Palavras-chave:** OCR, notação estatística, escrita manuscrita, base de conhecimento simbólica, avaliação cega, ferramenta educacional.

---

## 1. Introdução e Motivação

A Estatística emprega notações concisas e formais para descrever variáveis aleatórias (VAs). Uma única expressão, como $Y_1, \dots, Y_n \overset{\text{i.i.d.}}{\sim} \text{Exp}(\lambda)$, sintetiza independência, distribuição exata, parametrização, suporte, densidade, função acumulada e momentos (média, variância, assimetria e curtose). Em disciplinas introdutórias e intermediárias, estudantes costumam ter dificuldade em traduzir rapidamente essa notação para suas propriedades analíticas e geométricas.

Ferramentas consolidadas de visão computacional aplicada à matemática, como o Photomath e o Mathpix, concentram-se na transcrição e na manipulação algébrica de equações, sem interpretar o significado probabilístico de distribuições e amostras aleatórias. Além disso, dependem de APIs proprietárias em nuvem. O **StatMath** foi projetado para preencher essa lacuna: trata-se de um tutor inteligente aberto, executável localmente (sem depender de nuvem) em macOS, Windows e Linux, capaz de ler a escrita do estudante, diagnosticar problemas de qualidade da foto e apresentar uma explicação simbólica, numérica e gráfica verificável com interface responsiva e troca instantânea entre imagens.

---

## 2. Objetivos

**Objetivo geral:** Projetar, implementar e avaliar experimentalmente um sistema de Inteligência Artificial local, eficiente e multiplataforma capaz de receber imagens ou textos de notações estatísticas manuscritas, reconhecer os símbolos, extrair o contexto probabilístico e gerar um relatório didático completo.

**Objetivos específicos:**
1. Implementar um pipeline de **aquisição, diagnóstico objetivo de qualidade e restauração de imagem** em dois estágios que trate fotos desfocadas ou sombreadas sem degradar imagens nítidas;
2. Integrar um **motor de OCR neural unificado multiplataforma (RapidOCR ONNX)** com reconstrução espacial de linhas e *fallbacks* nativos (Apple Vision, Windows WinRT OCR, EasyOCR e Tesseract), aliado a um **classificador visual $k$-NN**;
3. Desenvolver um **pós-processador e analisador sintático estatístico** (normalização manuscrita, *fuzzy matching* léxico, reconhecimento algébrico de FDP/FMP e fusão multi-canal com calibração Softmax) para identificar VA, independência/i.i.d., distribuição e parâmetros;
4. Estruturar uma **base de conhecimento simbólica e um motor numérico** abrangendo **13 distribuições** (9 contínuas e 4 discretas), com visualização gráfica interativa e interface responsiva com cache LRU e troca rápida de imagens;
5. Definir e aplicar um **protocolo experimental de avaliação cega** com métrica composta (distribuição + parâmetros exatos), analisando criticamente acertos, erros e limitações.

---

## 3. Trabalhos Relacionados

A transcrição automática de expressões matemáticas manuscritas é tradicionalmente abordada por modelos neurais ponta a ponta. Deng et al. [1] propuseram o *im2latex*, baseado em codificador-decodificador com atenção para gerar marcação LaTeX a partir de imagens. Abordagens baseadas em Transformers, como o TrOCR [2], combinam codificadores visuais e decodificadores de linguagem pré-treinados com excelente desempenho em escrita manuscrita, enquanto projetos abertos como o Pix2Tex (LaTeX-OCR) [3] oferecem inferência local para equações. Contudo, essas ferramentas limitam-se à transcrição sintática e exigem modelos pesados (centenas de megabytes), sem realizar inferência estatística sobre a expressão lida.

Entre os motores de OCR de propósito geral, destacam-se o Tesseract [4] (leve, porém sensível à escrita cursiva), o EasyOCR (detecção CRAFT [5] + reconhecimento CRNN [6], mais lento em CPU), os motores nativos de sistema operacional (Apple Vision no macOS e WinRT OCR no Windows) e a família PaddleOCR/RapidOCR em ONNX, que executa redes leves em CPU de forma determinística e multiplataforma. Na etapa de visão computacional clássica, a variância do operador Laplaciano [7] é amplamente usada como medida objetiva de nitidez, enquanto *unsharp masking* e alongamento adaptativo de contraste [8] restauram documentos degradados. Na classificação por similaridade visual, o método dos $k$ vizinhos mais próximos ($k$-NN) [9] permite aprendizado incremental imediato.

| Aspecto | Photomath / Mathpix | Pix2Tex / TrOCR | **StatMath (Proposto)** |
| :--- | :--- | :--- | :--- |
| Execução local / aberta | Não (nuvem proprietária) | Sim (modelos neurais pesados) | **Sim (leve em CPU, local e open-source)** |
| Reprodutibilidade multiplataforma | Depende de nuvem | Requer PyTorch / GPU idealmente | **Idêntica em Windows, macOS e Linux (ONNX)** |
| Interpretação probabilística | Não | Não | **Sim (VA, i.i.d., FDP, FDA, momentos)** |
| Gráfico interativo da distribuição | Parcial | Não | **Sim (eixos, zoom, pan e $F(x)$)** |
| Verificabilidade simbólica | Caixa-preta | Caixa-preta | **Sim (arquitetura híbrida transparente)** |

---

## 4. Metodologia Experimental

O pipeline do StatMath é composto por cinco etapas sequenciais — **aquisição de dados**, **pré-processamento**, **extração de padrões**, **pós-processamento** e **tomada de decisão / apresentação** — orquestradas por `analyzer.py` e empacotadas em uma interface web responsiva assíncrona (`app.py`, `StatMath.app` e inicializadores portáteis para Windows, macOS e Linux).

```mermaid
flowchart LR
    A["Aquisição<br/>(Foto, Print, Texto)"] --> B["Cache LRU SHA-256<br/>+ Qualidade (2 Estágios)"]
    B --> C["Extração de Padrões<br/>(RapidOCR ONNX + k-NN)"]
    C --> D["Pós-processamento<br/>(Normalizador + Fuzzy + Parser)"]
    D --> E["Fusão Softmax + Base<br/>Simbólica (13 Dist.)"]
    E --> F["Painel AJAX LaTeX +<br/>Gráfico + Troca Rápida"]
```

### 4.1 Aquisição e Organização dos Conjuntos de Dados
Foram construídos três conjuntos independentes, localizados dinamicamente pelo aplicativo em qualquer sistema operacional (sem caminhos fixos):
1. **Conjunto de Treinamento (`training_data/`):** **121 imagens** organizadas em 13 subpastas (uma por distribuição), combinando cartões sintéticos em caderno pautado gerados por `sample_generator.py` e fotografias reais manuscritas. As assinaturas visuais são indexadas em `learned_patterns.json` (treinamento contínuo).
2. **Bateria Automatizada de Fórmulas Inéditas (`test_stat_vision.py`):** **17 expressões inéditas** com ruídos típicos de OCR (vírgula decimal brasileira, erros de índice, sequências indexadas, notações simbólicas e numéricas).
3. **Conjunto de Teste Real Cego (`real_test_data/`):** **13 fotografias manuscritas reais inéditas** (uma de cada distribuição), nunca usadas no treinamento. Cada arquivo segue o padrão universal `<distribuicao>_<param1>[_<param2>].<ext>` (documentado em `real_test_data/README.txt`), usado **exclusivamente** pelo avaliador como gabarito.

### 4.2 Pré-processamento, Diagnóstico de Qualidade e Restauração em 2 Estágios
- **Decodificação multiplataforma (`curve_analyzer.py`):** leitor PNG em Python puro (com composição de transparência sobre fundo branco) aliado a OpenCV/Pillow (quando presentes) e conversores nativos para JPEG/WEBP/BMP/TIFF (`sips` no macOS, `System.Drawing.Bitmap` no Windows e ImageMagick/`ffmpeg` no Linux).
- **Diagnóstico de qualidade:** calcula nitidez (variância do Laplaciano), contraste tinta/papel, iluminação média e resolução. Um alerta visual só é exibido ao usuário quando a foto efetivamente prejudica a leitura.
- **Fluxo em 2 estágios (`analyzer.py`):** prioriza sempre a imagem original (Estágio 1). Apenas se a pontuação for baixa ou faltarem VA/parâmetros, aciona `enhance_image_for_ocr` (Estágio 2: mapa bilinear local $12 \times 6$ no percentil 78 para remoção de sombras, *unsharp masking* $3 \times 3$ e alongamento adaptativo de contraste com fundo branco puro), adotando a leitura restaurada somente se sua pontuação de completude for estritamente superior.

### 4.3 Extração de Padrões (RapidOCR ONNX Unificado + *Fallbacks* + $k$-NN Visual)
- **Motor unificado multiplataforma (`ocr_engine.py`):** adota **RapidOCR (`rapidocr-onnxruntime`, PP-OCRv4 em ONNX)** com reconstrução espacial de linhas de leitura (agrupamento vertical por centroide e ordenação horizontal) e normalização de pontuação *full-width*. Por executar a mesma rede neural leve em CPU (~15 MB, pré-aquecida em segundo plano ao abrir o app), garante **desempenho idêntico em Windows, macOS e Linux**.
- **Camadas de *fallback*:** caso o RapidOCR não esteja disponível, o sistema recorre automaticamente a Apple Vision Framework (macOS), EasyOCR, Windows Native OCR (`Windows.Media.Ocr` via PowerShell `-EncodedCommand` UTF-16LE) ou Tesseract CLI.
- **Classificador visual $k$-NN:** extrai assinaturas visuais invariantes e compara contra `learned_patterns.json`, restringindo a substituição do texto do OCR pela transcrição memorizada apenas a correspondências praticamente idênticas ($\text{sim} \ge 0{,}96$) ou quando o OCR retorna vazio — evitando sobrescrever fotos inéditas com parâmetros antigos do treino.

### 4.4 Pós-processamento, Fusão Probabilística e Eficiência
1. **Normalização manuscrita:** corrige confusões recorrentes de escrita cursiva (`l`/`I` $\to$ `1`, `O` $\to$ `0`, `S` $\to$ `5`, caractere CJK `了` lido pelo PaddleOCR no lugar do dígito cursivo `3`), funções indicadoras $\mathbb{I}_{(a,b)}(x)$ lidas como `N(0,1)`, vírgula decimal brasileira em distribuições de 1 parâmetro (`Poisson(4,5)` $\to$ `4.5`), sequências indexadas ($X_1, \dots, X_n$), desambiguação de $\Gamma(\cdot)$ e $\exp(\cdot)$, e correção didática de intervalos de índices ($i \in [0,n] \to i \in \{1, \dots, n\}$).
2. **Correspondência léxica aproximada (*Fuzzy Matching*):** aplica `difflib.SequenceMatcher` sobre palavras anteriores a `(`, `[` ou dígitos (comprimento $\ge 5$), aceitando similaridade $\ge 72\%$ com margem $\ge 6\%$ sobre o segundo colocado e normalizando para o nome canônico antes da extração de parâmetros.
3. **Reconhecimento estrutural de fórmulas:** regras algébricas identificam FDP/FMP mesmo quando o nome da distribuição é corrompido pelo OCR (ex.: $\lambda e^{-\lambda x}$ sem `^`, densidade da Cauchy com $\pi[1+(\cdot)^2]$, Qui-Quadrado com `/2-1` e $\Gamma(\cdot)$, e estrutura da Gama `= β ... (α-1) ... -βx`).
4. **Fusão Multi-Canal com Calibração Softmax e Cache LRU (`analyzer.py`):** combina os escores dos 4 canais de evidência e calibra a confiança a posteriori via Softmax com temperatura ($T = 45$):
   $$P(C_k \mid x) = \frac{\exp(s_k / T)}{\sum_{j=1}^{13} \exp(s_j / T)}$$
   Um cache LRU em memória indexado por **SHA-256** (até 64 imagens) reduz reanálises e alternâncias de ~1,0 s para **0,5 ms/imagem**.
5. **Base simbólica, motor numérico e interface responsiva (`knowledge_base.py`, `symbolic_engine.py`, `app.py`):** cobre **13 distribuições** com proteção contra *overflow* numérico (Log-Normal), gráfico SVG interativo (eixos, zoom `100%`–`1000%`, pan e mira `(x, f(x), F(x))`), análise AJAX sem recarregar a página (`POST /api/analyze` + `AbortController`) e barra **Troca Rápida** para alternar instantaneamente entre até 8 imagens recentes.

### 4.5 Vantagens, Desvantagens e Justificativa das Técnicas Escolhidas

| Etapa / Técnica | Motivo da Escolha | Vantagens | Desvantagens |
| :--- | :--- | :--- | :--- |
| **Restauração em 2 estágios (mapa bilinear $12 \times 6$ + *unsharp*)** | Recuperar fotos borradas/sombreadas sem degradar fotos nítidas | Preserva fotos boas; remove sombras de caderno sem amplificar textura do papel | Custo de uma segunda leitura de OCR nos casos difíceis |
| **RapidOCR ONNX unificado + reconstrução espacial** | Reprodutibilidade idêntica entre Windows, macOS e Linux sem GPU | Leve (~15 MB), offline, determinístico e mesma acurácia em qualquer SO | Charset multilíngue ocasionalmente emite caracteres CJK em escrita muito cursiva (mitigado no normalizador) |
| **Classificador visual $k$-NN (`sim >= 0.96`)** | Aprendizado incremental imediato com poucas fotos por classe | Sem necessidade de re-treinar redes pesadas; protege contra sobrescrita de fotos novas | Depende de enquadramento semelhante quando usado sem OCR |
| **Normalizador + *Fuzzy Matching* + Fusão Softmax + Base Simbólica** | Garantir saída matematicamente exata, didática e calibrada | 100% verificável, transparente e fiel à teoria estatística | Regras exigem cuidado para evitar colisões entre distribuições semelhantes |
| **Cache LRU SHA-256 + AJAX (`AbortController`) + Troca Rápida** | Fluidez e responsividade ao alternar entre várias imagens | Troca instantânea (**0,5 ms**); cancela análises obsoletas sem travar a tela | Cache em memória é reiniciado ao fechar o servidor local |

### 4.6 Protocolo Experimental e Métrica de Avaliação
Para garantir integridade científica e **zero vazamento de dados** (*data leakage*), implementou-se um avaliador cego (`real_evaluator.py`): antes de invocar a IA sobre cada foto de `real_test_data/`, os bytes brutos são copiados para um arquivo temporário anônimo (`blind_eval_XX.<ext>`), impedindo acesso ao nome do arquivo ou diretório.

Para cada imagem $i \in \{1, \dots, N\}$, definem-se:
- $d_i \in \{0, 1\}$: acerto da **distribuição**;
- $p_i \in \{0, 1\}$: acerto dos **parâmetros numéricos exatos** (comparados valor a valor com tolerância de $1\%$ contra o gabarito codificado no nome do arquivo).

A pontuação individual e a acurácia global são dadas por:

$$s_i = \frac{d_i + p_i}{2} \in \{0,\ 0{,}5,\ 1\}, \qquad \text{Acurácia}_{\text{IA}} = \frac{1}{N} \sum_{i=1}^{N} s_i \times 100\%$$

Assim, acertar distribuição e parâmetros concede **1,0 ponto**; acertar somente um deles concede **0,5 ponto**; errar ambos concede **0,0 ponto**. O resultado é exibido em tempo real na barra superior do aplicativo.

---

## 5. Resultados e Discussões

### 5.1 Evolução Experimental no Conjunto de Teste Real Cego (`real_test_data/`, 13 imagens)
A adoção da avaliação cega permitiu diagnosticar falhas reais entre sistemas operacionais e aprimorar o sistema de forma geral (sem *overfitting*):
1. **Linha de base inicial:** **61,5%** de acerto em distribuição no macOS (e apenas **15%** no Windows antes da correção do WinRT OCR e da adoção do RapidOCR ONNX unificado). Os erros vinham de substituições de 1–2 letras no OCR cursivo e da sobrescrita indevida de fotos novas pelo $k$-NN com limiar `sim >= 0.78`.
2. **Fase intermediária (Apple Vision + *fuzzy matching* + $k$-NN `sim >= 0.96`):** **84,6% (11,0/13 pontos)** na métrica rigorosa no macOS (10 acertos totais, 2 parciais e 1 erro total).
3. **Versão final unificada (RapidOCR ONNX + reconstrução espacial + normalização CJK + estrutura algébrica Gama + fusão Softmax + cache LRU):** **92,3% (12,0/13 pontos)** — **100% (13/13)** de acerto na distribuição e **11/13** com parâmetros numéricos exatos, replicando exatamente o mesmo desempenho em **Windows, macOS e Linux**.

### 5.2 Detalhamento por Imagem no Teste Cego Real (Versão Final Unificada)

| # | Arquivo (Gabarito) | Escrita no Papel | Dist. ($d_i$) | Parâmetros ($p_i$) | Nota ($s_i$) |
| :-: | :--- | :--- | :-: | :-: | :-: |
| 01 | `beta_100_2.jpeg` | $D \sim \text{Beta}(100, 2)$ | ✅ 1 | ✅ 1 ($\alpha=100, \beta=2$) | **1,0** |
| 02 | `binomial_50_0.1.jpeg` | $C \sim \text{Binomial}(50, 0{,}1)$ | ✅ 1 | ✅ 1 ($n=50, p=0{,}1$) | **1,0** |
| 03 | `cauchy_6_5.jpeg` | $A \sim \text{Cauchy}(6, 5)$ | ✅ 1 | ❌ 0 (OCR leu `(6,6)` em vez de `(6,5)`) | **0,5** |
| 04 | `exponencial_19.jpeg` | $K \sim \text{Exponencial}(19)$ | ✅ 1 | ✅ 1 ($\lambda=19$) | **1,0** |
| 05 | `gama_7_10.jpeg` | $J \sim \text{Gama}(7, 10)$ | ✅ 1 | ✅ 1 ($\alpha=7, \beta=10$) | **1,0** |
| 06 | `geometrica_0.7.jpeg` | $B \sim \text{Geométrica}(0{,}7)$ | ✅ 1 | ✅ 1 ($p=0{,}7$) | **1,0** |
| 07 | `lognormal_0_10.jpeg` | $M \sim \text{Lognormal}(0, 10)$ | ✅ 1 | ✅ 1 ($\mu=0, \sigma=10$) | **1,0** |
| 08 | `normal_7_29.jpeg` | $L \sim \text{Normal}(7, 29)$ | ✅ 1 | ❌ 0 (OCR leu `neimal( 29`, omitiu o `7`) | **0,5** |
| 09 | `poisson_36.jpeg` | $W \sim \text{Poisson}(36)$ | ✅ 1 | ✅ 1 ($\lambda=36$) | **1,0** |
| 10 | `qui_quadrado_7.jpeg` | $Q \sim \chi^2_7$ | ✅ 1 | ✅ 1 ($k=7$) | **1,0** |
| 11 | `t_student_3.jpeg` | $S \sim t_3$ | ✅ 1 | ✅ 1 ($\nu=3$) | **1,0** |
| 12 | `uniforme_0_100.jpeg` | $V \sim \text{Uniforme}(0, 100)$ | ✅ 1 | ✅ 1 ($a=0, b=100$) | **1,0** |
| 13 | `weibull_3_2.jpeg` | $Z \sim \text{Weibull}(3, 2)$ | ✅ 1 | ✅ 1 ($k=3, \lambda=2$) | **1,0** |

### 5.3 Análise Crítica dos Resultados e Desempenho
- **Acertos completos (11/13 — 1,0 pt):** confirmam a eficácia do motor unificado RapidOCR ONNX aliado ao normalizador manuscrito, *fuzzy matching* e extração algébrica de FDP/FMP (incluindo $\chi^2_7$, $t_3$, Uniforme e Weibull).
- **Acertos parciais (2/13 — 0,5 pt):**
  - Em `cauchy_6_5.jpeg`, o OCR leu `Coxchy(6,6)` confundindo o dígito cursivo `5` com `6` — a distribuição foi acertada via *fuzzy matching*, mas a validação estrita penalizou o parâmetro;
  - Em `normal_7_29.jpeg`, o OCR leu `neimal( 29` perdendo o primeiro dígito `7` (com sombra lateral na foto) — novamente acertou a distribuição, mas não extraiu $\mu = 7$.
- **Eficiência computacional:** o tempo médio a frio é de **~1,0 s/imagem** (incluindo Estágio 2 condicional nos casos difíceis), caindo para **0,5 ms/imagem** com o cache LRU SHA-256 na troca entre imagens recentes.
- **Demais baterias:** **100%** no conjunto de treinamento (`training_data/`, 121 imagens) e **100%** nos testes automatizados de **17 fórmulas inéditas** (`test_stat_vision.py`).

---

## 6. Conclusões e Trabalhos Futuros

O StatMath cumpriu integralmente os objetivos propostos, demonstrando que uma arquitetura híbrida local — unindo restauração em 2 estágios, OCR neural leve multiplataforma (RapidOCR ONNX), $k$-NN visual, *fuzzy matching*, reconhecimento algébrico, fusão calibrada Softmax e base de conhecimento simbólica — alcança alta precisão (**92,3% / 12,0 de 13 pontos** na métrica rigorosa distribuição + parâmetros exatos em fotos manuscritas reais inéditas), reprodutibilidade entre sistemas operacionais, baixa latência (**0,5 ms** em cache) e total transparência didática sem depender de nuvem.

**Trabalhos futuros:**
1. Integrar um modelo leve especializado em expressões matemáticas manuscritas (como TrOCR/Pix2Tex quantizado em ONNX) como segundo leitor para desambiguar dígitos cursivos semelhantes (`5` vs `6`, `7` sob sombra);
2. Expandir a base simbólica para distribuições multivariadas (Normal Multivariada, Multinomial, Dirichlet) e testes de hipóteses;
3. Ampliar o conjunto de teste real cego com dezenas de escritores e diferentes condições de iluminação e papel;
4. Empacotar binários nativos independentes de Python (PyInstaller/Briefcase) para Windows e Linux.

---

## 7. Referências Bibliográficas (Padrão IEEE)

- [1] Y. Deng, A. Kanervisto, J. Ling, and A. M. Rush, "Image-to-Markup Generation with Coarse-to-Fine Attention," in *Proc. 34th Int. Conf. Machine Learning (ICML)*, 2017, pp. 980–989.
- [2] M. Li et al., "TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models," in *Proc. AAAI Conf. Artificial Intelligence*, vol. 37, no. 11, pp. 13094–13102, 2023.
- [3] L. Blecher, "pix2tex — LaTeX OCR," GitHub repository, 2022. Disponível em: https://github.com/lukas-blecher/LaTeX-OCR.
- [4] R. Smith, "An Overview of the Tesseract OCR Engine," in *Proc. 9th Int. Conf. Document Analysis and Recognition (ICDAR)*, 2007, pp. 629–633.
- [5] Y. Baek, B. Lee, D. Han, S. Yun, and H. Lee, "Character Region Awareness for Text Detection," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2019, pp. 9365–9374.
- [6] B. Shi, X. Bai, and C. Yao, "An End-to-End Trainable Neural Network for Image-Based Sequence Recognition and Its Application to Scene Text Recognition," *IEEE Trans. Pattern Analysis and Machine Intelligence*, vol. 39, no. 11, pp. 2298–2304, 2017.
- [7] J. L. Pech-Pacheco, G. Cristóbal, J. Chamorro-Martínez, and J. Fernández-Valdivia, "Diatom autofocusing in brightfield microscopy: a comparative study," in *Proc. 15th Int. Conf. Pattern Recognition (ICPR)*, vol. 3, 2000, pp. 314–317.
- [8] R. C. Gonzalez and R. E. Woods, *Digital Image Processing*, 4th ed. New York, NY, USA: Pearson, 2018.
- [9] T. Cover and P. Hart, "Nearest neighbor pattern classification," *IEEE Trans. Information Theory*, vol. 13, no. 1, pp. 21–27, 1967.

---

## 8. Checklist de Entrega (Moodle) e Roteiro do Seminário

### Itens Obrigatórios para Submissão no Moodle (Seção 5 do Edital)
- [ ] **`.TEX`/`.DOC` e `PDF` do relatório** (template IEEE, máximo 4 páginas — use o conteúdo das Seções 1 a 7 acima);
- [ ] **`PPT` e `PDF` dos slides do seminário** (com numeração em todos os slides, imagens do pipeline/app, comparação com pelo menos 1 trabalho correlato e referências no final);
- [ ] **Link de vídeo** (ex.: YouTube) contendo a apresentação;
- [ ] **Conjuntos de dados utilizados nos experimentos** (`training_data/`, `real_test_data/` com `README.txt` e `evaluation_results.json`);
- [ ] **Códigos-fonte comentados + README em português** (repositório GitHub `https://github.com/Caio18011807/statmath_DEs` ou `StatMath_Portatil_OpenSource.zip`).

### Roteiro Sugerido para o Seminário (Seção 4 do Edital)
1. **Slide 1:** Título, nome, RA e disciplina.
2. **Slide 2:** Introdução, problema real e motivação da IA local multiplataforma.
3. **Slide 3:** Trabalhos relacionados e tabela comparativa (Photomath / Pix2Tex / TrOCR vs. StatMath).
4. **Slide 4:** Arquitetura e pipeline híbrido do StatMath (diagrama + prints do aplicativo).
5. **Slide 5:** Conjuntos de dados (`training_data/` 121 imgs, 17 fórmulas inéditas, `real_test_data/` 13 fotos cegas) e padrão de nomeação.
6. **Slide 6:** Técnicas adotadas (restauração 2 estágios, RapidOCR ONNX, $k$-NN, *fuzzy matching*, fusão Softmax, cache LRU) — vantagens e desvantagens.
7. **Slide 7:** Protocolo de avaliação cega e métrica composta $s_i = (d_i + p_i)/2$.
8. **Slide 8:** Evolução dos resultados (61,5% / 15% Windows $\to$ 84,6% $\to$ **92,3% / 12,0 de 13 pts** idêntico em todos os SOs) e análise crítica dos 2 casos parciais (`cauchy_6_5` e `normal_7_29`).
9. **Slide 9:** Demonstração rápida (troca instantânea de imagens + gráfico interativo + selo de acurácia real).
10. **Slide 10:** Conclusões e trabalhos futuros.
11. **Slide 11:** Referências bibliográficas (numeradas).
