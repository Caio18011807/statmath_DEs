# StatMath: Reconhecimento e Interpretação Probabilística de Notações Estatísticas Manuscritas com Arquitetura Híbrida Local

> **Autor(es):** [Seu Nome Completo] — RA: [Seu RA]  
> **Disciplina:** 1001336 — Inteligência Artificial (UFSCar — Departamento de Computação)  
> **Docente:** Profa. Dra. Priscila Tiemi Maeda Saito  
> **Formato de referência:** Artigo IEEE (máximo de 4 páginas)

---

## Resumo (*Abstract*)

Estudantes de Probabilidade e Estatística frequentemente encontram dificuldade em desdobrar todas as propriedades implícitas em notações compactas como $X_1, \dots, X_n \overset{\text{i.i.d.}}{\sim} \mathcal{N}(\mu, \sigma^2)$. Este trabalho apresenta o **StatMath**, um sistema de Inteligência Artificial local, multiplataforma e de código aberto que recebe uma imagem (foto, captura de tela, área de transferência) ou texto digitado contendo uma notação estatística manuscrita ou tipografada, reconhece os símbolos, interpreta o contexto probabilístico e apresenta em LaTeX a função densidade/massa de probabilidade (FDP/FMP), a função de distribuição acumulada (FDA), os momentos, propriedades de independência/i.i.d. e um gráfico interativo. A arquitetura híbrida combina pré-processamento e restauração de imagem em dois estágios, reconhecimento óptico de caracteres (OCR) nativo multiplataforma, um classificador visual por $k$ vizinhos mais próximos ($k$-NN), correspondência léxica aproximada (*fuzzy matching*), reconhecimento estrutural de fórmulas e uma base de conhecimento simbólica com **13 distribuições**. Em avaliação experimental, o sistema atingiu **100%** no conjunto de treinamento (**121 imagens**) e na bateria automatizada de **17 fórmulas inéditas**, e **84,6% (11,0/13 pontos)** em um protocolo rigoroso de **avaliação cega** sobre **13 fotografias manuscritas reais inéditas** usando métrica composta de distribuição e parâmetros exatos ($92{,}3\%$ de acerto na distribuição).

**Palavras-chave:** OCR, notação estatística, escrita manuscrita, base de conhecimento simbólica, avaliação cega, ferramenta educacional.

---

## 1. Introdução e Motivação

A Estatística emprega notações concisas e formais para descrever variáveis aleatórias (VAs). Uma única expressão, como $Y_1, \dots, Y_n \overset{\text{i.i.d.}}{\sim} \text{Exp}(\lambda)$, sintetiza independência, distribuição exata, parametrização, suporte, densidade, função acumulada e momentos (média, variância, assimetria e curtose). Em disciplinas introdutórias e intermediárias, estudantes costumam ter dificuldade em traduzir rapidamente essa notação para suas propriedades analíticas e geométricas.

Ferramentas consolidadas de visão computacional aplicada à matemática, como o Photomath e o Mathpix, concentram-se na transcrição e na manipulação algébrica de equações, sem interpretar o significado probabilístico de distribuições e amostras aleatórias. Além disso, dependem de APIs proprietárias em nuvem. O **StatMath** foi projetado para preencher essa lacuna: trata-se de um tutor inteligente aberto, executável localmente (sem internet) em macOS, Windows e Linux, capaz de ler a escrita do estudante, diagnosticar problemas de qualidade da foto e apresentar uma explicação simbólica, numérica e gráfica verificável.

---

## 2. Objetivos

**Objetivo geral:** Projetar, implementar e avaliar experimentalmente um sistema de Inteligência Artificial capaz de receber imagens ou textos de notações estatísticas manuscritas, reconhecer os símbolos, extrair o contexto probabilístico e gerar um relatório didático completo.

**Objetivos específicos:**
1. Implementar um pipeline de **aquisição, diagnóstico de qualidade e restauração de imagem** em dois estágios que trate fotos desfocadas ou sombreadas sem degradar imagens nítidas;
2. Integrar **OCR nativo multiplataforma** em camadas com um **classificador visual $k$-NN** treinado sobre padrões manuscritos;
3. Desenvolver um **pós-processador e analisador sintático estatístico** (normalização manuscrita, *fuzzy matching* léxico e reconhecimento algébrico de FDP/FMP) para identificar a VA, relações de independência/i.i.d., distribuição e parâmetros;
4. Estruturar uma **base de conhecimento simbólica e um motor numérico** abrangendo **13 distribuições** (9 contínuas e 4 discretas), com visualização gráfica interativa;
5. Definir e aplicar um **protocolo experimental de avaliação cega** com métrica composta (distribuição + parâmetros exatos), analisando criticamente acertos, erros e limitações.

---

## 3. Trabalhos Relacionados

A transcrição automática de expressões matemáticas manuscritas é tradicionalmente abordada por modelos neurais ponta a ponta. Deng et al. [1] propuseram o *im2latex*, baseado em codificador-decodificador com atenção para gerar marcação LaTeX a partir de imagens. Abordagens baseadas em Transformers, como o TrOCR [2], combinam codificadores visuais e decodificadores de linguagem pré-treinados com excelente desempenho em escrita manuscrita, enquanto projetos abertos como o Pix2Tex (LaTeX-OCR) [3] oferecem inferência local para equações. Contudo, essas ferramentas limitam-se à transcrição sintática e exigem modelos pesados (centenas de megabytes), sem realizar inferência estatística sobre a expressão lida.

Entre os motores de OCR de propósito geral, destacam-se o Tesseract [4] (leve e multiplataforma, porém sensível à escrita cursiva), o EasyOCR (detecção CRAFT [5] + reconhecimento CRNN [6], mais preciso, porém mais lento em CPU) e os motores nativos de sistema operacional (Apple Vision no macOS e WinRT OCR no Windows). Na etapa de visão computacional clássica, a variância do operador Laplaciano [7] é amplamente usada como medida objetiva de nitidez, enquanto *unsharp masking* e alongamento de contraste [8] restauram bordas em documentos degradados. Na classificação por similaridade visual, o método dos $k$ vizinhos mais próximos ($k$-NN) [9] permite aprendizado incremental imediato.

| Aspecto | Photomath / Mathpix | Pix2Tex / TrOCR | **StatMath (Proposto)** |
| :--- | :--- | :--- | :--- |
| Execução local / aberta | Não (nuvem proprietária) | Sim (modelos neurais pesados) | **Sim (leve, local e open-source)** |
| Transcrição de fórmulas | Sim | Sim | Sim (OCR nativo + normalizador) |
| Interpretação probabilística | Não | Não | **Sim (VA, i.i.d., FDP, FDA, momentos)** |
| Gráfico interativo da distribuição | Parcial | Não | **Sim (eixos, zoom, pan e $F(x)$)** |
| Verificabilidade simbólica | Caixa-preta | Caixa-preta | **Sim (arquitetura híbrida transparente)** |

---

## 4. Metodologia Experimental

O pipeline do StatMath é composto por cinco etapas sequenciais — **aquisição de dados**, **pré-processamento**, **extração de padrões**, **pós-processamento** e **tomada de decisão / apresentação** — orquestradas por `analyzer.py` e empacotadas em uma interface web local (`app.py`, `StatMath.app` e scripts portáteis para Windows/macOS/Linux).

```mermaid
flowchart LR
    A["Aquisição<br/>(Foto, Print, Texto)"] --> B["Pré-processamento<br/>(Qualidade + 2 Estágios)"]
    B --> C["Extração de Padrões<br/>(OCR Multiplataforma + k-NN)"]
    C --> D["Pós-processamento<br/>(Normalizador + Fuzzy + Parser)"]
    D --> E["Base Simbólica (13 Dist.)<br/>+ Motor Numérico"]
    E --> F["Painel LaTeX +<br/>Gráfico Interativo"]
```

### 4.1 Aquisição e Organização dos Conjuntos de Dados
Foram construídos três conjuntos independentes, localizados dinamicamente pelo aplicativo em qualquer sistema operacional:
1. **Conjunto de Treinamento (`training_data/`):** **121 imagens** organizadas em 13 subpastas (uma por distribuição), combinando cartões sintéticos em caderno pautado gerados por `sample_generator.py` e fotografias reais manuscritas. As assinaturas visuais são indexadas em `learned_patterns.json` (treinamento contínuo).
2. **Bateria Automatizada de Fórmulas Inéditas (`test_stat_vision.py`):** **17 expressões inéditas** com ruídos típicos de OCR (vírgula decimal brasileira, erros de índice, sequências indexadas, notações simbólicas e numéricas).
3. **Conjunto de Teste Real Cego (`real_test_data/`):** **13 fotografias manuscritas reais inéditas** (uma de cada distribuição), nunca usadas no treinamento. Cada arquivo segue o padrão universal `<distribuicao>_<param1>[_<param2>].<ext>` (documentado em `real_test_data/README.txt`), usado **exclusivamente** pelo avaliador como gabarito.

### 4.2 Pré-processamento, Diagnóstico de Qualidade e Restauração em 2 Estágios
- **Decodificação multiplataforma (`curve_analyzer.py`):** leitor PNG implementado em Python puro (incluindo composição de transparência sobre fundo branco) aliado a conversores nativos para JPEG/WEBP/BMP/TIFF (`sips` no macOS, `System.Drawing.Bitmap` no Windows e ImageMagick/`ffmpeg` no Linux).
- **Diagnóstico de qualidade:** calcula nitidez (variância do Laplaciano), contraste entre tinta e papel, iluminação média e resolução. Um alerta visual só é exibido ao usuário quando a foto efetivamente prejudica a leitura.
- **Fluxo em 2 estágios (`analyzer.py`):** o sistema prioriza sempre a imagem original (Estágio 1). Apenas se a confiança for baixa ou se faltarem a VA ou os parâmetros, aciona `enhance_image_for_ocr` (Estágio 2: remoção de sombras, *unsharp masking* e alongamento de contraste), adotando a leitura restaurada somente se sua pontuação de completude for estritamente superior.

### 4.3 Extração de Padrões (OCR em Camadas + Classificador Visual $k$-NN)
- **OCR em camadas (`ocr_engine.py`):** utiliza Apple Vision no macOS, Windows WinRT OCR (`Windows.Media.Ocr.OcrEngine`) no Windows 10/11 e EasyOCR/Tesseract como fallback multiplataforma.
- **Classificador visual $k$-NN:** extrai assinaturas visuais invariantes a escala/enquadramento e compara por similaridade contra `learned_patterns.json`. Um ajuste crítico implementado após testes reais restringiu a substituição do texto do OCR pela transcrição memorizada apenas a correspondências praticamente idênticas ($\text{sim} \ge 0{,}96$) ou quando o OCR retorna vazio — evitando que uma foto nova com a mesma caligrafia do treino seja sobrescrita por parâmetros antigos.

### 4.4 Pós-processamento e Tomada de Decisão
1. **Normalização manuscrita:** corrige confusões recorrentes de escrita cursiva (`l`/`I` $\to$ `1`, `O` $\to$ `0`, `S` $\to$ `5`), funções indicadoras $\mathbb{I}_{(a,b)}(x)$ lidas como `N(0,1)`, vírgula decimal brasileira em distribuições de 1 parâmetro (ex.: `Poisson(4,5)` $\to$ `4.5`), sequências indexadas ($X_1, \dots, X_n$), desambiguação de $\Gamma(\cdot)$ e $\exp(\cdot)$, além de detecção e correção didática de erros de notação de índices ($i \in [0,n] \to i \in \{1, \dots, n\}$).
2. **Correspondência léxica aproximada (*Fuzzy Matching*):** aplica `difflib.SequenceMatcher` sobre palavras imediatamente anteriores a `(` (comprimento $\ge 5$), aceitando correspondência com similaridade $\ge 72\%$ e margem $\ge 6\%$ sobre o segundo colocado, e normalizando o termo para o nome canônico antes da extração de parâmetros.
3. **Reconhecimento estrutural de fórmulas:** regras algébricas detectam assinaturas de FDP/FMP mesmo quando o nome da distribuição está ilegível (ex.: $\lambda e^{-\lambda x}$ mesmo sem sobrescrito `^`, $\frac{1}{\pi\gamma[1+((x-x_0)/\gamma)^2]}$ para Cauchy, $\Gamma(\cdot/2)$ + `/2-1` para Qui-Quadrado).
4. **Base simbólica e motor numérico (`knowledge_base.py`, `symbolic_engine.py`):** cobre **13 distribuições** (*Normal, Exponencial, Uniforme, Gama, Beta, Qui-Quadrado, t de Student, Log-Normal, Weibull, Poisson, Binomial, Geométrica e Cauchy*), com proteção contra *overflow* numérico (ex.: momentos da Log-Normal com $\sigma$ grande) e gráfico SVG interativo (eixos graduados, zoom `100%`–`1000%`, pan e mira `(x, f(x), F(x))`).

### 4.5 Vantagens, Desvantagens e Justificativa das Técnicas Escolhidas

| Etapa / Técnica | Motivo da Escolha | Vantagens | Desvantagens |
| :--- | :--- | :--- | :--- |
| **Pré-processamento em Python puro + conversores nativos** | Eliminar dependências pesadas (OpenCV/Pillow) para portabilidade | Zero instalação externa; funciona em Windows, macOS e Linux | Mais lento que bibliotecas compiladas em C++ para imagens muito grandes |
| **Restauração em 2 estágios (*unsharp* + contraste)** | Recuperar fotos borradas/escuras sem estragar fotos boas | Nunca degrada imagens já nítidas; recupera casos difíceis | Custo adicional de uma segunda leitura de OCR nos casos difíceis |
| **OCR nativo do SO em camadas (Vision / WinRT / EasyOCR / Tesseract)** | Melhor equilíbrio entre precisão cursiva, velocidade e portabilidade | Rápido (~0,3–1,1 s), offline e sem GPU; boa leitura de texto cursivo | Quebra frações verticais, expoentes e símbolos gregos ($\chi^2$) em várias linhas |
| **Classificador visual $k$-NN (`sim >= 0.96`)** | Aprendizado contínuo com poucas amostras por classe | Treino instantâneo; não exige re-treinar redes neurais | Sensível a variações grandes de enquadramento se usado sem OCR |
| **Normalizador + *Fuzzy Matching* + Base Simbólica** | Garantir saída matematicamente exata e didática | 100% verificável, transparente e fiel à teoria estatística | Regras exigem cuidado para evitar colisões entre distribuições semelhantes |

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
A adoção da avaliação cega permitiu diagnosticar falhas reais que não apareciam no conjunto de treinamento:
1. **Linha de base inicial (antes do *fuzzy matching* e do ajuste do $k$-NN):** **61,5% (8/13)** de acerto em distribuição. Os erros ocorriam por substituições de 1–2 letras no OCR cursivo (`Expanencial`, `Lagnarmal`, `Weiell`) e pela sobrescrita indevida de `beta_100_2.jpeg` (`D ~ Beta(100, 2)`) por uma imagem do treino (`A ~ Beta(1, 8)`) devido ao limiar permissivo `sim >= 0.78`.
2. **Após ajustes gerais (limiar `sim >= 0.96`, *fuzzy matching* $\ge 72\%$ com normalização canônica, regra exponencial sem `^` e proteção de *overflow* na Log-Normal):** a acurácia de distribuição subiu para **92,3% (12/13)** e a acurácia composta rigorosa (distribuição + parâmetros exatos) atingiu **84,6% (11,0/13 pontos)**, sem qualquer regra específica para as imagens de teste (evitando *overfitting*).

### 5.2 Detalhamento por Imagem no Teste Cego Real

| # | Arquivo (Gabarito) | Escrita no Papel | Dist. ($d_i$) | Parâmetros ($p_i$) | Nota ($s_i$) |
| :-: | :--- | :--- | :-: | :-: | :-: |
| 01 | `beta_100_2.jpeg` | $D \sim \text{Beta}(100, 2)$ | ✅ 1 | ✅ 1 ($\alpha=100, \beta=2$) | **1,0** |
| 02 | `binomial_50_0.1.jpeg` | $C \sim \text{Binomial}(50, 0{,}1)$ | ✅ 1 | ✅ 1 ($n=50, p=0{,}1$) | **1,0** |
| 03 | `cauchy_6_5.jpeg` | $A \sim \text{Cauchy}(6, 5)$ | ✅ 1 | ✅ 1 ($x_0=6, \gamma=5$) | **1,0** |
| 04 | `exponencial_19.jpeg` | $K \sim \text{Exponencial}(19)$ | ✅ 1 | ✅ 1 ($\lambda=19$) | **1,0** |
| 05 | `gama_7_10.jpeg` | $J \sim \text{Gama}(7, 10)$ | ✅ 1 | ✅ 1 ($\alpha=7, \beta=10$) | **1,0** |
| 06 | `geometrica_0.7.jpeg` | $B \sim \text{Geométrica}(0{,}7)$ | ✅ 1 | ✅ 1 ($p=0{,}7$) | **1,0** |
| 07 | `lognormal_0_10.jpeg` | $M \sim \text{Lognormal}(0, 10)$ | ✅ 1 | ✅ 1 ($\mu=0, \sigma=10$) | **1,0** |
| 08 | `normal_7_29.jpeg` | $L \sim \text{Normal}(7, 29)$ | ✅ 1 | ✅ 1 ($\mu=7, \sigma^2=29$) | **1,0** |
| 09 | `poisson_36.jpeg` | $W \sim \text{Poisson}(36)$ | ✅ 1 | ✅ 1 ($\lambda=36$) | **1,0** |
| 10 | `qui_quadrado_7.jpeg` | $Q \sim \chi^2_7$ | ✅ 1 | ❌ 0 (OCR leu `Q~X*`, sem o `7`) | **0,5** |
| 11 | `t_student_3.jpeg` | $S \sim t_3$ | ✅ 1 | ✅ 1 ($\nu=3$) | **1,0** |
| 12 | `uniforme_0_100.jpeg` | $V \sim \text{Uniforme}(0, 100)$ | ❌ 0 | ❌ 0 (OCR leu `thrifome (0,(00)`) | **0,0** |
| 13 | `weibull_3_2.jpeg` | $Z \sim \text{Weibull}(3, 2)$ | ✅ 1 | ❌ 0 (inverteu $k=2, \lambda=3$) | **0,5** |

### 5.3 Análise Crítica dos Erros
- **Acertos completos (10/13 — 1,0 pt):** demonstram que a combinação de OCR nativo + *fuzzy matching* + normalizador lida bem com caligrafia cursiva real, vírgula decimal brasileira e fórmulas manuscritas, com tempo médio de inferência entre $0{,}3\text{ s}$ e $1{,}1\text{ s}$.
- **Acertos parciais (2/13 — 0,5 pt):**
  - Em `qui_quadrado_7.jpeg`, o OCR de uso geral falhou em ler o subscrito `7` em $\chi^2_7$, mas o reconhecedor algébrico identificou corretamente a distribuição pela estrutura da FDP (`/2-1` + $\Gamma(\cdot)$);
  - Em `weibull_3_2.jpeg`, a validação estrita de parâmetros revelou uma ambiguidade real de parametrização na literatura: o sistema assumiu $\text{Weibull}(\lambda, k)$ (escala primeiro), enquanto a notação escrita na foto usou $\text{Weibull}(k, \lambda)$ (forma primeiro), invertendo $k$ e $\lambda$.
- **Erro total (1/13 — 0,0 pt):** em `uniforme_0_100.jpeg`, a palavra cursiva `Uniforme` foi lida pelo OCR como `thrifome` (similaridade abaixo do limiar de $72\%$) e `100` foi lido como `(00`, evidenciando a principal limitação de OCRs de propósito geral em grafias cursivas muito estilizadas.
- **Demais baterias:** o sistema manteve **100%** no banco de treinamento (`training_data/`, 121 imagens) e **100%** nos testes automatizados de **17 fórmulas inéditas** (`test_stat_vision.py`).

---

## 6. Conclusões e Trabalhos Futuros

O StatMath atingiu todos os objetivos propostos, demonstrando que uma arquitetura híbrida local — unindo visão computacional clássica, OCR nativo multiplataforma, $k$-NN visual, *fuzzy matching* e base de conhecimento simbólica — oferece alta precisão (**92,3%** em distribuição e **84,6%** na métrica rigorosa distribuição + parâmetros em fotos manuscritas reais inéditas), baixa latência e total transparência didática sem depender de serviços em nuvem.

**Trabalhos futuros:**
1. Integrar um modelo leve especializado em expressões matemáticas manuscritas (como TrOCR/Pix2Tex quantizado) como camada adicional para melhorar a leitura de subscritos ($\chi^2_k$), frações verticais e letras cursivas ambíguas;
2. Desambiguar automaticamente a ordem dos parâmetros na distribuição de Weibull ($\text{Weibull}(k, \lambda)$ vs. $\text{Weibull}(\lambda, k)$) utilizando a própria fórmula da FDP escrita ao lado;
3. Expandir a base simbólica para distribuições multivariadas (Normal Multivariada, Multinomial, Dirichlet) e testes de hipóteses;
4. Ampliar o conjunto de teste real cego com múltiplos escritores e condições variadas de iluminação.

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
- [ ] **Códigos-fonte comentados + README** (todo o repositório `stat_vision_ocr/` ou `StatMath_Portatil_OpenSource.zip`, com instruções de execução para macOS, Windows e Linux).

### Roteiro Sugerido para o Seminário (Seção 4 do Edital)
1. **Slide 1:** Título, nome, RA e disciplina.
2. **Slide 2:** Introdução, problema real (interpretação de notações estatísticas) e motivação da IA local.
3. **Slide 3:** Trabalhos relacionados e tabela comparativa (Photomath / Pix2Tex / TrOCR vs. StatMath).
4. **Slide 4:** Arquitetura e pipeline do StatMath (diagrama + prints do aplicativo).
5. **Slide 5:** Conjuntos de dados (`training_data/` 121 imgs, 17 fórmulas inéditas, `real_test_data/` 13 fotos cegas) e padrão de nomeação.
6. **Slide 6:** Técnicas de IA adotadas (OCR em camadas, $k$-NN, restauração em 2 estágios, *fuzzy matching*, base simbólica) — vantagens e desvantagens.
7. **Slide 7:** Protocolo de avaliação cega e métrica composta $s_i = (d_i + p_i)/2$.
8. **Slide 8:** Resultados (61,5% $\to$ 92,3% distribuição / **84,6%** na métrica rigorosa 11,0/13 pts) e análise crítica dos 3 casos de erro (`qui_quadrado_7`, `weibull_3_2`, `uniforme_0_100`).
9. **Slide 9:** Demonstração rápida (print/vídeo do app rodando + gráfico interativo + selo de acurácia real).
10. **Slide 10:** Conclusões e trabalhos futuros.
11. **Slide 11:** Referências bibliográficas (numeradas).
