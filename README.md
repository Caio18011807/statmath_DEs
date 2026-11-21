# StatMath — Reconhecimento e Interpretação Probabilística de Distribuições Estatísticas

O **StatMath** (`StatVision OCR`) é um aplicativo local, **100% gratuito, de código aberto e multiplataforma (Windows, macOS e Linux)** que recebe uma **imagem** (foto de caderno, captura de tela, imagem colada via `Ctrl+V` / `⌘V` ou câmera) ou **texto digitado** contendo uma notação estatística — como $X_1, \dots, X_n \overset{\text{i.i.d.}}{\sim} \mathcal{N}(\mu, \sigma^2)$ — e retorna um relatório didático completo em **LaTeX** junto com um **gráfico interativo**.

---

## Principais Recursos

1. **13 Distribuições Suportadas** (9 contínuas + 4 discretas):
   - **Contínuas:** Normal, Exponencial, Uniforme, Gama, Beta, Qui-Quadrado ($\chi^2$), $t$ de Student, Log-Normal, Weibull e Cauchy.
   - **Discretas:** Poisson, Binomial (e Bernoulli) e Geométrica.
2. **Arquitetura Híbrida de IA (100% Offline):**
   - **OCR nativo multiplataforma:** Apple Vision Framework (macOS), Windows WinRT OCR (Windows 10/11) e fallback EasyOCR / Tesseract (Linux/universal).
   - **Diagnóstico de qualidade e restauração em 2 estágios:** mede nitidez (variância do Laplaciano), contraste, iluminação e resolução; prioriza sempre a imagem original e só aciona restauração (*unsharp masking* + alongamento de contraste) quando necessário.
   - **Classificador visual $k$-NN** treinado sobre `training_data/` + **normalizador manuscrito** + ***fuzzy matching* léxico** + **reconhecimento estrutural de fórmulas de FDP/FMP**.
3. **Relatório Estatístico Completo em LaTeX:**
   - FDP/FMP e FDA (acumulada) gerais e substituídas com os parâmetros lidos da imagem;
   - Variável aleatória, independência ($X \perp Y$) e amostragem i.i.d.;
   - Média, variância, desvio-padrão, mediana, moda, assimetria, excesso de curtose e curtose bruta;
   - Fórmula geral de momentos brutos, centrais, Função Geradora de Momentos (FGM) e Função Característica;
   - Todas as parametrizações alternativas da distribuição.
4. **Gráfico Interativo da Distribuição:** eixos $X$ e $Y$ graduados, zoom (`100%`–`1000%`), movimentação (pan) e leitura dinâmica de $(x,\ f(x),\ F(x))$.
5. **Avaliação Cega de Acurácia Real (`real_test_data/`):** exibe na barra superior do aplicativo a acurácia real medida em modo 100% cego (a IA nunca vê o nome do arquivo).

---

## Como Baixar e Executar (Windows, macOS e Linux)

> **Requisito mínimo:** apenas **Python 3** instalado (o aplicativo roda sem precisar instalar nenhuma biblioteca externa!).

### 1. Baixar o projeto
- **Via Git (recomendado para atualizar facilmente):**
  ```bash
  git clone https://github.com/Caio18011807/statmath_DEs.git
  cd statmath_DEs
  ```
- **Ou sem Git:** clique no botão verde **`Code → Download ZIP`** aqui no GitHub e descompacte a pasta em qualquer lugar do computador.

### 2. Abrir o aplicativo
- **Windows:** dê **dois cliques** em `Iniciar_StatMath_Windows.bat` *(ou rode `python app.py` no Prompt/PowerShell)*.
- **macOS:** dê **dois cliques** em `StatMath.app` ou `Abrir_StatMath.command` *(ou rode `python3 app.py` no Terminal)*.
- **Linux:** rode `./Abrir_StatMath_Linux.sh` *(ou `python3 app.py` no Terminal)*.

O aplicativo abrirá automaticamente em `http://127.0.0.1:8501` e localizará sozinho as pastas `training_data/` e `real_test_data/`.

### 3. Como atualizar o aplicativo no futuro
Se você baixou via Git, basta rodar dentro da pasta do projeto:
```bash
git pull origin main
```

---

## Avaliação Cega de Acurácia Real (`real_test_data/`)

O StatMath possui uma pasta de teste real ([real_test_data/](real_test_data/)) onde você pode colocar fotos inéditas de distribuições para medir a acurácia real do modelo sem vazamento de dados:
- **Padrão de nome do arquivo** (usado apenas como gabarito — a IA recebe uma cópia temporária anônima):
  ```text
  <distribuicao>_<param1>[_<param2>].<ext>
  ```
  Exemplos: `beta_100_2.jpeg`, `binomial_50_0.1.jpeg`, `normal_7_29.jpeg`, `exponencial_19.jpeg`, `cauchy_6_5.jpeg`.
- **Métrica composta por imagem ($s_i \in \{0,\ 0{,}5,\ 1\}$):**
  - **1,0 ponto:** acertou a distribuição **e** os parâmetros numéricos exatos;
  - **0,5 ponto:** acertou somente um dos dois (distribuição ou parâmetros);
  - **0,0 ponto:** errou ambos.
  $$\text{Acurácia}_{\text{IA}} = \frac{1}{N}\sum_{i=1}^{N} \frac{d_i + p_i}{2} \times 100\%$$
- **Rodar a avaliação pelo terminal:**
  ```bash
  python3 evaluate_real_test.py
  ```

---

## Comandos Úteis (Terminal / CLI)

```bash
# Abrir interface web local
python3 app.py

# Avaliar acurácia real (modo cego em real_test_data/)
python3 evaluate_real_test.py

# Avaliar ou treinar novos padrões em training_data/
python3 train_and_evaluate.py --evaluate
python3 train_and_evaluate.py --learn

# Analisar uma imagem direto pelo terminal (Markdown ou JSON)
python3 cli.py --image caminho/para/foto.jpg
python3 cli.py --image caminho/para/foto.jpg --json

# Rodar a bateria de testes automatizados
python3 test_stat_vision.py
```

---

## Estrutura Principal do Projeto

- `app.py` — Servidor web local e interface interativa (LaTeX + gráfico SVG + selo de acurácia real).
- `stat_vision/analyzer.py` — Pipeline principal em 2 estágios (imagem original + restauração condicional).
- `stat_vision/ocr_engine.py` — OCR multiplataforma, $k$-NN visual, normalizador manuscrito, *fuzzy matching* e parser estatístico.
- `stat_vision/curve_analyzer.py` — Leitor PNG em Python puro, conversores de imagem, detector de qualidade e restauração.
- `stat_vision/knowledge_base.py` — Base de conhecimento simbólica das 13 distribuições.
- `stat_vision/symbolic_engine.py` — Motor numérico de momentos e geração de curvas.
- `stat_vision/real_evaluator.py` — Avaliador cego de teste real com métrica composta.
- `training_data/` — Banco de treinamento organizado por distribuição (121 imagens).
- `real_test_data/` — Banco de teste real cego (13 fotografias inéditas + `README.txt`).
- `stat_vision_completo_ai_studio.py` — Versão consolidada em arquivo único Python.
