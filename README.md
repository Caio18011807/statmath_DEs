# StatVision OCR: Statistical Distribution & Property Image Analyzer

**StatVision OCR** is an offline Python software suite that inspects images of **statistical distribution plots (PDF/PMF/CDF)**, **textbook formulas**, or **statistical problem statements** using a multi-backend **OCR + Visual Curve Geometry** pipeline and returns a complete mathematical & statistical profile:

1. **Distribution Name & Family** (Continuous or Discrete, Support, Canonical Parameters)
2. **Density / Mass Function (PDF / PMF)** & **Cumulative Distribution Function (CDF)** in LaTeX
3. **Location & Dispersion**: Mean ($\mathbb{E}[X]$), Variance ($\operatorname{Var}(X)$), Standard Deviation, **Median** (exact formula, Wilson–Hilferty / incomplete beta approximations, or existence conditions), and **Mode**
4. **Higher-Order Shape & Kurtosis**: Skewness ($\gamma_1$), **Excess Kurtosis** ($\gamma_2 = \beta_2 - 3$), and **Raw Kurtosis** ($\beta_2 = \mathbb{E}[(X-\mu)^4]/\sigma^4$)
5. **General Moments Formula**:
   - $k$-th Raw Moment ($\mu'_k = \mathbb{E}[X^k]$)
   - $k$-th Central Moment ($\mu_k = \mathbb{E}[(X-\mu)^k]$)
   - Moment Generating Function ($M_X(t) = \mathbb{E}[e^{tX}]$)
   - Characteristic Function ($\varphi_X(t) = \mathbb{E}[e^{itX}]$)
   - Fisher Information Matrix ($\mathcal{I}(\theta)$) & Differential/Discrete Shannon Entropy
6. **Every Parameterization of the Distribution**:
   - Standard mathematical, Scale vs. Rate ($\beta = 1/\lambda$), Precision ($\tau = 1/\sigma^2$), Mean-Dispersion (GLM), Natural Exponential Family canonical form, and Location-Scale / Shifted forms
7. **Statistical Context & Independence / i.i.d. Details**:
   - Extracts explicit notation from the image: $X_1, \dots, X_n \stackrel{\text{i.i.d.}}{\sim} D(\theta)$, independence ($X \perp\!\!\!\perp Y$), joint factorization $f_{X,Y}(x,y)=f_X(x)f_Y(y)$, conditioning ($P(A \mid B)$), and numeric parameter values
   - Provides theoretical i.i.d. sample closure laws, Basu's/Cochran's/Lukacs'/Raikov's theorems, complete sufficient statistics, and Maximum Likelihood Estimators (MLE / UMVUE)
8. **Image & Curve Geometry Properties**:
   - Image dimensions, aspect ratio, luminance, contrast, foreground ink density, empirical curve skewness/kurtosis, symmetry index, rectangular flatness, and monotonicity

---

## Quick Start

### 1. Generate Sample Distribution Images & Run via CLI

```bash
cd /Users/caio/.gemini/antigravity/scratch/stat_vision_ocr

# Generate 6 synthetic sample distribution plots/cards in ./samples/
python3 cli.py --generate-samples

# Analyze a sample image and print the full Markdown + LaTeX report
python3 cli.py --image samples/normal_iid_sample.png

# Output as structured JSON
python3 cli.py --image samples/poisson_discrete_events.png --json
```

### 2. Launch the Interactive Web UI Dashboard

Works out-of-the-box with zero external dependencies using the built-in KaTeX + SVG HTTP server, or with Streamlit if installed:

```bash
# Zero-dependency Web UI (opens on http://127.0.0.1:8501)
python3 app.py

# Or with Streamlit (if installed via requirements.txt)
streamlit run app.py
```

### 3. Run the Automated Unit Test Suite

```bash
python3 test_stat_vision.py
```
