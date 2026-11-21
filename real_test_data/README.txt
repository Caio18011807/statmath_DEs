================================================================================
 PASTA DE TESTE REAL (real_test_data/) — STATMATH
================================================================================

Coloque aqui as suas imagens de teste real (fotos ou prints de distribuições
escritas à mão ou digitadas) para medir a acurácia real da IA.

--------------------------------------------------------------------------------
 1. PADRÃO OBRIGATÓRIO DE NOME DO ARQUIVO (COMPATÍVEL COM WINDOWS, MAC E LINUX)
--------------------------------------------------------------------------------

    <distribuicao>_<param1>[_<param2>][__<id_opcional>].<ext>

Regras:
  - Separe o nome da distribuição e cada parâmetro com underline `_`.
  - Use ponto `.` para decimais (ex.: 0.1, 0.7) — vírgula `,` também é aceita.
  - Use sinal `-` para números negativos (ex.: normal_-2_9.jpeg).
  - Se tiver mais de uma foto com os mesmos parâmetros, adicione `__2`, `__3`
    no final (com DOIS underlines), ex.: normal_7_29__2.jpeg.
  - Extensões aceitas: .png, .jpg, .jpeg, .webp.

Ordem dos parâmetros para cada uma das 13 distribuições:
  1. Normal        : normal_<mu>_<sigma2>          Ex.: normal_7_29.jpeg
  2. Exponencial   : exponencial_<lambda>          Ex.: exponencial_19.jpeg
  3. Uniforme      : uniforme_<a>_<b>              Ex.: uniforme_0_100.jpeg
  4. Gama          : gama_<alpha>_<beta>           Ex.: gama_7_10.jpeg
  5. Beta          : beta_<alpha>_<beta>           Ex.: beta_100_2.jpeg
  6. Qui-Quadrado  : qui_quadrado_<k>              Ex.: qui_quadrado_7.jpeg
  7. t de Student  : t_student_<nu>                Ex.: t_student_3.jpeg
  8. Log-Normal    : lognormal_<mu>_<sigma>        Ex.: lognormal_0_10.jpeg
  9. Weibull       : weibull_<k_forma>_<lam_esc>   Ex.: weibull_3_2.jpeg
 10. Poisson       : poisson_<lambda>              Ex.: poisson_36.jpeg
 11. Binomial      : binomial_<n>_<p>              Ex.: binomial_50_0.1.jpeg
 12. Geométrica    : geometrica_<p>                Ex.: geometrica_0.7.jpeg
 13. Cauchy        : cauchy_<x0>_<gamma>           Ex.: cauchy_6_5.jpeg

--------------------------------------------------------------------------------
 2. COMO A ACURÁCIA É CALCULADA (AVALIAÇÃO 100% CEGA)
--------------------------------------------------------------------------------
  - A IA NUNCA vê o nome do arquivo: antes da análise, a imagem é copiada para
    um arquivo temporário anônimo. O nome serve APENAS como gabarito.
  - Para cada imagem i:
      * d_i = 1 se acertou a distribuição (0 se errou)
      * p_i = 1 se acertou os valores numéricos dos parâmetros (tolerância 1%)
      * Nota da imagem: s_i = (d_i + p_i) / 2
          - 1.0 ponto (100%): acertou distribuição E parâmetros
          - 0.5 ponto  (50%): acertou somente um dos dois
          - 0.0 ponto   (0%): errou distribuição e parâmetros
  - Acurácia Real da IA = (soma de s_i / N) * 100%

--------------------------------------------------------------------------------
 3. COMO RODAR A AVALIAÇÃO
--------------------------------------------------------------------------------
  - Pelo aplicativo StatMath: basta abrir o app ou clicar no selo de Acurácia
    Real na barra superior e clicar em "Reavaliar Pasta Agora".
  - Pelo terminal:
      python3 evaluate_real_test.py
================================================================================
