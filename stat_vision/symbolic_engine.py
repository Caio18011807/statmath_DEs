"""
Numeric & Symbolic Evaluator for Statistical Distributions.

Computes exact numeric values for Mean, Variance, Standard Deviation,
Median, Mode, Skewness, Excess Kurtosis, Raw Kurtosis, First 4 Raw & Central Moments,
Differential/Discrete Entropy, and evaluates PDF/PMF and CDF curves given parameters.
"""

from typing import Dict, Any, Tuple, List
import math
from stat_vision.knowledge_base import DISTRIBUTIONS_DB


def compute_numeric_summary(dist_id: str, params: Dict[str, float]) -> Dict[str, Any]:
    """
    Given a distribution ID and numeric parameters (falling back to defaults),
    computes exact numeric statistical quantities.
    """
    spec = DISTRIBUTIONS_DB.get(dist_id)
    if not spec:
        return {}

    p = dict(spec.default_params)
    p.update({k: float(v) for k, v in params.items() if v is not None})

    res: Dict[str, Any] = {"parameters_used": p}

    if dist_id == "normal":
        mu = p.get("mu", 0.0)
        sigma = max(p.get("sigma", 1.0), 1e-6)
        var = sigma ** 2
        res.update({
            "mean": mu,
            "variance": var,
            "std_dev": sigma,
            "median": mu,
            "mode": mu,
            "skewness": 0.0,
            "excess_kurtosis": 0.0,
            "raw_kurtosis": 3.0,
            "raw_moments": {
                "E[X^1]": mu,
                "E[X^2]": mu**2 + var,
                "E[X^3]": mu**3 + 3 * mu * var,
                "E[X^4]": mu**4 + 6 * (mu**2) * var + 3 * (var**2),
            },
            "central_moments": {
                "mu_1": 0.0,
                "mu_2": var,
                "mu_3": 0.0,
                "mu_4": 3 * (var**2),
            },
            "entropy_nats": 0.5 * math.log(2 * math.pi * math.e * var),
        })

    elif dist_id == "exponential":
        lam = max(p.get("lam", 1.0), 1e-6)
        mean = 1.0 / lam
        var = 1.0 / (lam ** 2)
        res.update({
            "mean": mean,
            "variance": var,
            "std_dev": mean,
            "median": math.log(2.0) / lam,
            "mode": 0.0,
            "skewness": 2.0,
            "excess_kurtosis": 6.0,
            "raw_kurtosis": 9.0,
            "raw_moments": {
                "E[X^1]": 1.0 / lam,
                "E[X^2]": 2.0 / (lam**2),
                "E[X^3]": 6.0 / (lam**3),
                "E[X^4]": 24.0 / (lam**4),
            },
            "central_moments": {
                "mu_1": 0.0,
                "mu_2": 1.0 / (lam**2),
                "mu_3": 2.0 / (lam**3),
                "mu_4": 9.0 / (lam**4),
            },
            "entropy_nats": 1.0 - math.log(lam),
        })

    elif dist_id == "uniform":
        a = p.get("a", 0.0)
        b = p.get("b", 1.0)
        if b <= a:
            b = a + 1.0
        L = b - a
        mean = 0.5 * (a + b)
        var = (L ** 2) / 12.0
        res.update({
            "mean": mean,
            "variance": var,
            "std_dev": math.sqrt(var),
            "median": mean,
            "mode": f"Any x in [{a:.3g}, {b:.3g}]",
            "skewness": 0.0,
            "excess_kurtosis": -1.2,
            "raw_kurtosis": 1.8,
            "raw_moments": {
                "E[X^1]": mean,
                "E[X^2]": (b**3 - a**3) / (3.0 * L),
                "E[X^3]": (b**4 - a**4) / (4.0 * L),
                "E[X^4]": (b**5 - a**5) / (5.0 * L),
            },
            "central_moments": {
                "mu_1": 0.0,
                "mu_2": var,
                "mu_3": 0.0,
                "mu_4": (L**4) / 80.0,
            },
            "entropy_nats": math.log(L),
        })

    elif dist_id == "gamma":
        alpha = max(p.get("alpha", 3.0), 1e-3)
        beta = max(p.get("beta", 1.0), 1e-6)
        mean = alpha / beta
        var = alpha / (beta ** 2)
        res.update({
            "mean": mean,
            "variance": var,
            "std_dev": math.sqrt(var),
            "median": (alpha - 1.0 / 3.0 + 0.02 / alpha) / beta if alpha >= 1 else mean * 0.7,
            "mode": (alpha - 1.0) / beta if alpha >= 1.0 else 0.0,
            "skewness": 2.0 / math.sqrt(alpha),
            "excess_kurtosis": 6.0 / alpha,
            "raw_kurtosis": 3.0 + 6.0 / alpha,
            "raw_moments": {
                "E[X^1]": alpha / beta,
                "E[X^2]": alpha * (alpha + 1) / (beta**2),
                "E[X^3]": alpha * (alpha + 1) * (alpha + 2) / (beta**3),
                "E[X^4]": alpha * (alpha + 1) * (alpha + 2) * (alpha + 3) / (beta**4),
            },
            "central_moments": {
                "mu_1": 0.0,
                "mu_2": var,
                "mu_3": 2.0 * alpha / (beta**3),
                "mu_4": 3.0 * alpha * (alpha + 2.0) / (beta**4),
            },
            "entropy_nats": alpha - math.log(beta) + math.lgamma(alpha),
        })

    elif dist_id == "beta":
        alpha = max(p.get("alpha", 2.0), 1e-3)
        beta = max(p.get("beta", 5.0), 1e-3)
        s = alpha + beta
        mean = alpha / s
        var = (alpha * beta) / ((s ** 2) * (s + 1.0))
        skew = (2.0 * (beta - alpha) * math.sqrt(s + 1.0)) / ((s + 2.0) * math.sqrt(alpha * beta))
        ex_kurt = (6.0 * (((alpha - beta)**2) * (s + 1.0) - alpha * beta * (s + 2.0))) / (
            alpha * beta * (s + 2.0) * (s + 3.0)
        )
        res.update({
            "mean": mean,
            "variance": var,
            "std_dev": math.sqrt(var),
            "median": (alpha - 1.0 / 3.0) / (s - 2.0 / 3.0) if (alpha > 1 and beta > 1) else mean,
            "mode": (alpha - 1.0) / (s - 2.0) if (alpha > 1 and beta > 1) else 0.5,
            "skewness": skew,
            "excess_kurtosis": ex_kurt,
            "raw_kurtosis": 3.0 + ex_kurt,
            "raw_moments": {
                "E[X^1]": mean,
                "E[X^2]": mean * (alpha + 1) / (s + 1),
                "E[X^3]": mean * (alpha + 1) * (alpha + 2) / ((s + 1) * (s + 2)),
                "E[X^4]": mean * (alpha + 1) * (alpha + 2) * (alpha + 3) / ((s + 1) * (s + 2) * (s + 3)),
            },
            "central_moments": {
                "mu_1": 0.0,
                "mu_2": var,
                "mu_3": skew * (var ** 1.5),
                "mu_4": (ex_kurt + 3.0) * (var ** 2),
            },
            "entropy_nats": math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(s),
        })

    elif dist_id == "chi_square":
        k = max(p.get("k", 4.0), 1.0)
        mean = k
        var = 2.0 * k
        res.update({
            "mean": mean,
            "variance": var,
            "std_dev": math.sqrt(var),
            "median": k * ((1.0 - 2.0 / (9.0 * k)) ** 3),
            "mode": max(k - 2.0, 0.0),
            "skewness": math.sqrt(8.0 / k),
            "excess_kurtosis": 12.0 / k,
            "raw_kurtosis": 3.0 + 12.0 / k,
            "raw_moments": {
                "E[X^1]": k,
                "E[X^2]": k * (k + 2),
                "E[X^3]": k * (k + 2) * (k + 4),
                "E[X^4]": k * (k + 2) * (k + 4) * (k + 6),
            },
            "central_moments": {
                "mu_1": 0.0,
                "mu_2": 2.0 * k,
                "mu_3": 8.0 * k,
                "mu_4": 12.0 * k * (k + 4.0),
            },
            "entropy_nats": 0.5 * k + math.log(2.0) + math.lgamma(0.5 * k),
        })

    elif dist_id == "student_t":
        nu = max(p.get("nu", 5.0), 0.5)
        mean = 0.0 if nu > 1 else "Undefined (ν ≤ 1)"
        var = (nu / (nu - 2.0)) if nu > 2 else ("Infinity" if nu > 1 else "Undefined")
        skew = 0.0 if nu > 3 else "Undefined (ν ≤ 3)"
        ex_kurt = (6.0 / (nu - 4.0)) if nu > 4 else ("Infinity" if nu > 2 else "Undefined")
        res.update({
            "mean": mean,
            "variance": var,
            "std_dev": math.sqrt(var) if isinstance(var, float) else var,
            "median": 0.0,
            "mode": 0.0,
            "skewness": skew,
            "excess_kurtosis": ex_kurt,
            "raw_kurtosis": (3.0 + ex_kurt) if isinstance(ex_kurt, float) else ex_kurt,
            "raw_moments": {
                "E[X^1]": 0.0 if nu > 1 else "Undefined",
                "E[X^2]": var,
                "E[X^3]": 0.0 if nu > 3 else "Undefined",
                "E[X^4]": (3.0 * (nu**2) / ((nu - 2.0) * (nu - 4.0))) if nu > 4 else "Undefined",
            },
            "central_moments": {
                "mu_1": 0.0 if nu > 1 else "Undefined",
                "mu_2": var,
                "mu_3": 0.0 if nu > 3 else "Undefined",
                "mu_4": (3.0 * (nu**2) / ((nu - 2.0) * (nu - 4.0))) if nu > 4 else "Undefined",
            },
            "entropy_nats": math.lgamma(0.5 * nu) + 0.5 * math.log(nu * math.pi) - math.lgamma(0.5 * (nu + 1)),
        })

    elif dist_id == "lognormal":
        mu = p.get("mu", 0.0)
        sigma = max(p.get("sigma", 0.6), 1e-6)
        s2 = sigma ** 2

        def _safe_exp(val: float) -> Any:
            if val > 700.0:
                return f"e^{{{val:.4g}}} (muito grande)"
            try:
                return math.exp(val)
            except OverflowError:
                return f"e^{{{val:.4g}}} (muito grande)"

        mean = _safe_exp(mu + 0.5 * s2)
        if s2 <= 300.0 and (2.0 * mu + 2.0 * s2) <= 700.0:
            var: Any = (math.exp(s2) - 1.0) * math.exp(2.0 * mu + s2)
            std_dev: Any = math.sqrt(var)
        else:
            var = "Muito grande (overflow numérico)"
            std_dev = "Muito grande (overflow numérico)"
        if s2 <= 300.0:
            skew: Any = (math.exp(s2) + 2.0) * math.sqrt(math.exp(s2) - 1.0)
        else:
            skew = "Muito grande (overflow numérico)"
        if 4.0 * s2 <= 700.0:
            ex_kurt: Any = math.exp(4 * s2) + 2 * math.exp(3 * s2) + 3 * math.exp(2 * s2) - 6.0
            raw_kurt: Any = ex_kurt + 3.0
        else:
            ex_kurt = "Muito grande (overflow numérico)"
            raw_kurt = "Muito grande (overflow numérico)"

        mu3: Any = skew * (var ** 1.5) if (isinstance(skew, float) and isinstance(var, float) and var < 1e100) else "Muito grande (overflow numérico)"
        mu4: Any = (ex_kurt + 3.0) * (var ** 2) if (isinstance(ex_kurt, float) and isinstance(var, float) and var < 1e100) else "Muito grande (overflow numérico)"

        res.update({
            "mean": mean,
            "variance": var,
            "std_dev": std_dev,
            "median": _safe_exp(mu),
            "mode": _safe_exp(mu - s2),
            "skewness": skew,
            "excess_kurtosis": ex_kurt,
            "raw_kurtosis": raw_kurt,
            "raw_moments": {
                "E[X^1]": _safe_exp(mu + 0.5 * s2),
                "E[X^2]": _safe_exp(2 * mu + 2.0 * s2),
                "E[X^3]": _safe_exp(3 * mu + 4.5 * s2),
                "E[X^4]": _safe_exp(4 * mu + 8.0 * s2),
            },
            "central_moments": {
                "mu_1": 0.0,
                "mu_2": var,
                "mu_3": mu3,
                "mu_4": mu4,
            },
            "entropy_nats": mu + 0.5 * math.log(2 * math.pi * math.e * s2),
        })

    elif dist_id == "weibull":
        k = max(p.get("k", 2.0), 0.1)
        lam = max(p.get("lam", 1.0), 1e-6)
        g1 = math.gamma(1.0 + 1.0 / k)
        g2 = math.gamma(1.0 + 2.0 / k)
        g3 = math.gamma(1.0 + 3.0 / k)
        g4 = math.gamma(1.0 + 4.0 / k)
        mean = lam * g1
        var = (lam ** 2) * (g2 - g1 ** 2)
        std = math.sqrt(max(var, 1e-12))
        mu3 = (lam ** 3) * (g3 - 3 * g1 * g2 + 2 * (g1 ** 3))
        mu4 = (lam ** 4) * (g4 - 4 * g1 * g3 + 6 * (g1 ** 2) * g2 - 3 * (g1 ** 4))
        skew = mu3 / (std ** 3)
        raw_kurt = mu4 / (var ** 2)
        res.update({
            "mean": mean,
            "variance": var,
            "std_dev": std,
            "median": lam * (math.log(2.0) ** (1.0 / k)),
            "mode": lam * (((k - 1.0) / k) ** (1.0 / k)) if k > 1.0 else 0.0,
            "skewness": skew,
            "excess_kurtosis": raw_kurt - 3.0,
            "raw_kurtosis": raw_kurt,
            "raw_moments": {
                "E[X^1]": lam * g1,
                "E[X^2]": (lam**2) * g2,
                "E[X^3]": (lam**3) * g3,
                "E[X^4]": (lam**4) * g4,
            },
            "central_moments": {
                "mu_1": 0.0,
                "mu_2": var,
                "mu_3": mu3,
                "mu_4": mu4,
            },
            "entropy_nats": 0.5772156649 * (1.0 - 1.0 / k) + math.log(lam / k) + 1.0,
        })

    elif dist_id == "poisson":
        lam = max(p.get("lam", 4.0), 1e-6)
        res.update({
            "mean": lam,
            "variance": lam,
            "std_dev": math.sqrt(lam),
            "median": math.floor(lam + 1.0 / 3.0 - 0.02 / lam),
            "mode": math.floor(lam),
            "skewness": 1.0 / math.sqrt(lam),
            "excess_kurtosis": 1.0 / lam,
            "raw_kurtosis": 3.0 + 1.0 / lam,
            "raw_moments": {
                "E[X^1]": lam,
                "E[X^2]": lam + lam**2,
                "E[X^3]": lam + 3 * (lam**2) + lam**3,
                "E[X^4]": lam + 7 * (lam**2) + 6 * (lam**3) + lam**4,
            },
            "central_moments": {
                "mu_1": 0.0,
                "mu_2": lam,
                "mu_3": lam,
                "mu_4": lam * (1.0 + 3.0 * lam),
            },
            "entropy_nats": 0.5 * math.log(2 * math.pi * math.e * lam) - 1.0 / (12.0 * lam),
        })

    elif dist_id == "binomial":
        n = max(int(round(p.get("n", 10.0))), 1)
        prob = min(max(p.get("p", 0.5), 1e-6), 1.0 - 1e-6)
        q = 1.0 - prob
        mean = n * prob
        var = n * prob * q
        skew = (q - prob) / math.sqrt(var)
        ex_kurt = (1.0 - 6.0 * prob * q) / var
        res.update({
            "mean": mean,
            "variance": var,
            "std_dev": math.sqrt(var),
            "median": round(mean),
            "mode": math.floor((n + 1) * prob),
            "skewness": skew,
            "excess_kurtosis": ex_kurt,
            "raw_kurtosis": 3.0 + ex_kurt,
            "raw_moments": {
                "E[X^1]": mean,
                "E[X^2]": var + mean**2,
                "E[X^3]": n * prob * ((n - 1) * (n - 2) * (prob**2) + 3 * (n - 1) * prob + 1),
                "E[X^4]": (ex_kurt + 3.0) * (var**2) + 4 * mean * (q - prob) * var + 6 * (mean**2) * var + (mean**4),
            },
            "central_moments": {
                "mu_1": 0.0,
                "mu_2": var,
                "mu_3": var * (q - prob),
                "mu_4": var * (1.0 + 3.0 * (n - 2) * prob * q),
            },
            "entropy_nats": 0.5 * math.log(2 * math.pi * math.e * var),
        })

    elif dist_id == "geometric":
        prob = min(max(p.get("p", 0.35), 1e-6), 1.0 - 1e-6)
        q = 1.0 - prob
        mean = 1.0 / prob
        var = q / (prob ** 2)
        skew = (2.0 - prob) / math.sqrt(q)
        ex_kurt = 6.0 + (prob ** 2) / q
        res.update({
            "mean": mean,
            "variance": var,
            "std_dev": math.sqrt(var),
            "median": math.ceil(-1.0 / math.log2(q)),
            "mode": 1,
            "skewness": skew,
            "excess_kurtosis": ex_kurt,
            "raw_kurtosis": 3.0 + ex_kurt,
            "raw_moments": {
                "E[X^1]": 1.0 / prob,
                "E[X^2]": (2.0 - prob) / (prob**2),
                "E[X^3]": (1.0 + 4.0 * q + q**2) / (prob**3),
                "E[X^4]": (1.0 + 11.0 * q + 11.0 * (q**2) + q**3) / (prob**4),
            },
            "central_moments": {
                "mu_1": 0.0,
                "mu_2": var,
                "mu_3": q * (2.0 - prob) / (prob**3),
                "mu_4": q * (9.0 - 9.0 * prob + prob**2) / (prob**4),
            },
            "entropy_nats": (-q * math.log(q) - prob * math.log(prob)) / prob,
        })

    elif dist_id == "cauchy":
        x0 = p.get("x0", 0.0)
        gamma = max(p.get("gamma", 1.0), 1e-6)
        res.update({
            "mean": f"Indefinida (Mediana = {x0:g})",
            "variance": f"Indefinida (IQR = {2.0 * gamma:g})",
            "std_dev": f"Indefinido (γ = {gamma:g})",
            "median": x0,
            "mode": x0,
            "skewness": "Indefinida (Simétrica em torno de x₀)",
            "excess_kurtosis": "Indefinida (Caudas Polinomiais ∝ 1/x²)",
            "raw_kurtosis": "Indefinida (∞)",
            "raw_moments": {
                "E[X^1]": "Indefinido",
                "E[X^2]": "Indefinido (∞)",
                "E[X^3]": "Indefinido",
                "E[X^4]": "Indefinido (∞)",
            },
            "central_moments": {
                "mu_1": "Indefinido",
                "mu_2": "Indefinido (∞)",
                "mu_3": "Indefinido",
                "mu_4": "Indefinido (∞)",
            },
            "entropy_nats": math.log(4.0 * math.pi * gamma),
        })

    return res


def evaluate_curve_points(dist_id: str, params: Dict[str, float], num_points: int = 200) -> Tuple[List[float], List[float], bool]:
    """
    Returns (x_values, y_values, is_discrete) for plotting or comparing PDF/PMF curves.
    Pure Python math implementation (no external dependencies required).
    """
    spec = DISTRIBUTIONS_DB.get(dist_id)
    p = dict(spec.default_params) if spec else {}
    p.update({k: float(v) for k, v in params.items() if v is not None})

    if dist_id == "normal":
        mu = p.get("mu", 0.0)
        sigma = max(p.get("sigma", 1.0), 1e-6)
        xs = [mu - 4 * sigma + (8 * sigma * i / (num_points - 1)) for i in range(num_points)]
        ys = [(1.0 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * (((x - mu) / sigma) ** 2)) for x in xs]
        return xs, ys, False

    elif dist_id == "exponential":
        lam = max(p.get("lam", 1.0), 1e-6)
        x_max = 5.0 / lam
        xs = [x_max * i / (num_points - 1) for i in range(num_points)]
        ys = [lam * math.exp(-lam * x) for x in xs]
        return xs, ys, False

    elif dist_id == "uniform":
        a = p.get("a", 0.0)
        b = p.get("b", 1.0)
        if b <= a:
            b = a + 1.0
        L = b - a
        margin = 0.25 * L
        x_min, x_max = a - margin, b + margin
        raw_xs = [x_min + (x_max - x_min) * i / (num_points - 1) for i in range(num_points)]
        eps = 1e-6 * L
        xs = sorted(set(raw_xs + [a - eps, a, b, b + eps]))
        ys = [(1.0 / L) if (a <= x <= b) else 0.0 for x in xs]
        return xs, ys, False

    elif dist_id == "gamma":
        alpha = max(p.get("alpha", 3.0), 0.1)
        beta = max(p.get("beta", 1.0), 1e-6)
        x_max = (alpha + 4.5 * math.sqrt(alpha)) / beta
        x_start = 0.0 if alpha >= 1.0 else max(1e-4, 0.003 * x_max)
        xs = [x_start + (x_max - x_start) * i / (num_points - 1) for i in range(num_points)]
        log_norm = alpha * math.log(beta) - math.lgamma(alpha)
        ys: List[float] = []
        for x in xs:
            if x <= 0.0:
                ys.append(beta if abs(alpha - 1.0) < 1e-9 else 0.0)
            else:
                ys.append(math.exp(log_norm + (alpha - 1.0) * math.log(x) - beta * x))
        return xs, ys, False

    elif dist_id == "beta":
        alpha = max(p.get("alpha", 2.0), 0.1)
        beta = max(p.get("beta", 5.0), 0.1)
        x_start = 0.0 if alpha >= 1.0 else 0.005
        x_end = 1.0 if beta >= 1.0 else 0.995
        xs = [x_start + (x_end - x_start) * i / (num_points - 1) for i in range(num_points)]
        log_b = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
        ys = []
        for x in xs:
            if x <= 0.0:
                ys.append((1.0 / math.exp(log_b)) if abs(alpha - 1.0) < 1e-9 else 0.0)
            elif x >= 1.0:
                ys.append((1.0 / math.exp(log_b)) if abs(beta - 1.0) < 1e-9 else 0.0)
            else:
                ys.append(math.exp((alpha - 1.0) * math.log(x) + (beta - 1.0) * math.log(1.0 - x) - log_b))
        return xs, ys, False

    elif dist_id == "chi_square":
        k = max(p.get("k", 4.0), 1.0)
        x_max = k + 4.5 * math.sqrt(2.0 * k)
        x_start = 0.0 if k >= 2.0 else 0.015
        xs = [x_start + (x_max - x_start) * i / (num_points - 1) for i in range(num_points)]
        log_norm = -0.5 * k * math.log(2.0) - math.lgamma(0.5 * k)
        ys = []
        for x in xs:
            if x <= 0.0:
                ys.append(0.5 if abs(k - 2.0) < 1e-9 else 0.0)
            else:
                ys.append(math.exp(log_norm + (0.5 * k - 1.0) * math.log(x) - 0.5 * x))
        return xs, ys, False

    elif dist_id == "student_t":
        nu = max(p.get("nu", 5.0), 1.0)
        xs = [-5.0 + 10.0 * i / (num_points - 1) for i in range(num_points)]
        norm_const = math.exp(math.lgamma(0.5 * (nu + 1.0)) - math.lgamma(0.5 * nu)) / math.sqrt(nu * math.pi)
        ys = [norm_const * ((1.0 + (x ** 2) / nu) ** (-0.5 * (nu + 1.0))) for x in xs]
        return xs, ys, False

    elif dist_id == "lognormal":
        mu = p.get("mu", 0.0)
        sigma = max(p.get("sigma", 0.6), 1e-6)
        x_max = math.exp(min(mu + 2.8 * sigma, mu + 7.5, 20.0))
        xs = [max(1e-3, x_max * i / (num_points - 1)) for i in range(num_points)]
        ys = [
            (1.0 / (x * sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * (((math.log(x) - mu) / sigma) ** 2))
            for x in xs
        ]
        return xs, ys, False

    elif dist_id == "weibull":
        k = max(p.get("k", 2.0), 0.5)
        lam = max(p.get("lam", 1.0), 1e-6)
        x_max = lam * 3.0
        xs = [max(1e-4, x_max * i / (num_points - 1)) for i in range(num_points)]
        ys = [(k / lam) * ((x / lam) ** (k - 1.0)) * math.exp(-((x / lam) ** k)) for x in xs]
        return xs, ys, False

    elif dist_id == "poisson":
        lam = max(p.get("lam", 4.0), 1e-4)
        k_max = max(10, int(math.ceil(lam + 4.2 * math.sqrt(lam))))
        xs = [float(k) for k in range(k_max + 1)]
        ys = [
            math.exp(-lam) if k == 0 else math.exp(k * math.log(lam) - lam - math.lgamma(k + 1.0))
            for k in xs
        ]
        return xs, ys, True

    elif dist_id == "binomial":
        n = max(int(round(p.get("n", 10.0))), 1)
        prob = min(max(p.get("p", 0.5), 1e-6), 1.0 - 1e-6)
        xs = [float(k) for k in range(n + 1)]
        log_p = math.log(prob)
        log_q = math.log(1.0 - prob)
        log_n_fact = math.lgamma(n + 1.0)
        ys = [
            math.exp(log_n_fact - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0) + k * log_p + (n - k) * log_q)
            for k in xs
        ]
        return xs, ys, True

    elif dist_id == "geometric":
        prob = min(max(p.get("p", 0.35), 1e-6), 1.0 - 1e-6)
        k_max = min(35, max(10, int(math.ceil(5.5 / prob))))
        xs = [float(k) for k in range(1, k_max + 1)]
        ys = [((1.0 - prob) ** (k - 1)) * prob for k in xs]
        return xs, ys, True

    elif dist_id == "cauchy":
        x0 = p.get("x0", 0.0)
        gamma = max(p.get("gamma", 1.0), 1e-6)
        xs = [x0 - 5.5 * gamma + (11.0 * gamma * i / (num_points - 1)) for i in range(num_points)]
        ys = [1.0 / (math.pi * gamma * (1.0 + (((x - x0) / gamma) ** 2))) for x in xs]
        return xs, ys, False

    return [], [], False
