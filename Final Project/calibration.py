

import math
import time
import warnings
from typing import List, Tuple, Optional

import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.special import ndtr
from scipy.integrate import quad, IntegrationWarning
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from Erdos import heston_call as _erdos_heston_call, heston_put as _erdos_heston_put

from rough import (
    rough_heston_paths_parallel,
    rough_heston_paths_parallel_pool,
    rough_heston_terminal_parallel_pool,
)


# ------------------------- Black–Scholes helpers -------------------------
SQRT_2PI = math.sqrt(2.0 * math.pi)


def _bs_call(F: float, K: float, df: float, T: float, sigma: float) -> float:
    volT = sigma * math.sqrt(T)
    inv = 1.0 / max(volT, 1e-12)
    d1 = (math.log(F / K) + 0.5 * volT * volT) * inv
    d2 = d1 - volT
    return df * (F * ndtr(d1) - K * ndtr(d2))


def _bs_vega(S0: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    F = S0 * math.exp((r - q) * T)
    df = math.exp(-r * T)
    volT = sigma * math.sqrt(T)
    inv = 1.0 / max(volT, 1e-12)
    d1 = (math.log(F / K) + 0.5 * volT * volT) * inv
    return df * F * math.exp(-0.5 * d1 * d1) / SQRT_2PI * math.sqrt(T)


def _implied_vol_call(price: float, S0: float, K: float, T: float, r: float, q: float,
                      sigma0: Optional[float] = None, iters: int = 8, tol: float = 1e-8) -> float:
    # Map dividend to forward measure for stability
    F = S0 * math.exp((r - q) * T)
    df = math.exp(-r * T)
    intrinsic = max(S0 * math.exp(-q * T) - K * df, 0.0)
    upper = S0 * math.exp(-q * T)
    C = min(max(price, intrinsic), upper)
    if sigma0 is None:
        # ATM approximation
        if abs(math.log(F / K)) < 1e-6:
            sigma0 = min(max((C / (df * F)) * math.sqrt(2.0 * math.pi) / math.sqrt(T), 1e-4), 3.0)
        else:
            sigma0 = 0.2
    sigma = float(np.clip(sigma0, 1e-6, 5.0))
    for _ in range(max(1, int(iters))):
        volT = sigma * math.sqrt(T)
        inv = 1.0 / max(volT, 1e-12)
        d1 = (math.log(F / K) + 0.5 * volT * volT) * inv
        d2 = d1 - volT
        price_model = df * (F * ndtr(d1) - K * ndtr(d2))
        vega = df * F * math.exp(-0.5 * d1 * d1) / SQRT_2PI * math.sqrt(T)
        vega = max(vega, 1e-12)
        step = (price_model - C) / vega
        sigma = float(np.clip(sigma - step, 1e-6, 5.0))
        if abs(step) < tol:
            break
    return sigma


def _implied_vol(S0: float, K: float, T: float, r: float, q: float, price: float, cp: str) -> float:
    cp_l = str(cp).lower()
    S_eff = float(S0) * math.exp(-float(q) * float(T))
    r_eff = float(r) - float(q)
    eps = 1e-10
    if cp_l == "put":
        intrinsic_put = max(0.0, float(K) - S_eff)
        upper_put = float(K) * math.exp(-r_eff * float(T))
        if not (intrinsic_put - eps < price < upper_put + eps):
            return np.nan
        call_price = price + float(S0) * math.exp(-float(q) * float(T)) - float(K) * math.exp(-float(r) * float(T))
    else:
        intrinsic_call = max(0.0, S_eff - float(K))
        upper_call = S_eff
        if not (intrinsic_call - eps < price < upper_call + eps):
            return np.nan
        call_price = price
    try:
        return _implied_vol_call(call_price, float(S0), float(K), float(T), float(r), float(q), sigma0=0.2)
    except Exception:
        return np.nan


def _smile_from_ST(ST: np.ndarray, r: float, T: float, strikes: np.ndarray, cp: str = "call") -> np.ndarray:
    DF = math.exp(-r * T)
    ST = np.asarray(ST, dtype=float).reshape(-1)
    K = np.asarray(strikes, dtype=float).reshape(-1)
    if cp == "call":
        payoff = np.maximum(ST[:, None] - K[None, :], 0.0)
    else:
        payoff = np.maximum(K[None, :] - ST[:, None], 0.0)
    return DF * payoff.mean(axis=0)


def _vega_weight_array(
    S0: float, r: float, q: float, T: float, strikes: np.ndarray, ivs: Optional[np.ndarray],
    floor: float = 0.25, cap: float = 4.0,
    wing_boost_alpha: Optional[float] = 0.35, wing_boost_power: Optional[float] = 1.0,
) -> Optional[np.ndarray]:
    if ivs is None:
        return None
    strikes = np.asarray(strikes, float)
    ivs = np.asarray(ivs, float)
    vega = np.empty_like(strikes, dtype=float)
    vega[:] = np.nan
    for idx, (K, sig) in enumerate(zip(strikes, ivs)):
        if not np.isfinite(sig):
            continue
        vega[idx] = _bs_vega(S0, float(K), T, r, float(sig), q=q)

    mask = np.isfinite(vega)
    if not np.any(mask):
        return None

    weights = np.abs(vega[mask])
    scale = np.percentile(weights, 75)
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = np.nanmean(weights)
    scaled = np.ones_like(weights) if (not np.isfinite(scale) or scale <= 1e-8) else weights / scale
    mean_val = np.nanmean(scaled)
    if np.isfinite(mean_val) and mean_val > 1e-8:
        scaled = scaled / mean_val

    if wing_boost_alpha is not None and float(wing_boost_alpha) > 0.0:
        try:
            F = float(S0) * math.exp((float(r) - float(q)) * float(T))
        except Exception:
            F = float(S0)
        with np.errstate(divide="ignore", invalid="ignore"):
            m = np.abs(np.log(strikes[mask] / F))
        pwr = float(wing_boost_power or 1.0)
        boost = 1.0 + float(wing_boost_alpha) * np.power(m, pwr)
        boost = np.clip(boost, 1.0, 1e3)
        scaled = scaled * boost

    scaled = np.clip(scaled, float(floor), float(cap))
    out = np.zeros_like(strikes, dtype=float)
    out[mask] = scaled
    return out


class _CalibMonitor:
    def __init__(self, tag: str = "calib", print_every: int = 1, verbose: bool = True):
        self.tag = tag
        self.print_every = max(1, int(print_every))
        self.verbose = bool(verbose)
        self.iter = 0
        self.last_x = None
        self.last_f = None
        self.t0 = None
        self.t_last = None
        self.history = []

    def wrap_obj(self, fn):
        def _wrapped(x):
            val = fn(x)
            self.last_x = np.array(x, dtype=float)
            self.last_f = float(val)
            return val
        return _wrapped

    def start(self, start_idx: int = 1):
        self.iter = 0
        self.last_x = None
        self.last_f = None
        self.t0 = time.time()
        self.t_last = self.t0
        if self.verbose:
            print(f"[{self.tag} start #{start_idx}] iter=0")

    def cb(self, xk):
        self.iter += 1
        now = time.time()
        dt = now - (self.t_last or now)
        self.t_last = now

        df = np.nan
        dx = np.nan
        if self.last_f is not None and self.history:
            df = self.last_f - self.history[-1]["f"]
        if self.last_x is not None and self.history:
            dx = float(np.linalg.norm(self.last_x - self.history[-1]["x"], ord=2))

        rec = dict(iter=self.iter, f=self.last_f, df=df, dx=dx, t=now - (self.t0 or now), dt=dt, x=None)
        if self.last_x is not None:
            rec["x"] = self.last_x.copy()
        self.history.append(rec)

        if self.verbose and (self.iter % self.print_every == 0):
            f_str = "nan" if self.last_f is None else f"{self.last_f:.6f}"
            df_str = "nan" if np.isnan(df) else f"{df:+.2e}"
            dx_str = "nan" if np.isnan(dx) else f"{dx:.2e}"
            print(f"[{self.tag}] iter={self.iter}  f={f_str}  df={df_str}  |dx|={dx_str}  {dt:.2f}s")

    def done(self, label: str = "best"):
        if self.verbose:
            if self.history:
                f_best = self.history[-1]["f"]
            else:
                f_best = self.last_f
            total = time.time() - (self.t0 or time.time())
            print(f"[{self.tag} {label}] f={f_best:.6f}  iters={self.iter}  {total:.2f}s")


# --------------------------- Classic Heston helpers ---------------------------

def heston_smile_prices(
    S0: float,
    r: float,
    q: float,
    T: float,
    strikes: np.ndarray,
    *,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    option: str = "call",
    integration_limit: float = 200.0,
) -> np.ndarray:
    strikes = np.asarray(strikes, dtype=float)
    option_l = str(option).lower()
    if option_l not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'")

    if T <= 1e-8:
        disc_q = math.exp(-q * T)
        disc_r = math.exp(-r * T)
        payoff = strikes * disc_r - S0 * disc_q if option_l == "put" else S0 * disc_q - strikes * disc_r
        return np.maximum(payoff, 0.0)

    r_eff = float(r - q)
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)

    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)

    prices = []
    for K in strikes:
        K = float(K)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=IntegrationWarning)
                raw_call = _erdos_heston_call(
                    float(S0), K, float(v0), r_eff, float(T),
                    float(kappa), float(theta), float(sigma), float(rho)
                )
            call_price = float(raw_call) * disc_q
        except Exception:
            call_price = float("nan")  # fallback handled below
        if not math.isfinite(call_price):
            F = S0 * math.exp(r_eff * T)
            sigma_bs = math.sqrt(max(v0, 1e-6))
            bs_call = _bs_call(F, K, disc_r, float(T), sigma_bs)
            call_price = bs_call

        if option_l == "put":
            adj_price = call_price - S0 * disc_q + K * disc_r
        else:
            adj_price = call_price
        prices.append(adj_price)

    return np.asarray(prices, dtype=float)


# ------------------------ Objective and calibration ------------------------


def _heston_objective(params, data, metric, weights, mc):
    v0, kappa, theta, sigma, rho = params
    eps = 1e-8
    if rho <= -0.999:
        rho = -0.999 + eps
    elif rho >= 0.999:
        rho = 0.999 - eps
    if v0 <= 0 or theta <= 0 or sigma <= 0 or kappa <= 0 or not (-0.999 < rho < 0.999):
        return 1e6

    integ_limit = float(mc.get("integration_limit", 100.0))
    err2 = 0.0
    for (S0, r, q, T, strikes, mids, cp, mkt_iv_opt) in data:
        model_px = heston_smile_prices(
            S0, r, q, T, strikes,
            v0=v0, kappa=kappa, theta=theta, sigma=sigma, rho=rho,
            option=cp, integration_limit=integ_limit,
        )

        key = (T, cp)
        if metric == "price":
            resid = model_px - mids
            if weights is not None and key in weights:
                w = np.asarray(weights[key], float)
                mask = np.isfinite(resid) & np.isfinite(w)
                resid = resid[mask] * w[mask]
            else:
                mask = np.isfinite(resid)
                resid = resid[mask]
        else:
            mod_iv = np.array([_implied_vol(S0, K, T, r, q, p, cp) for K, p in zip(strikes, model_px)])
            mkt_iv = mkt_iv_opt
            mask = np.isfinite(mod_iv)
            if mkt_iv is not None:
                mkt_iv = np.asarray(mkt_iv, float)
                mask &= np.isfinite(mkt_iv)
            if weights is not None and key in weights:
                w = np.asarray(weights[key], float)
                mask &= np.isfinite(w)
            else:
                w = None
            if not np.any(mask):
                return 1e6
            resid = mod_iv[mask] - mkt_iv[mask]
            if w is not None:
                resid = resid * w[mask]
            resid = resid[np.isfinite(resid)]

        if resid.size == 0:
            return 1e6
        err2 += float(resid @ resid)

    return 1e6 if (not np.isfinite(err2)) else err2

def _rough_objective(params, data, metric, weights, mc, seed, exec_ctx=None, terminal_only=True):
    v0, kappa, theta, eta, rho, H = params
    eps = 1e-6
    if rho <= -0.999:
        rho = -0.999 + eps
    elif rho >= 0.999:
        rho = 0.999 - eps
    if v0 <= 0 or theta <= 0 or eta <= 0 or kappa <= 0 or not (-0.999 < rho < 0.999):
        return 1e6
    if H <= 0.02:
        H = 0.02 + eps
    elif H >= 0.5:
        H = 0.5 - eps

    err2 = 0.0
    use_numba_flag = mc.get("use_numba", None)
    for (S0, r, q, T, strikes, mids, cp, mkt_iv_opt) in data:
        base_seed = seed + int(1000 * T)
        N_eff = mc.get("N", 192)
        Npy = mc.get("N_per_year", None)
        if Npy is not None:
            try:
                N_eff = max(int(mc.get("N_min", 128)), int(math.ceil(float(Npy) * float(T))))
            except Exception:
                N_eff = mc.get("N", 192)
        paths = int(mc.get("paths", 10000))
        batch_size = int(mc.get("batch_size", 4096))

        if exec_ctx is None:
            _t, S, _V = rough_heston_paths_parallel(
                S0=S0, v0=v0, T=T, N=N_eff, n_paths=paths, H=H, kappa=kappa, theta=theta, eta=eta, rho=rho,
                r=r, q=q, base_seed=base_seed, n_workers=int(mc.get("n_workers", 4)),
                batch_size=batch_size, use_numba=use_numba_flag,
            )
            ST = S[:, -1]
        else:
            if terminal_only:
                ST = rough_heston_terminal_parallel_pool(
                    exec_ctx,
                    S0=S0, v0=v0, T=T, N=N_eff, n_paths=paths, H=H, kappa=kappa, theta=theta, eta=eta, rho=rho,
                    r=r, q=q, base_seed=base_seed, batch_size=batch_size, use_numba=use_numba_flag,
                )
            else:
                _t, S, _V = rough_heston_paths_parallel_pool(
                    exec_ctx,
                    S0=S0, v0=v0, T=T, N=N_eff, n_paths=paths, H=H, kappa=kappa, theta=theta, eta=eta, rho=rho,
                    r=r, q=q, base_seed=base_seed, batch_size=batch_size, use_numba=use_numba_flag,
                )
                ST = S[:, -1]

        model_px = _smile_from_ST(ST, r, T, strikes, cp=cp)

        key = (T, cp)

        if metric == "price":
            resid = model_px - mids
            if weights is not None and key in weights:
                w = np.asarray(weights[key], float)
                mask = np.isfinite(resid) & np.isfinite(w)
                resid = resid[mask] * w[mask]
            else:
                mask = np.isfinite(resid)
                resid = resid[mask]
        else:
            mod_iv = np.array([_implied_vol(S0, K, T, r, q, p, cp) for K, p in zip(strikes, model_px)])
            mkt_iv = mkt_iv_opt
            mask = np.isfinite(mod_iv)
            if mkt_iv is not None:
                mkt_iv = np.asarray(mkt_iv, float)
                mask &= np.isfinite(mkt_iv)
            if weights is not None and key in weights:
                w = np.asarray(weights[key], float)
                mask &= np.isfinite(w)
            else:
                w = None
            if not np.any(mask):
                return 1e6
            resid = mod_iv[mask] - mkt_iv[mask]
            if w is not None:
                resid = resid * w[mask]
            mask_resid = np.isfinite(resid)
            resid = resid[mask_resid]

        if resid.size == 0:
            return 1e6
        err2 += float(resid @ resid)

    if not np.isfinite(err2):
        return 1e6
    return err2


def calibrate_rough_heston(
    smiles: List[Tuple],
    metric: str = "iv",
    vega_weight: bool = True,
    bounds=((1e-4, 0.5), (0.05, 6.0), (1e-4, 0.5), (0.05, 3.0), (-0.999, -0.01), (0.02, 0.45)),
    x0=(0.04, 1.5, 0.04, 1.8, -0.7, 0.1),
    mc=dict(N=192, paths=12000, batch_size=4096, n_workers=4, use_numba=True),
    seed: int = 7777,
    n_workers: Optional[int] = 4,
    parallel_backend: str = "process",
    terminal_only: bool = True,
    options=None,
    multistart: int = 3,
    verbose: bool = True,
    print_every: int = 1,
    wing_boost_alpha: float | None = 0.35,
    wing_boost_power: float | None = 1.0,
):
    """
    Calibrate rough Heston (v0, kappa, theta, eta, rho, H) to one or more smiles.
    Smiles entries are (S0, r, q, T, strikes, mids, cp).
    """
    if n_workers is not None:
        mc = dict(mc)
        mc["n_workers"] = int(n_workers)
    else:
        mc = dict(mc)

    if "use_numba" not in mc:
        mc["use_numba"] = True

    try:
        paths = int(mc.get("paths", 0))
        workers = int(mc.get("n_workers", 0))
    except (TypeError, ValueError):
        paths = workers = 0
    if paths > 0 and workers > 0:
        current_bs = int(mc.get("batch_size", max(paths, 1)))
        target_bs = max(512, int(math.ceil(paths / (workers * 2))))
        if current_bs > target_bs:
            mc["batch_size"] = target_bs

    dat = []
    for (S0, r, q, T, strikes, mids, cp) in smiles:
        cp_l = str(cp).lower()
        if metric == "iv":
            mkt_iv = np.array([_implied_vol(S0, K, T, r, q, p, cp_l) for K, p in zip(strikes, mids)])
        else:
            mkt_iv = None
        dat.append((S0, r, q, T, np.asarray(strikes, float), np.asarray(mids, float), cp_l, mkt_iv))

    weights = None
    if vega_weight and metric == "iv":
        weights = {}
        for (S0, r, q, T, strikes, mids, cp, mkt_iv) in dat:
            w = _vega_weight_array(
                S0, r, q, T, strikes, mkt_iv,
                wing_boost_alpha=wing_boost_alpha,
                wing_boost_power=wing_boost_power,
            )
            if w is not None:
                weights[(T, cp)] = w

    b = Bounds([b[0] for b in bounds], [b[1] for b in bounds])

    best = None
    best_hist = None
    rng = np.random.default_rng(2024)
    starts = [np.array(x0, dtype=float)]
    for _ in range(max(0, int(multistart) - 1)):
        noise = np.array([0.2, 0.2, 0.2, 0.2, 0.05, 0.1]) * (rng.random(6) - 0.5)
        starts.append(np.clip(starts[0] * (1.0 + noise), b.lb, b.ub))

    for i, guess in enumerate(starts, 1):
        tag = f"RoughHeston #{i}"
        mon = _CalibMonitor(tag=tag, print_every=print_every, verbose=verbose)

        max_workers = int(mc.get("n_workers", 4))
        Executor = ThreadPoolExecutor if str(parallel_backend).lower().startswith("thread") else ProcessPoolExecutor
        with Executor(max_workers=max_workers) as ex:
            obj = lambda x: _rough_objective(x, dat, metric, weights, mc, seed, ex, terminal_only)
            obj_wrapped = mon.wrap_obj(obj)
            mon.start(start_idx=i)
            _opt = {"maxiter": 200, "disp": False}
            if options:
                _opt.update(options)
            if ("eps" not in _opt) and ("finite_diff_rel_step" not in _opt):
                _opt["finite_diff_rel_step"] = 5e-2
            res = minimize(
                obj_wrapped,
                x0=np.array(guess),
                method="L-BFGS-B",
                bounds=b,
                options=_opt,
                callback=mon.cb,
            )
            mon.done(label="best")
            res.history = mon.history

        if (best is None) or (res.fun < best.fun):
            best = res
            best_hist = mon.history

    p = best.x
    out = dict(
        v0=p[0], kappa=p[1], theta=p[2], eta=p[3], rho=p[4], H=p[5],
        obj=float(best.fun), success=bool(best.success), nit=int(getattr(best, "nit", 0)),
        history=best_hist,
    )
    return out, best


def calibrate_heston(
    smiles: List[Tuple],
    metric: str = "iv",
    vega_weight: bool = True,
    bounds=((1e-4, 0.5), (0.05, 8.0), (1e-4, 0.5), (0.02, 2.5), (-0.999, -0.01)),
    x0=(0.04, 1.5, 0.04, 1.0, -0.7),
    mc=dict(integration_limit=100.0),
    seed: Optional[int] = None,
    n_workers: Optional[int] = None,
    parallel_backend: str = "thread",
    options=None,
    multistart: int = 2,
    verbose: bool = True,
    print_every: int = 1,
):
    """
    Calibrate classic Heston (v0, kappa, theta, sigma, rho) to one or more smiles.
    """
    mc = dict(mc or {})
    dat = []
    for (S0, r, q, T, strikes, mids, cp) in smiles:
        cp_l = str(cp).lower()
        if metric == "iv":
            mkt_iv = np.array([_implied_vol(S0, K, T, r, q, p, cp_l) for K, p in zip(strikes, mids)])
        else:
            mkt_iv = None
        dat.append((S0, r, q, T, np.asarray(strikes, float), np.asarray(mids, float), cp_l, mkt_iv))

    weights = None
    if vega_weight and metric == "iv":
        weights = {}
        for (S0, r, q, T, strikes, mids, cp_l, mkt_iv) in dat:
            w = _vega_weight_array(
                S0, r, q, T, strikes, mkt_iv,
                wing_boost_alpha=0.35,
                wing_boost_power=1.0,
            )
            if w is not None:
                weights[(T, cp_l)] = w

    b = Bounds([b[0] for b in bounds], [b[1] for b in bounds])

    best = None
    best_hist = None
    rng = np.random.default_rng(3101)
    starts = [np.array(x0, dtype=float)]
    for _ in range(max(0, int(multistart) - 1)):
        noise = np.array([0.2, 0.2, 0.2, 0.25, 0.05]) * (rng.random(5) - 0.5)
        starts.append(np.clip(starts[0] * (1.0 + noise), b.lb, b.ub))

    for i, guess in enumerate(starts, 1):
        tag = f"Heston #{i}"
        mon = _CalibMonitor(tag=tag, print_every=print_every, verbose=verbose)
        obj = lambda x: _heston_objective(x, dat, metric, weights, mc)
        obj_wrapped = mon.wrap_obj(obj)
        mon.start(start_idx=i)
        _opt = {"maxiter": 200, "disp": False, "ftol": 1e-8, "gtol": 1e-6}
        if options:
            _opt.update(options)
        if ("eps" not in _opt) and ("finite_diff_rel_step" not in _opt):
            _opt["eps"] = np.array([3e-4, 2e-2, 3e-4, 1e-2, 5e-3], dtype=float)
        res = minimize(
            obj_wrapped,
            x0=np.array(guess),
            method="L-BFGS-B",
            bounds=b,
            options=_opt,
            callback=mon.cb,
        )
        mon.done(label="best")
        res.history = mon.history
        if (best is None) or (res.fun < best.fun):
            best = res
            best_hist = mon.history

    p = best.x
    out = dict(
        v0=p[0], kappa=p[1], theta=p[2], sigma=p[3], rho=p[4],
        obj=float(best.fun), success=bool(best.success), nit=int(getattr(best, "nit", 0)),
        history=best_hist,
    )
    return out, best
