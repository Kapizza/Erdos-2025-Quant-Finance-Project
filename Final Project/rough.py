"""
Rough Heston log-Euler simulator (Volterra discretization) with optional Numba.

References
- O. El Euch and M. Rosenbaum (2019), "The characteristic function of rough
  Heston models", Mathematical Finance 29(1), 3–38.
- E. Abi Jaber, M. Larsson, and S. Pulido (2019), "Affine Volterra processes",
  Annals of Applied Probability 29(5), 3155–3200.
- J. Gatheral, T. Jaisson, and M. Rosenbaum (2018), "Volatility is rough",
  Quantitative Finance 18(6), 933–949.

"""

import math
import os
import numpy as np
from typing import Tuple, Optional


# ------------------------ utilities and guards ------------------------


def _floor_pos(x: float, eps: float = 1e-12) -> float:
    """
    Clamp a value to be strictly non-negative by flooring at ``eps``.

    Parameters
    ----------
    x : float
        Input value.
    eps : float, default 1e-12
        Minimum allowed value; returned if ``x < eps``.

    Returns
    -------
    float
        ``max(float(x), eps)``.
    """
    return max(float(x), eps)


# ---------- parallel utilities ----------

from concurrent.futures import ProcessPoolExecutor
from numpy.random import SeedSequence
from typing import List, Tuple


def _split_batches(n, batch_size):
    """
    Split a total count ``n`` into a list of chunk sizes of at most ``batch_size``.

    Parameters
    ----------
    n : int
        Total number of items to split.
    batch_size : int
        Maximum size of each chunk (clamped to at least 1).

    Returns
    -------
    list[int]
        Sizes whose sum equals ``n``.
    """
    n = int(n); batch_size = int(max(1, batch_size))
    sizes = []
    done = 0
    while done < n:
        take = min(batch_size, n - done)
        sizes.append(take)
        done += take
    return sizes

def _child_seeds(base_seed, n_children):
    """
    Derive reproducible child seeds from a base seed using ``SeedSequence``.

    Parameters
    ----------
    base_seed : int
        Base integer seed.
    n_children : int
        Number of child seeds to produce.

    Returns
    -------
    list[int]
        Raw 32-bit seeds usable by ``np.random.default_rng``.
    """
    ss = SeedSequence(int(base_seed))
    kids = ss.spawn(int(n_children))
    # return raw ints for np.random.default_rng
    return [int(k.generate_state(1)[0]) for k in kids]


# Try to avoid thread oversubscription when we parallelize at Python level.
# Respect existing env if the user already configured them.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    if _var not in os.environ:
        os.environ[_var] = "1"

# Optional Numba acceleration
try:  # pragma: no cover - optional dependency
    from numba import njit, prange
except Exception:  # pragma: no cover
    njit = None
    prange = range  # fallback

_HAS_NUMBA = njit is not None





# ------------------------ Rough Heston paths ------------------------

def _validate_option(opt):
    """
    Validate and normalize an option type.

    Parameters
    ----------
    opt : str
        Option type; must be "call" or "put" (case-insensitive).

    Returns
    -------
    str
        Lowercase normalized option type ("call" or "put").
    """
    s=str(opt).lower()
    if s not in ("call","put"): raise ValueError("option must be 'call' or 'put'")
    return s
def _gamma(x):
    """Thin wrapper over math.gamma returning float."""
    return math.gamma(float(x))

def _kernel_weights(H, N, dt):
    """
    Fractional kernel coefficients for the rough Heston Volterra form.

    Parameters
    ----------
    H : float
        Hurst parameter in (0, 1).
    N : int
        Number of time steps.
    dt : float
        Time step size (T/N).

    Returns
    -------
    tuple
        (cH, alpha, drift_scale, diff_scale) where
        - cH = 1/Gamma(H+1/2)
        - alpha: length N+1 weights used in the discrete convolution
        - drift_scale = cH * dt^(H+1/2)
        - diff_scale  = cH * dt^H
    """
    H=float(H); N=int(N)
    if not (0.0 < H < 1.0): raise ValueError("H must be in (0,1)")
    if N<1: raise ValueError("N must be >= 1")
    m = np.arange(0, N+1, dtype=float)
    alpha = np.zeros(N+1, dtype=float)
    if N>=1: alpha[1:] = m[1:]**(H+0.5) - m[:-1]**(H+0.5)
    cH = 1.0/_gamma(H+0.5)
    drift_scale = cH * (dt**(H+0.5))
    diff_scale  = cH * (dt**H)
    return cH, alpha, drift_scale, diff_scale

def _rough_heston_paths_python(S0, v0, T, N, n_paths, H, kappa, theta, eta, rho, r=0.0, q=0.0, seed=None, batch_size=1024):
    """
    Simulate Rough Heston price/variance paths via a log-Euler scheme (NumPy).

    Parameters
    ----------
    S0, v0 : float
        Initial price and variance.
    T : float
        Maturity in years.
    N : int
        Number of time steps.
    n_paths : int
        Number of Monte Carlo paths.
    H : float
        Hurst parameter in (0, 1).
    kappa, theta, eta : float
        Mean reversion, long-run variance, vol-of-vol.
    rho : float
        Correlation between variance and price Brownian motions.
    r, q : float, optional
        Risk-free rate and dividend yield.
    seed : int, optional
        RNG seed for reproducibility.
    batch_size : int, optional
        Per-batch path count to control memory.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (t, S, V) where t is shape (N+1,), S and V are arrays of
        shape (n_paths, N+1).
    """
    S0=_floor_pos(S0); v0=max(float(v0),0.0); T=_floor_pos(T)
    N=int(N); n_paths=int(n_paths)
    if N<1 or n_paths<1: raise ValueError("N and n_paths must be >= 1")
    if not (-0.999 < rho < 0.999): raise ValueError("rho must be in (-0.999,0.999)")
    if not (0.0 < float(H) < 1.0): raise ValueError("H must be in (0,1)")
    kappa=float(kappa); theta=max(float(theta),0.0); eta=_floor_pos(eta)
    r=float(r); q=float(q); batch_size=int(max(1,batch_size))

    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, T, N+1); dt = T/N; sqrt_dt = math.sqrt(dt)
    _, alpha, drift_scale, diff_scale = _kernel_weights(H, N, dt)

    S = np.empty((n_paths, N+1)); V = np.empty((n_paths, N+1))
    S[:,0]=S0; V[:,0]=v0

    for start in range(0, n_paths, batch_size):
        end=min(n_paths,start+batch_size); m=end-start
        Z1 = rng.standard_normal((m,N))
        Z2 = rng.standard_normal((m,N))
        dW_S = rho*Z2 + math.sqrt(1.0-rho*rho)*Z1

        Sb = np.empty((m,N+1)); Vb = np.empty((m,N+1))
        Sb[:,0]=S0; Vb[:,0]=v0

        for k in range(N):
            Vk = np.maximum(Vb[:,k],1e-14); sqrtVk = np.sqrt(Vk)

            idx = np.arange(k+1); mlag=(k+1)-idx
            a = alpha[mlag]
            V_hist  = Vb[:,:k+1]
            Z2_hist = Z2[:,:k+1]
            sqrtV   = np.sqrt(np.maximum(V_hist,1e-14))
            drift_conv = np.dot(kappa*(theta - V_hist), a)
            diff_conv  = np.dot(sqrtV * Z2_hist, a)

            V_next = v0 + drift_scale*drift_conv + eta*diff_scale*diff_conv
            V_next = np.maximum(V_next,1e-14)
            Vb[:,k+1]=V_next

            Sb[:,k+1] = Sb[:,k] * np.exp((r-q)*dt - 0.5*Vk*dt + sqrtVk*sqrt_dt*dW_S[:,k])

        S[start:end,:]=Sb; V[start:end,:]=Vb

    return t, S, V


if _HAS_NUMBA:  # pragma: no cover - optional dependency

    @njit(parallel=True)
    def _rough_heston_kernel(S0, v0, dt, sqrt_dt, N, rho, r, q, kappa, theta, eta,
                             alpha, drift_scale, diff_scale, Z1, Z2, S_out, V_out):
        """
        Numba kernel to simulate Rough Heston paths in place.

        Parameters
        ----------
        S0, v0, dt, sqrt_dt, N, rho, r, q, kappa, theta, eta : float
            Model and time-grid parameters.
        alpha : np.ndarray
            Discrete kernel weights of length N+1.
        drift_scale, diff_scale : float
            Scaling coefficients for convolutions.
        Z1, Z2 : np.ndarray
            Standard normal arrays (n_paths, N) for independent drivers.
        S_out, V_out : np.ndarray
            Output arrays (n_paths, N+1).
        """
        n_paths = S_out.shape[0]
        sqrt1mrho2 = math.sqrt(max(1.0 - rho * rho, 0.0))
        for i in prange(n_paths):
            S_out[i, 0] = S0
            V_out[i, 0] = v0
            for k in range(N):
                Vk_raw = V_out[i, k]
                Vk = Vk_raw if Vk_raw > 1e-14 else 1e-14
                sqrtVk = math.sqrt(Vk)

                drift_conv = 0.0
                diff_conv = 0.0
                for j in range(k + 1):
                    a = alpha[k + 1 - j]
                    v_hist = V_out[i, j]
                    drift_conv += kappa * (theta - v_hist) * a
                    v_hist_clamped = v_hist if v_hist > 1e-14 else 1e-14
                    diff_conv += math.sqrt(v_hist_clamped) * Z2[i, j] * a

                V_next = v0 + drift_scale * drift_conv + eta * diff_scale * diff_conv
                if V_next < 1e-14:
                    V_next = 1e-14
                V_out[i, k + 1] = V_next

                dW_S = rho * Z2[i, k] + sqrt1mrho2 * Z1[i, k]
                S_out[i, k + 1] = S_out[i, k] * math.exp((r - q) * dt - 0.5 * Vk * dt + sqrtVk * sqrt_dt * dW_S)

    def _rough_heston_paths_numba(S0, v0, T, N, n_paths, H, kappa, theta, eta, rho,
                                  r=0.0, q=0.0, seed=None, batch_size=1024):
        """
        Simulate Rough Heston paths using the Numba kernel in batches.

        Parameters
        ----------
        See `_rough_heston_paths_python` for argument meanings.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            (t, S, V) with the same shapes as the Python implementation.
        """
        S0 = _floor_pos(S0); v0 = max(float(v0), 0.0); T = _floor_pos(T)
        N = int(N); n_paths = int(n_paths)
        if N < 1 or n_paths < 1:
            raise ValueError("N and n_paths must be >= 1")
        if not (-0.999 < rho < 0.999):
            raise ValueError("rho must be in (-0.999,0.999)")
        if not (0.0 < float(H) < 1.0):
            raise ValueError("H must be in (0,1)")
        kappa = float(kappa); theta = max(float(theta), 0.0); eta = _floor_pos(eta)
        r = float(r); q = float(q); batch_size = int(max(1, batch_size))

        rng = np.random.default_rng(seed)
        t = np.linspace(0.0, T, N + 1)
        dt = T / N
        sqrt_dt = math.sqrt(dt)
        _, alpha, drift_scale, diff_scale = _kernel_weights(H, N, dt)
        alpha = np.ascontiguousarray(alpha, dtype=float)

        S = np.empty((n_paths, N + 1), dtype=float)
        V = np.empty((n_paths, N + 1), dtype=float)

        for start in range(0, n_paths, batch_size):
            end = min(n_paths, start + batch_size)
            m = end - start
            Z1 = rng.standard_normal((m, N))
            Z2 = rng.standard_normal((m, N))
            Sb = np.empty((m, N + 1), dtype=float)
            Vb = np.empty((m, N + 1), dtype=float)
            _rough_heston_kernel(S0, v0, dt, sqrt_dt, N, float(rho), float(r), float(q),
                                 float(kappa), float(theta), float(eta),
                                 alpha, float(drift_scale), float(diff_scale),
                                 Z1, Z2, Sb, Vb)
            S[start:end, :] = Sb
            V[start:end, :] = Vb

        return t, S, V

else:

    def _rough_heston_paths_numba(*args, **kwargs):  # pragma: no cover
        raise RuntimeError("Numba is not available")


def _resolve_use_numba(flag):
    """
    Resolve whether to use Numba acceleration.

    Parameters
    ----------
    flag : bool | None
        If None, use availability of Numba; else require both `flag` and availability.

    Returns
    -------
    bool
        True if Numba path should be used.
    """
    if flag is None:
        return _HAS_NUMBA
    return bool(flag) and _HAS_NUMBA


def rough_heston_paths(S0, v0, T, N, n_paths, H, kappa, theta, eta, rho,
                       r=0.0, q=0.0, seed=None, batch_size=1024, use_numba=None):
    """
    Rough Heston path generator with optional Numba acceleration.

    Parameters
    ----------
    See `_rough_heston_paths_python` for argument meanings.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (t, S, V) arrays as described above.
    """
    use_numba_flag = _resolve_use_numba(use_numba)
    if use_numba_flag:
        return _rough_heston_paths_numba(S0, v0, T, N, n_paths, H, kappa, theta, eta, rho,
                                         r=r, q=q, seed=seed, batch_size=batch_size)
    return _rough_heston_paths_python(S0, v0, T, N, n_paths, H, kappa, theta, eta, rho,
                                      r=r, q=q, seed=seed, batch_size=batch_size)


# ---------- Rough Heston parallel wrapper ----------

def _rough_heston_worker(args):
    """
    Worker to generate a batch of Rough Heston paths for parallel execution.

    Parameters
    ----------
    args : tuple
        Tuple with the same fields as `rough_heston_paths` but with `n_paths`
        replaced by the batch size for this task.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (t, S, V) for the batch.
    """
    (S0,v0,T,N,n_i,H,kappa,theta,eta,rho,r,q,seed,batch_size,use_numba) = args
    return rough_heston_paths(
        S0=S0, v0=v0, T=T, N=N, n_paths=n_i, H=H, kappa=kappa, theta=theta,
        eta=eta, rho=rho, r=r, q=q, seed=seed, batch_size=batch_size, use_numba=use_numba
    )

def _rough_heston_terminal_worker(args):
    """
    Worker returning only terminal prices for reduced IPC.

    Parameters
    ----------
    args : tuple
        Same as `_rough_heston_worker`.

    Returns
    -------
    np.ndarray
        Terminal prices S[:, -1] for the batch.
    """
    (S0,v0,T,N,n_i,H,kappa,theta,eta,rho,r,q,seed,batch_size,use_numba) = args
    t, S, V = rough_heston_paths(
        S0=S0, v0=v0, T=T, N=N, n_paths=n_i, H=H, kappa=kappa, theta=theta,
        eta=eta, rho=rho, r=r, q=q, seed=seed, batch_size=batch_size, use_numba=use_numba
    )
    return S[:, -1]

def rough_heston_paths_parallel(
    S0, v0, T, N, n_paths, H, kappa, theta, eta, rho,
    r=0.0, q=0.0, base_seed=12345, n_workers=4, batch_size=4096, use_numba=None
):
    """
    Generate Rough Heston paths in parallel across processes.

    Parameters
    ----------
    base_seed : int
        Base seed used to derive per-batch child seeds.
    n_workers : int
        Number of worker processes.
    batch_size : int
        Max paths per batch per worker.
    Other parameters are the same as `rough_heston_paths`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (t, S, V) for all paths concatenated in the same order.
    """
    sizes = _split_batches(n_paths, batch_size)
    seeds = _child_seeds(base_seed, len(sizes))
    use_numba_flag = _resolve_use_numba(use_numba)

    tasks = []
    for n_i, s in zip(sizes, seeds):
        tasks.append((S0, v0, T, N, n_i, H, kappa, theta, eta, rho, r, q, s, batch_size, use_numba_flag))

    with ProcessPoolExecutor(max_workers=int(n_workers)) as ex:
        outs = list(ex.map(_rough_heston_worker, tasks))

    # merge
    t = outs[0][0]
    S = np.vstack([o[1] for o in outs])
    V = np.vstack([o[2] for o in outs])
    return t, S, V

def rough_heston_paths_parallel_pool(
    executor,
    S0, v0, T, N, n_paths, H, kappa, theta, eta, rho,
    r=0.0, q=0.0, base_seed=12345, batch_size=4096, use_numba=None
):
    """
    Same as `rough_heston_paths_parallel` but reuses a provided executor.

    Parameters
    ----------
    executor : concurrent.futures.Executor
        Executor to submit jobs to (e.g., ProcessPoolExecutor).
    base_seed, batch_size : see `rough_heston_paths_parallel`.
    Other parameters are the same as `rough_heston_paths`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (t, S, V) for all paths concatenated.
    """
    sizes = _split_batches(n_paths, batch_size)
    seeds = _child_seeds(base_seed, len(sizes))
    use_numba_flag = _resolve_use_numba(use_numba)
    tasks = [(S0, v0, T, N, n_i, H, kappa, theta, eta, rho, r, q, s, batch_size, use_numba_flag) for n_i, s in zip(sizes, seeds)]
    outs = list(executor.map(_rough_heston_worker, tasks))
    t = outs[0][0]
    S = np.vstack([o[1] for o in outs])
    V = np.vstack([o[2] for o in outs])
    return t, S, V

def rough_heston_terminal_parallel_pool(
    executor,
    S0, v0, T, N, n_paths, H, kappa, theta, eta, rho,
    r=0.0, q=0.0, base_seed=12345, batch_size=4096, use_numba=None
):
    """
    Parallel simulator returning only terminal prices to minimize IPC.

    Parameters
    ----------
    executor : concurrent.futures.Executor
        Executor to submit jobs to.
    Other parameters are the same as `rough_heston_paths`.

    Returns
    -------
    np.ndarray
        1D array of terminal prices of length `n_paths`.
    """
    sizes = _split_batches(n_paths, batch_size)
    seeds = _child_seeds(base_seed, len(sizes))
    use_numba_flag = _resolve_use_numba(use_numba)
    tasks = [(S0, v0, T, N, n_i, H, kappa, theta, eta, rho, r, q, s, batch_size, use_numba_flag) for n_i, s in zip(sizes, seeds)]
    outs = list(executor.map(_rough_heston_terminal_worker, tasks))
    ST = np.concatenate(outs, axis=0)
    return ST


def rough_heston_euro_mc(S0, v0, K, T, N, n_paths, H, kappa, theta, eta, rho, r=0.0, q=0.0, option="call", seed=None, batch_size=1024):
    """
    Monte Carlo price of a European option under Rough Heston.

    Parameters
    ----------
    S0, v0, K, T, N, n_paths, H, kappa, theta, eta, rho : see `rough_heston_paths`
    r, q : float
        Risk-free rate and dividend yield.
    option : {"call", "put"}
        Option type.
    seed : int, optional
        RNG seed.
    batch_size : int, optional
        Paths per batch inside the simulator.

    Returns
    -------
    tuple[float, float]
        (price, stderr) discounted under risk-neutral measure.
    """
    option=_validate_option(option); K=_floor_pos(K)
    if N < 1 or n_paths < 1:
        raise ValueError("N and n_paths must be >= 1")
    if K <= 0.0:
        raise ValueError("K must be positive")
    t,S,V = rough_heston_paths(S0,v0,T,N,n_paths,H,kappa,theta,eta,rho,r,q,seed,batch_size)
    ST = S[:,-1]
    payoff = np.maximum(ST-K,0.0) if option=="call" else np.maximum(K-ST,0.0)
    DF = math.exp(-r*T); disc = DF*payoff
    price = float(np.mean(disc)); stderr = float(np.std(disc, ddof=1)/math.sqrt(S.shape[0]))
    return price, stderr


def check_put_call_parity_rough(
    S0: float, v0: float, r: float, T: float, rho: float,
    kappa: float, theta: float, eta: float, H: float,
    K: float, n_paths: int, N: int,
    *, q: float = 0.0, seed=None, batch_size: int = 4096, use_numba: Optional[bool] = None,
) -> Tuple[float, float, float]:
    """
    Check European put-call parity using the Rough Heston simulator.

    Parameters
    ----------
    S0, v0, r, T, rho, kappa, theta, eta, H, K, n_paths, N : see `rough_heston_paths`
    q : float, optional
        Dividend yield.
    seed : int, optional
        RNG seed.
    batch_size : int, optional
        Paths per batch.
    use_numba : bool | None
        Force Numba path if True (when available), else auto if None.

    Returns
    -------
    tuple[float, float, float]
        (residual, call_price, put_price) where
        residual = C - P - (S0*e^{-qT} - K*e^{-rT}).
    """
    _t, S, _V = rough_heston_paths(
        S0=S0, v0=v0, T=T, N=int(N), n_paths=int(n_paths), H=H,
        kappa=kappa, theta=theta, eta=eta, rho=rho,
        r=r, q=q, seed=seed, batch_size=batch_size, use_numba=use_numba,
    )
    ST = S[:, -1]
    disc = math.exp(-r * T)
    C = float(disc * np.maximum(ST - K, 0.0).mean())
    P = float(disc * np.maximum(K - ST, 0.0).mean())
    target = S0 * math.exp(-q * T) - K * math.exp(-r * T)
    residual = C - P - target
    return float(residual), float(C), float(P)


def rough_heston_delta_mc(
    S0: float,
    v0: float,
    K: float,
    T: float,
    N: int,
    n_paths: int,
    H: float,
    kappa: float,
    theta: float,
    eta: float,
    rho: float,
    r: float = 0.0,
    q: float = 0.0,
    option: str = "call",
    seed: Optional[int] = None,
    batch_size: int = 1024,
    method: str = "pathwise",
    h: float = 1e-4,
    use_numba: Optional[bool] = None,
) -> Tuple[float, float]:
    """
    Monte Carlo delta for a European option under Rough Heston.

    Parameters
    ----------
    S0, v0, K, T, N, n_paths, H, kappa, theta, eta, rho : see `rough_heston_paths`
    r, q : float, optional
        Risk-free rate and dividend yield.
    option : {"call", "put"}
        Payoff type.
    seed : int, optional
        RNG seed.
    batch_size : int, optional
        Paths per batch for path generation.
    method : {"pathwise", "bump"}
        - "pathwise": uses dS_T/dS_0 = S_T/S_0 for the log-Euler scheme
          (unbiased for calls/puts under this discretization).
        - "bump": common-random-numbers central difference on S0 with
          relative bump ``h``.
    h : float, optional
        Relative bump size for the bump method.
    use_numba : bool | None, optional
        Use Numba implementation when available if True or None.

    Returns
    -------
    tuple[float, float]
        (delta, stderr) as the MC estimate and its standard error.
    """
    option = _validate_option(option)
    S0 = _floor_pos(S0); K = _floor_pos(K); T = _floor_pos(T)
    N = int(N); n_paths = int(n_paths)
    if N < 1 or n_paths < 1:
        raise ValueError("N and n_paths must be >= 1")

    method_l = str(method).lower()
    if method_l not in ("pathwise", "bump"):
        raise ValueError("method must be 'pathwise' or 'bump'")

    if method_l == "pathwise":
        # Reuse the same paths used for pricing
        t, S, _V = rough_heston_paths(
            S0=S0, v0=v0, T=T, N=N, n_paths=n_paths, H=H,
            kappa=kappa, theta=theta, eta=eta, rho=rho,
            r=r, q=q, seed=seed, batch_size=batch_size, use_numba=use_numba,
        )
        ST = S[:, -1]
        # Pathwise derivative of payoff wrt S0: g'(ST) * (ST/S0)
        if option == "call":
            gprime = (ST > K).astype(float)
        else:
            gprime = -(ST < K).astype(float)
        DF = math.exp(-r * T)
        contrib = DF * gprime * (ST / S0)
        delta = float(np.mean(contrib))
        stderr = float(np.std(contrib, ddof=1) / math.sqrt(n_paths))
        return delta, stderr

    # Bump-and-reprice (common random numbers)
    hup = float(h)
    if not (hup > 0):
        raise ValueError("h must be > 0 for bump method")

    S_up = S0 * (1.0 + hup)
    S_dn = max(S0 * (1.0 - hup), 1e-12)

    # Generate once; reuse common random numbers so ST scales with S0
    _t, S, _V = rough_heston_paths(
        S0=S0, v0=v0, T=T, N=N, n_paths=n_paths, H=H,
        kappa=kappa, theta=theta, eta=eta, rho=rho,
        r=r, q=q, seed=seed, batch_size=batch_size, use_numba=use_numba,
    )
    ST = S[:, -1]
    scale_up = S_up / S0
    scale_dn = S_dn / S0
    ST_up = ST * scale_up
    ST_dn = ST * scale_dn

    if option == "call":
        payoff_up = np.maximum(ST_up - K, 0.0)
        payoff_dn = np.maximum(ST_dn - K, 0.0)
    else:
        payoff_up = np.maximum(K - ST_up, 0.0)
        payoff_dn = np.maximum(K - ST_dn, 0.0)

    DF = math.exp(-r * T)
    grad_per_path = DF * (payoff_up - payoff_dn) / (S_up - S_dn)
    delta = float(np.mean(grad_per_path))
    stderr = float(np.std(grad_per_path, ddof=1) / math.sqrt(n_paths))
    return delta, stderr


def validate_rough_heston(
    S0=100.0, v0=0.04, r=0.01, T=1.0, q=0.0,
    rho=-0.7, H=0.10, kappa=1.5, theta=0.04, eta=0.5,
    K=100.0,
    n_paths=20_000, N=1000, steps_grid=(500, 1000, 1500),
    seed=123, batch_size=4096, use_numba=None,
    sigma_mult_parity=3.0, sigma_mult_steps=3.0, verbose=True,
):
    """
    Validation for the log-Euler Rough Heston simulator in this module.

    Checks
    - Put-call parity within N sigma.
    - Martingale: E[e^{-rT} S_T] ≈ S0 e^{-qT}.
    - Step convergence across a grid of N values within combined CI.

    Returns
    - dict with metrics and pass flags.
    """
    # Single-run parity and martingale
    t, S, _V = rough_heston_paths(
        S0=S0, v0=v0, T=T, N=int(N), n_paths=int(n_paths), H=H,
        kappa=kappa, theta=theta, eta=eta, rho=rho,
        r=r, q=q, seed=seed, batch_size=batch_size, use_numba=use_numba,
    )
    ST = S[:, -1]
    disc = math.exp(-r * T)

    C = float(disc * np.maximum(ST - K, 0.0).mean())
    P = float(disc * np.maximum(K - ST, 0.0).mean())
    residual = C - P - (S0 * math.exp(-q * T) - K * math.exp(-r * T))
    se_parity = float(disc * ST.std(ddof=1) / math.sqrt(n_paths))
    parity_pass = abs(residual) <= sigma_mult_parity * se_parity

    mart_err = float(disc * ST.mean() - S0 * math.exp(-q * T))
    mart_pass = abs(mart_err) <= 3e-3 * S0

    # Step convergence CI over steps_grid
    prices = []
    ses = []
    for Ni in steps_grid:
        _t, Si, _Vi = rough_heston_paths(
            S0=S0, v0=v0, T=T, N=int(Ni), n_paths=int(n_paths), H=H,
            kappa=kappa, theta=theta, eta=eta, rho=rho,
            r=r, q=q, seed=seed, batch_size=batch_size, use_numba=use_numba,
        )
        STi = Si[:, -1]
        pay = disc * np.maximum(STi - K, 0.0)
        prices.append(float(pay.mean()))
        ses.append(float(pay.std(ddof=1) / math.sqrt(n_paths)))

    steps_pass = True
    max_excess = 0.0
    for i in range(1, len(steps_grid)):
        diff = abs(prices[i] - prices[i - 1])
        thresh = sigma_mult_steps * math.sqrt(ses[i]**2 + ses[i - 1]**2) + 0.01
        if diff > thresh:
            steps_pass = False
            excess = diff - thresh
            if excess > max_excess:
                max_excess = excess

    if verbose:
        print("[validate_rough_heston]")
        print(f"  Parity residual {residual:+.6f}  SE {se_parity:.6f}  PASS={parity_pass}")
        print(f"  Martingale err  {mart_err:+.6f}  PASS={mart_pass}")
        print(f"  Step grid {steps_grid} prices {[round(p,6) for p in prices]}")
        print(f"  Step conv PASS={steps_pass}  (max excess {max_excess:+.6f})")

    return dict(
        parity_residual=float(residual), parity_se=float(se_parity), parity_pass=bool(parity_pass),
        martingale_error=float(mart_err), martingale_pass=bool(mart_pass),
        steps_grid=list(steps_grid), step_prices=[float(p) for p in prices], steps_pass=bool(steps_pass),
    )


def calibrate_rough_heston(
    smiles: List[Tuple],
    metric: str = "iv",
    vega_weight: bool = True,
    bounds=((1e-4, 0.5), (0.05, 6.0), (1e-4, 0.5), (0.05, 3.0), (-0.999, -0.01), (0.02, 0.45)),
    x0=(0.04, 1.5, 0.04, 1.8, -0.7, 0.1),
    mc=dict(N=192, paths=12000, batch_size=1024, n_workers=4, use_numba=True),
    seed: int = 7777,
    n_workers: int = 4,
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
    smiles: list of (S0, r, q, T, strikes, mids, cp)
    """
    if n_workers is not None:
        mc = dict(mc)
        mc["n_workers"] = int(n_workers)
    else:
        mc = dict(mc)

    # Prefer compiled paths when available.
    if "use_numba" not in mc:
        mc["use_numba"] = True

    # Keep process pool busy: reduce batch size if it would yield <2 batches per worker.
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
        if metric == "iv":
            mkt_iv = np.array([_implied_vol(S0, K, T, r, q, p, cp) for K, p in zip(strikes, mids)])
        else:
            mkt_iv = None
        dat.append((S0, r, q, T, np.asarray(strikes, float), np.asarray(mids, float), cp, mkt_iv))

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
                weights[T] = w

    b = Bounds([b[0] for b in bounds], [b[1] for b in bounds])

    best = None
    rng = np.random.default_rng(2024)
    starts = [np.array(x0, dtype=float)]
    for _ in range(max(0, multistart - 1)):
        noise = np.array([0.2, 0.2, 0.2, 0.2, 0.05, 0.1]) * (rng.random(6) - 0.5)
        starts.append(np.clip(starts[0] * (1.0 + noise), b.lb, b.ub))

    best_hist = None
    for i, guess in enumerate(starts, 1):
        tag = f"RoughHeston #{i}"
        mon = _CalibMonitor(tag=tag, print_every=print_every, verbose=verbose)

        max_workers = int(mc.get("n_workers", 4))
        Executor = ThreadPoolExecutor if str(parallel_backend).lower().startswith("thread") else ProcessPoolExecutor
        with Executor(max_workers=max_workers) as ex:
            obj = lambda x: _rough_heston_objective(x, dat, metric, weights, mc, seed, ex, terminal_only)
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
    out = dict(v0=p[0], kappa=p[1], theta=p[2], eta=p[3], rho=p[4], H=p[5],
               obj=best.fun, success=best.success, nit=best.nit, history=best_hist)
    return out, best




import math
import time
from typing import List, Tuple, Optional

import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.special import ndtr
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

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


# ------------------------ Objective and calibration ------------------------

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

        if metric == "price":
            resid = model_px - mids
            if weights is not None and T in weights:
                w = np.asarray(weights[T], float)
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
            if weights is not None and T in weights:
                w = np.asarray(weights[T], float)
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
                weights[T] = w

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
