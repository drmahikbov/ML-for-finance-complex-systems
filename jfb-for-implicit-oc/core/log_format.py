"""
core.log_format
---------------
ANSI-colour helpers for the trainer's per-epoch console output.

`EpochColourizer` keeps rolling histories of loss / grad-norm / time and
colours each field by trend: red = divergence/NaN, yellow = degraded,
green = healthy, cyan = epoch number, magenta = lr drop events.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Callable, Optional

# ANSI escape codes
_RED     = "\033[1;31m"
_YELLOW  = "\033[1;33m"
_GREEN   = "\033[1;32m"
_CYAN    = "\033[1;36m"
_MAGENTA = "\033[1;35m"
_RESET   = "\033[0m"


def _wrap(code: str, body: str) -> str:
    return f"{code}{body}{_RESET}" if code else body


class EpochColourizer:
    """Rolling-history-aware per-field colourizer for epoch logs.

    Parameters
    ----------
    history_window
        How many past epochs to track for the rolling-median rules
        (used by ``loss``, ``grad_norm``, ``time``).
    """

    def __init__(self, history_window: int = 10):
        self._loss_hist = deque(maxlen=history_window)
        self._grad_hist = deque(maxlen=history_window)
        self._time_hist = deque(maxlen=history_window)
        self._cadj_hist = deque(maxlen=history_window)
        self._chjb_hist = deque(maxlen=history_window)
        self._best_loss = math.inf
        self._prev_lr: Optional[float] = None

    # ------------------------------------------------------------------
    # History upkeep (call exactly once per epoch, *after* colourize calls)
    # ------------------------------------------------------------------
    def update_history(
        self,
        *,
        loss: Optional[float] = None,
        grad_norm: Optional[float] = None,
        time_per_epoch: Optional[float] = None,
        lr: Optional[float] = None,
        cadj: Optional[float] = None,
        cHJB: Optional[float] = None,
    ) -> None:
        if loss is not None and math.isfinite(loss):
            self._loss_hist.append(loss)
            if loss < self._best_loss:
                self._best_loss = loss
        if grad_norm is not None and math.isfinite(grad_norm):
            self._grad_hist.append(grad_norm)
        if time_per_epoch is not None and math.isfinite(time_per_epoch):
            self._time_hist.append(time_per_epoch)
        if cadj is not None and math.isfinite(cadj):
            self._cadj_hist.append(abs(cadj))
        if cHJB is not None and math.isfinite(cHJB):
            self._chjb_hist.append(abs(cHJB))
        if lr is not None:
            self._prev_lr = lr

    # ------------------------------------------------------------------
    # Convenience: format a scalar with a colour determined by the
    # supplied predicates.  The first matching predicate wins, in the
    # order: fail (red) > warn (yellow) > ok (green).
    # ------------------------------------------------------------------
    @staticmethod
    def _format_field(
        value: float,
        fmt: str,
        *,
        fail: Optional[Callable[[float], bool]] = None,
        warn: Optional[Callable[[float], bool]] = None,
        ok: Optional[Callable[[float], bool]] = None,
    ) -> str:
        try:
            body = format(value, fmt)
        except (TypeError, ValueError):
            body = str(value)
        if fail and fail(value):
            return _wrap(_RED, body)
        if warn and warn(value):
            return _wrap(_YELLOW, body)
        if ok and ok(value):
            return _wrap(_GREEN, body)
        return body

    @staticmethod
    def _median(seq) -> Optional[float]:
        if not seq:
            return None
        s = sorted(seq)
        n = len(s)
        return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])

    # ------------------------------------------------------------------
    # Per-field rules
    # ------------------------------------------------------------------
    def epoch(self, epoch: int) -> str:
        """The leading ``Epoch NNN`` field is always cyan-bold."""
        return _wrap(_CYAN, f"Epoch {epoch:03d}")

    def loss(self, value: float) -> str:
        prev = self._loss_hist[-1] if self._loss_hist else None
        best = self._best_loss
        return self._format_field(
            value, ".3e",
            fail=lambda v: not math.isfinite(v) or (math.isfinite(best) and v > 10.0 * max(best, 1e-12)),
            warn=lambda v: prev is not None and v > 1.5 * prev,
            ok=lambda v: prev is not None and v < prev,
        )

    def cadj(self, value: float) -> str:
        med = self._median(self._cadj_hist)
        return self._format_field(
            value, ".2e",
            fail=lambda v: not math.isfinite(v) or (med is not None and abs(v) > 10.0 * med),
            warn=lambda v: med is not None and abs(v) > 3.0 * med,
            ok=lambda v: med is not None and abs(v) < med,
        )

    def cHJB(self, value: float) -> str:
        med = self._median(self._chjb_hist)
        return self._format_field(
            value, ".3e",
            fail=lambda v: not math.isfinite(v) or (med is not None and abs(v) > 10.0 * med),
            warn=lambda v: med is not None and abs(v) > 3.0 * med,
            ok=lambda v: med is not None and abs(v) < med,
        )

    def grad_norm(self, value: float) -> str:
        med = self._median(self._grad_hist)
        return self._format_field(
            value, ".2e",
            fail=lambda v: not math.isfinite(v) or (med is not None and v > 10.0 * med),
            warn=lambda v: med is not None and v > 3.0 * med,
            ok=lambda v: med is not None and v < med,
        )

    def fp_itrs(self, value: float, max_iters: int) -> str:
        cap = max(int(max_iters), 1)
        return self._format_field(
            float(value), ".0f",
            fail=lambda v: int(v) >= cap,
            warn=lambda v: int(v) >= 0.5 * cap,
            ok=lambda v: int(v) < 0.1 * cap,
        )

    def res_norm(self, value: float, tol: float) -> str:
        return self._format_field(
            value, ".3e",
            fail=lambda v: not math.isfinite(v) or v > 1.0,
            warn=lambda v: v > 100.0 * tol,
            ok=lambda v: v < tol,
        )

    def max_grad_H(self, value: float, fp_alpha: float) -> str:
        # FP-stability bound: |1 - alpha * L| < 1  =>  L < 2/alpha.
        # If max_grad_H exceeds 2/alpha we are in the divergent regime.
        bound = 2.0 / max(fp_alpha, 1e-12)
        return self._format_field(
            value, ".3e",
            fail=lambda v: not math.isfinite(v) or v > bound,
            warn=lambda v: v > 0.5 * bound,
        )

    def lr(self, value: float) -> str:
        if self._prev_lr is not None and value < self._prev_lr - 1e-15:
            return _wrap(_MAGENTA, format(value, ".3e"))
        return format(value, ".3e")

    def time(self, value: float) -> str:
        med = self._median(self._time_hist)
        return self._format_field(
            value, ".2f",
            fail=lambda v: med is not None and v > 5.0 * med,
            warn=lambda v: med is not None and v > 2.0 * med,
        )

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------
    @staticmethod
    def legend(
        *,
        fp_max_iters: int,
        fp_tol: float,
        fp_alpha: float,
    ) -> str:
        """Return a compact ANSI-colour legend explaining the per-epoch log.

        ``fp_max_iters``, ``fp_tol``, ``fp_alpha`` come from the ``ImplicitNetOC``
        attached to the trainer; they let us spell out the *actual* thresholds
        used by :meth:`fp_itrs`, :meth:`res_norm` and :meth:`max_grad_H` for
        this run instead of leaving them abstract.
        """
        cap = int(fp_max_iters)
        bound = 2.0 / max(float(fp_alpha), 1e-12)
        g = lambda s: _wrap(_GREEN,   s)
        y = lambda s: _wrap(_YELLOW,  s)
        r = lambda s: _wrap(_RED,     s)
        c = lambda s: _wrap(_CYAN,    s)
        m = lambda s: _wrap(_MAGENTA, s)

        lines = [
            "ANSI colour legend  (rolling window = 10 epochs):",
            f"  {c('cyan')}    Epoch number (always)",
            f"  {g('green')}   below recent median / better than previous",
            f"  {y('yellow')}  3x worse than recent median (warning)",
            f"  {r('red')}     catastrophic: NaN, ≥10x median, or hard threshold breached",
            f"  {m('magenta')} structural event (lr scheduler dropped lr)",
            "Per-field rules (only fields with rules are coloured):",
            f"  Loss        : {g('< prev')} | {y('> 1.5x prev')} | {r('> 10x best')} or NaN",
            f"  HJB / Adj   : {g('< median')} | {y('> 3x median')} | {r('> 10x median')}",
            f"  Grad        : {g('< median')} | {y('> 3x median')} | {r('> 10x median')}",
            f"  Time        : -        | {y('> 2x median')} | {r('> 5x median')}",
            f"  max_fp_itrs : {g('< ' + str(int(0.1 * cap)))} | {y('>= ' + str(int(0.5 * cap)))} | "
            f"{r('>= ' + str(cap) + '  (cap)')}",
            f"  res_norm    : {g('< ' + format(fp_tol, '.0e') + '  (tol)')} | "
            f"{y('> ' + format(100 * fp_tol, '.0e'))} | {r('> 1')}",
            f"  max_grad_H  : -        | {y('> ' + format(0.5 * bound, '.2g'))} | "
            f"{r('> ' + format(bound, '.2g') + '  (= 2/α, FP-divergent)')}",
            f"  lr          : {m('magenta when scheduler dropped lr')}",
        ]
        return "\n".join(lines)


__all__ = ["EpochColourizer"]
