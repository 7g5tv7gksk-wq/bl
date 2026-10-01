import os
import csv
import time
import logging
import logging.handlers
import queue
import threading
import requests
from flask import Flask
from curl_cffi import requests as c_requests
# =====================================================================
# LOGGING
# =====================================================================
# Logging is routed through a queue instead of writing to stdout directly
# from application threads. The application threads only enqueue records;
# the QueueListener performs the actual stream writes on its own thread.
_log_queue = queue.Queue(-1)
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
)
_queue_listener = logging.handlers.QueueListener(
    _log_queue,
    _stream_handler
)
_queue_listener.start()
_root_logger = logging.getLogger()
_root_logger.setLevel(logging.INFO)
# Avoid installing duplicate QueueHandlers if this module is reloaded.
if not any(
    isinstance(handler, logging.handlers.QueueHandler)
    and getattr(handler, "_markov_queue_handler", False)
    for handler in _root_logger.handlers
):
    _queue_handler = logging.handlers.QueueHandler(_log_queue)
    _queue_handler._markov_queue_handler = True
    _root_logger.addHandler(_queue_handler)
# =====================================================================
# CONFIGURATION
# =====================================================================
PAPER_TRADING = True
SOL_TRADE_SIZE = 0.1          # SOL per trade
# Pair quality filters
MIN_LIQUIDITY_USD = 3_000.0
MIN_MARKET_CAP = 10_000.0
MAX_MARKET_CAP = 250_000.0
MIN_5M_VOLUME = 500.0
MAX_MACRO_DRAWDOWN = -25.0    # Reject if 6h or 24h change worse than this
# Markov 2.0 parameters
STRIDE_BARS = 4               # Non-overlapping window size (4 × 5m = 20-min epochs)
MIN_WINDOWS = 12              # Min stride windows before Markov signal is trusted
ATR_BULL_MULT = 0.5           # Window return > +0.5×ATR% → BULL
ATR_BEAR_MULT = 0.5           # Window return < -0.5×ATR% → BEAR
MARKOV_THRESHOLD = 0.20       # P(bull) − P(bear) required to enter via Markov
MOMENTUM_THRESHOLD = 0.25     # Threshold for cold-start momentum fallback
# SOL macro regime
SOL_BEAR_H24 = -15.0
SOL_BULL_H24 = 10.0
SOL_MINT = "So11111111111111111111111111111111111111112"
# Position management
TAKE_PROFIT_PCT = 0.35
STOP_LOSS_PCT = 0.12
BLACKLIST_COOLDOWN = 7200
SECURITY_REJECT_COOLDOWN = 14400
MAX_POSITIONS = 1
LOOP_INTERVAL = 30
OHLCV_TTL = 300
OHLCV_RETRY_COOLDOWN = 45
TRACKER_STALE_SECONDS = 3600
POOL_RESOLVE_MAX_ATTEMPTS = 5
# Entry accuracy & cost model
MAX_ENTRY_DRIFT_PCT = 0.03
FEE_SLIPPAGE_PCT = 0.01
# Hard risk limits
MAX_DAILY_LOSS_SOL = 0.50
MAX_CONSECUTIVE_LOSSES = 3
MAX_API_ERRORS = 5
# Trade log
TRADE_LOG_PATH = "/tmp/trades.csv"
# Familiars
FAMILIARS_KEY = os.environ.get("FAMILIARS_API_KEY", "")
FAMILIARS_URL = "https://familiars.family"
# Jupiter
JUPITER_API_KEY = os.environ.get("JUPITER_API_KEY", "")
JUP_BASE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}
def jup_headers():
    headers = dict(JUP_BASE_HEADERS)
    if JUPITER_API_KEY:
        headers["x-api-key"] = JUPITER_API_KEY
    return headers
_jup_backoff = {"until": 0}
JUP_BACKOFF_SECONDS = 30
_gecko_backoff = {"until": 0}
GECKO_BACKOFF_SECONDS = 60
def gecko_available():
    return time.time() >= _gecko_backoff["until"]
def gecko_mark_limited():
    _gecko_backoff["until"] = time.time() + GECKO_BACKOFF_SECONDS
def jup_available():
    return time.time() >= _jup_backoff["until"]
def jup_mark_limited():
    _jup_backoff["until"] = time.time() + JUP_BACKOFF_SECONDS
# =====================================================================
# STATE
# =====================================================================
active_positions = {}
stopped_out_tokens = {}
security_rejected = {}
coin_trackers = {}
_pool_cache = {}
trade_stats = {
    "total_closed": 0,
    "wins": 0,
    "losses": 0,
    "net_sol_pnl": 0.0,
}
_sol_cache = {
    "state": "SIDEWAYS",
    "last_check": 0,
}
risk_state = {
    "consecutive_losses": 0,
    "daily_loss_sol": 0.0,
    "daily_loss_date": "",
    "kill_switch": False,
}
# Lock shared trading state because run_bot() and run_monitor()
# operate on these dictionaries from separate threads.
state_lock = threading.RLock()
BULL, BEAR, SIDEWAYS = 0, 1, 2
STATE_NAME = {
    BULL: "BULL",
    BEAR: "BEAR",
    SIDEWAYS: "SIDEWAYS",
}
# =====================================================================
# TRADE LOG
# =====================================================================
def _ensure_trade_log():
    try:
        if (
            not os.path.exists(TRADE_LOG_PATH)
            or os.path.getsize(TRADE_LOG_PATH) == 0
        ):
            with open(TRADE_LOG_PATH, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "utc",
                    "symbol",
                    "mint",
                    "outcome",
                    "entry_price",
                    "exit_price",
                    "gross_pnl_pct",
                    "net_pnl_pct",
                    "net_pnl_sol",
                    "signal_src",
                    "signal",
                ])
    except Exception as e:
        logging.warning(
            f"⚠️ [LOG] Could not create trade log: {e}"
        )
def log_trade(
    symbol,
    mint,
    outcome,
    entry_price,
    exit_price,
    signal_src,
    signal,
    trade_size=None,
):
    try:
        if entry_price <= 0:
            return
        gross = (exit_price - entry_price) / entry_price
        net = gross - 2 * FEE_SLIPPAGE_PCT
        if trade_size is None:
            trade_size = SOL_TRADE_SIZE
        net_sol = trade_size * net
        with open(
            TRADE_LOG_PATH,
            "a",
            newline="",
            encoding="utf-8"
        ) as f:
            csv.writer(f).writerow([
                time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ",
                    time.gmtime()
                ),
                symbol,
                mint,
                outcome,
                f"{entry_price:.10f}",
                f"{exit_price:.10f}",
                f"{gross * 100:.2f}",
                f"{net * 100:.2f}",
                f"{net_sol:.5f}",
                signal_src,
                f"{signal:.4f}",
            ])
    except Exception as e:
        logging.warning(
            f"⚠️ [LOG] Trade write failed: {e}"
        )
_ensure_trade_log()
# =====================================================================
# RISK ENGINE
# =====================================================================
def risk_check_daily_reset():
    today = time.strftime("%Y-%m-%d", time.gmtime())
    with state_lock:
        if risk_state["daily_loss_date"] != today:
            risk_state["daily_loss_date"] = today
            risk_state["daily_loss_sol"] = 0.0
            if risk_state["kill_switch"]:
                logging.info(
                    "🔓 [RISK] New UTC day — daily kill-switch cleared"
                )
                risk_state["kill_switch"] = False
def risk_on_loss(sol_lost):
    with state_lock:
        risk_state["consecutive_losses"] += 1
        risk_state["daily_loss_sol"] += abs(sol_lost)
        if risk_state["consecutive_losses"] >= MAX_CONSECUTIVE_LOSSES:
            logging.warning(
                f"⚠️ [RISK] "
                f"{risk_state['consecutive_losses']} consecutive losses — "
                f"trade size halved until next win"
            )
        if risk_state["daily_loss_sol"] >= MAX_DAILY_LOSS_SOL:
            risk_state["kill_switch"] = True
            logging.error(
                f"🛑 [RISK] Daily loss limit hit "
                f"({risk_state['daily_loss_sol']:.4f} SOL ≥ "
                f"{MAX_DAILY_LOSS_SOL} SOL) — "
                f"kill-switch latched, no new entries until UTC midnight"
            )
def risk_on_win():
    with state_lock:
        risk_state["consecutive_losses"] = 0
def effective_trade_size():
    with state_lock:
        if (
            risk_state["consecutive_losses"]
            >= MAX_CONSECUTIVE_LOSSES
        ):
            return SOL_TRADE_SIZE * 0.5
        return SOL_TRADE_SIZE
def risk_entry_ok():
    risk_check_daily_reset()
    with state_lock:
        return not risk_state["kill_switch"]
# =====================================================================
# FAMILIARS INTEGRATION
# =====================================================================
def familiars_post(kind, text, mint=None, signature=None):
    if not FAMILIARS_KEY:
        return
    payload = {
        "kind": kind,
        "text": str(text)[:500],
    }
    if mint:
        payload["mint"] = mint
    if signature:
        payload["signature"] = signature
    try:
        requests.post(
            f"{FAMILIARS_URL}/api/posts",
            headers={
                "Authorization": f"Bearer {FAMILIARS_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=5,
        )
    except Exception:
        pass
def familiars_limits():
    if not FAMILIARS_KEY:
        return {}
    try:
        r = requests.get(
            f"{FAMILIARS_URL}/api/agent/me",
            headers={
                "Authorization": f"Bearer {FAMILIARS_KEY}"
            },
            timeout=5,
        )
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, dict):
                return data.get("settings", {}) or {}
    except Exception:
        pass
    return {}
# =====================================================================
# LAYER 1 — SOL MACRO REGIME
# =====================================================================
def get_sol_regime():
    now = time.time()
    if now - _sol_cache["last_check"] < OHLCV_TTL:
        return _sol_cache["state"]
    if not jup_available():
        return _sol_cache["state"]
    try:
        r = requests.get(
            f"https://api.jup.ag/tokens/v2/search?query={SOL_MINT}",
            headers=jup_headers(),
            timeout=5,
        )
        if r.status_code == 429:
            jup_mark_limited()
            return _sol_cache["state"]
        if r.status_code != 200:
            return _sol_cache["state"]
        data = r.json()
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = (
                data.get("data")
                or data.get("tokens")
                or []
            )
        else:
            items = []
        token = next(
            (
                t for t in items
                if (
                    t.get("id")
                    or t.get("address")
                    or t.get("mint")
                ) == SOL_MINT
            ),
            None,
        )
        if token:
            stats24h = token.get("stats24h") or {}
            stats6h = token.get("stats6h") or {}
            stats1h = token.get("stats1h") or {}
            h24 = float(
                stats24h.get("priceChange") or 0
            )
            h6 = float(
                stats6h.get("priceChange") or 0
            )
            h1 = float(
                stats1h.get("priceChange") or 0
            )
            if h24 <= SOL_BEAR_H24 or (
                h6 <= -10 and h1 <= -5
            ):
                state = "BEAR"
            elif h24 >= SOL_BULL_H24 and h6 >= 5:
                state = "BULL"
            else:
                state = "SIDEWAYS"
            _sol_cache.update({
                "state": state,
                "last_check": now,
            })
            logging.info(
                f"🌐 [SOL MACRO] {state} | "
                f"24h={h24:+.1f}%  "
                f"6h={h6:+.1f}%  "
                f"1h={h1:+.1f}%"
            )
    except Exception as e:
        logging.error(
            f"❌ [SOL REGIME] {e}"
        )
    return _sol_cache["state"]
# =====================================================================
# OHLCV — GECKOTERMINAL
# =====================================================================
def fetch_ohlcv(pool_address, agg_min=5, limit=200):
    if not pool_address:
        return []
    if not gecko_available():
        return []
    url = (
        "https://api.geckoterminal.com/api/v2/"
        "networks/solana"
        f"/pools/{pool_address}/ohlcv/minute"
        f"?aggregate={agg_min}"
        f"&limit={limit}"
        "&currency=usd"
    )
    try:
        r = requests.get(
            url,
            headers={"Accept": "application/json"},
            timeout=8,
        )
        if r.status_code == 404:
            return []
        if r.status_code == 429:
            gecko_mark_limited()
            logging.warning(
                f"⚠️ [GECKO] Rate limited on "
                f"{pool_address[:8]}... backing off "
                f"{GECKO_BACKOFF_SECONDS}s"
            )
            return []
        if r.status_code != 200:
            return []
        payload = r.json()
        raw = (
            payload.get("data", {})
            .get("attributes", {})
            .get("ohlcv_list", [])
        )
        if not raw:
            return []
        candles = []
        for c in reversed(raw):
            if len(c) < 6:
                continue
            try:
                if c[1] is None or c[4] is None:
                    continue
                candles.append({
                    "ts": c[0],
                    "o": float(c[1]),
                    "h": float(c[2]),
                    "l": float(c[3]),
                    "c": float(c[4]),
                    "v": float(c[5]),
                })
            except (TypeError, ValueError):
                continue
        return candles
    except Exception as e:
        logging.error(
            f"❌ [GECKO OHLCV] "
            f"{pool_address[:8]}...: {e}"
        )
        return []
def resolve_pool_address(
    mint,
    graduated_pool_hint=None
):
    if graduated_pool_hint:
        return graduated_pool_hint, False
    with state_lock:
        if mint in _pool_cache:
            return _pool_cache[mint], False
    if not gecko_available():
        return None, True
    url = (
        "https://api.geckoterminal.com/api/v2/"
        f"networks/solana/tokens/{mint}/pools"
    )
    try:
        r = requests.get(
            url,
            headers={"Accept": "application/json"},
            timeout=8,
        )
        if r.status_code == 200:
            data = r.json().get("data", [])
            if data:
                pool_addr = (
                    data[0]
                    .get("attributes", {})
                    .get("address")
                )
                if pool_addr:
                    with state_lock:
                        _pool_cache[mint] = pool_addr
                    return pool_addr, False
            return None, False
        if r.status_code == 429:
            gecko_mark_limited()
            logging.warning(
                f"⚠️ [GECKO POOL] Rate limited resolving "
                f"pool for {mint[:8]}... backing off "
                f"{GECKO_BACKOFF_SECONDS}s"
            )
            return None, True
        return None, False
    except Exception as e:
        logging.error(
            f"❌ [GECKO POOL] "
            f"{mint[:8]}...: {e}"
        )
        return None, True
# =====================================================================
# MARKOV 2.0 ENGINE
# =====================================================================
def _atr_pct(candles, period=14):
    if len(candles) < period + 1:
        return 3.0
    trs = []
    for i in range(1, len(candles)):
        h = candles[i]["h"]
        l = candles[i]["l"]
        pc = candles[i - 1]["c"]
        if pc == 0:
            continue
        trs.append(
            max(
                h - l,
                abs(h - pc),
                abs(l - pc),
            ) / pc * 100
        )
    tail = trs[-period:]
    return (
        sum(tail) / len(tail)
        if tail
        else 3.0
    )
def _label_states(
    candles,
    stride,
    bull_mult,
    bear_mult
):
    atr = _atr_pct(candles)
    bull_thresh = atr * bull_mult
    bear_thresh = -atr * bear_mult
    states = []
    for i in range(
        0,
        len(candles) - stride + 1,
        stride
    ):
        w = candles[i:i + stride]
        o = w[0]["o"]
        c = w[-1]["c"]
        if o == 0:
            continue
        ret = (c - o) / o * 100
        if ret >= bull_thresh:
            states.append(BULL)
        elif ret <= bear_thresh:
            states.append(BEAR)
        else:
            states.append(SIDEWAYS)
    return states, bull_thresh, bear_thresh
def _build_matrix(states):
    counts = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
    for i in range(len(states) - 1):
        current = states[i]
        next_state = states[i + 1]
        counts[current][next_state] += 1
    matrix = []
    for row in counts:
        total = sum(row)
        matrix.append([
            v / total if total else 1 / 3
            for v in row
        ])
    stickiness = {
        STATE_NAME[s]: round(matrix[s][s], 3)
        for s in (BULL, BEAR, SIDEWAYS)
    }
    return matrix, stickiness
def _verify_labels(states, candles, stride):
    if not states:
        return False
    errors = 0
    indexes = {
        0,
        len(states) // 2,
        len(states) - 1,
    }
    for idx in indexes:
        if idx < 0 or idx >= len(states):
            continue
        start = idx * stride
        w = candles[
            start:start + stride
        ]
        if not w or w[0]["o"] == 0:
            continue
        ret = (
            (w[-1]["c"] - w[0]["o"])
            / w[0]["o"]
            * 100
        )
        if ret > 5 and states[idx] == BEAR:
            errors += 1
        if ret < -5 and states[idx] == BULL:
            errors += 1
    if errors:
        logging.warning(
            f"⚠️ [MARKOV FIX2] "
            f"{errors} label anomaly(s) — "
            f"matrix built but confidence is lower"
        )
    return errors == 0
def _markov_signal(matrix, current_state):
    if not matrix:
        return None
    if current_state not in (
        BULL,
        BEAR,
        SIDEWAYS,
    ):
        return None
    return round(
        matrix[current_state][BULL]
        - matrix[current_state][BEAR],
        4,
    )
# =====================================================================
# PER-COIN MARKOV TRACKER
# =====================================================================
class CoinMarkovTracker:
    def __init__(
        self,
        mint,
        symbol,
        graduated_pool_hint=None
    ):
        self.mint = mint
        self.symbol = symbol
        self.graduated_pool_hint = graduated_pool_hint
        self.pool_address = None
        self.pool_last_try = 0
        self.pool_resolve_attempts = 0
        self.pool_abandoned = False
        self.candles = []
        self.states = []
        self.matrix = None
        self.stickiness = None
        self.signal = None
        self.current_state = SIDEWAYS
        self.windows = 0
        self.last_fetch = 0
        self.verified = False
        self.last_seen = time.time()
    @property
    def ready(self):
        return (
            self.windows >= MIN_WINDOWS
            and self.signal is not None
        )
    @property
    def confidence(self):
        if self.windows < MIN_WINDOWS:
            return 0.0
        return min(
            (
                self.windows - MIN_WINDOWS
            ) / (MIN_WINDOWS * 3),
            1.0,
        )
    def refresh(self):
        if self.pool_abandoned:
            return
        now = time.time()
        if not self.pool_address:
            if (
                now - self.pool_last_try
                < OHLCV_RETRY_COOLDOWN
            ):
                return
            self.pool_last_try = now
            addr, was_rate_limited = (
                resolve_pool_address(
                    self.mint,
                    self.graduated_pool_hint,
                )
            )
            if addr:
                self.pool_address = addr
            else:
                if not was_rate_limited:
                    self.pool_resolve_attempts += 1
                    if (
                        self.pool_resolve_attempts
                        >= POOL_RESOLVE_MAX_ATTEMPTS
                    ):
                        self.pool_abandoned = True
                        logging.info(
                            f"📉 [POOL] Giving up on "
                            f"${self.symbol} — no pool after "
                            f"{self.pool_resolve_attempts} checks, "
                            f"momentum-only"
                        )
                return
        ttl = (
            OHLCV_TTL
            if self.candles
            else OHLCV_RETRY_COOLDOWN
        )
        if now - self.last_fetch < ttl:
            return
        candles = fetch_ohlcv(
            self.pool_address
        )
        self.last_fetch = now
        if not candles:
            return
        self.candles = candles
        (
            states,
            bull_t,
            bear_t,
        ) = _label_states(
            candles,
            STRIDE_BARS,
            ATR_BULL_MULT,
            ATR_BEAR_MULT,
        )
        if len(states) < 3:
            return
        self.verified = _verify_labels(
            states,
            candles,
            STRIDE_BARS,
        )
        self.states = states
        self.windows = len(states)
        (
            self.matrix,
            self.stickiness,
        ) = _build_matrix(states)
        self.current_state = states[-1]
        self.signal = _markov_signal(
            self.matrix,
            self.current_state,
        )
        logging.info(
            f"📈 [MARKOV] ${self.symbol} | "
            f"Windows={self.windows} | "
            f"State={STATE_NAME[self.current_state]} | "
            f"S={self.signal:+.4f} | "
            f"Conf={self.confidence:.0%} | "
            f"Stick={self.stickiness} | "
            f"Thresh=[+{bull_t:.2f}% / "
            f"{bear_t:.2f}%] | "
            f"Labels={'✅' if self.verified else '⚠️'}"
        )
# =====================================================================
# CANDIDATE DISCOVERY — JUPITER TOKEN API V2
# =====================================================================
def fetch_jupiter_candidates():
    token_map = {}
    source_report = []
    sources = [
        (
            "recent",
            "https://api.jup.ag/tokens/v2/recent",
        ),
        (
            "trending5m",
            "https://api.jup.ag/tokens/v2/toptrending/5m",
        ),
    ]
    for label, url in sources:
        if not jup_available():
            source_report.append(
                (label, 0, "skipped(backoff)")
            )
            continue
        try:
            r = requests.get(
                url,
                headers=jup_headers(),
                timeout=5,
            )
            if r.status_code == 200:
                before = len(token_map)
                data = r.json()
                if isinstance(data, list):
                    items = data
                elif isinstance(data, dict):
                    items = (
                        data.get("data")
                        or data.get("tokens")
                        or []
                    )
                else:
                    items = []
                for item in items[:5]:
                    if not isinstance(item, dict):
                        continue
                    addr = (
                        item.get("address")
                        or item.get("mint")
                        or item.get("id")
                    )
                    if addr and addr not in token_map:
                        token_map[addr] = item
                added = len(token_map) - before
                if added:
                    note = "ok"
                elif isinstance(data, dict):
                    note = (
                        f"0 items, keys="
                        f"{list(data.keys())}"
                    )
                else:
                    note = "0 items"
                source_report.append(
                    (label, added, note)
                )
            elif r.status_code == 429:
                jup_mark_limited()
                source_report.append(
                    (
                        label,
                        0,
                        "HTTP 429→backoff",
                    )
                )
            else:
                source_report.append(
                    (
                        label,
                        0,
                        f"HTTP {r.status_code}",
                    )
                )
        except Exception as e:
            source_report.append(
                (
                    label,
                    0,
                    f"exc: {e}",
                )
            )
    breakdown = " | ".join(
        f"{label}:{count}({note})"
        for label, count, note
        in source_report
    )
    logging.info(
        f"📡 [SCRAPER] "
        f"{len(token_map)} candidate tokens | "
        f"{breakdown}"
    )
    return token_map
# =====================================================================
# SAFE NUMERIC HELPERS
# =====================================================================
def safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
def safe_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default
# =====================================================================
# TOKEN FILTERS
# =====================================================================
def validate_token(token):
    if not isinstance(token, dict):
        return False
    liq = safe_float(
        token.get("liquidity")
    )
    if liq < MIN_LIQUIDITY_USD:
        return False
    mc = safe_float(
        token.get("mcap")
        or token.get("fdv")
    )
    if (
        mc < MIN_MARKET_CAP
        or mc > MAX_MARKET_CAP
    ):
        return False
    stats5m = token.get("stats5m") or {}
    stats6h = token.get("stats6h") or {}
    stats24h = token.get("stats24h") or {}
    vol5m = (
        safe_float(stats5m.get("buyVolume"))
        + safe_float(stats5m.get("sellVolume"))
    )
    if vol5m < MIN_5M_VOLUME:
        return False
    if (
        safe_float(stats6h.get("priceChange"))
        < MAX_MACRO_DRAWDOWN
    ):
        return False
    if (
        safe_float(stats24h.get("priceChange"))
        < MAX_MACRO_DRAWDOWN
    ):
        return False
    buys = safe_int(
        stats5m.get("numBuys")
    )
    sells = safe_int(
        stats5m.get("numSells")
    )
    if (
        buys + sells < 12
        or buys < sells * 1.2
    ):
        return False
    return True
# =====================================================================
# LAYER 2 — MOMENTUM SIGNAL
# =====================================================================
def compute_momentum_signal(token):
    stats5m = token.get("stats5m") or {}
    stats1h = token.get("stats1h") or {}
    stats6h = token.get("stats6h") or {}
    m5 = safe_float(
        stats5m.get("priceChange")
    )
    h1 = safe_float(
        stats1h.get("priceChange")
    )
    h6 = safe_float(
        stats6h.get("priceChange")
    )
    if h1 > 50 or m5 > 30:
        return None
    v_m5 = m5 / 5.0
    v_h1 = h1 / 60.0
    v_h6 = h6 / 360.0
    delta_v = v_m5 - v_h1
    stability = (
        1.0
        if abs(v_h1 - v_h6) < 0.5
        else 0.5
    )
    signal = (
        delta_v * 0.6
        + v_m5 * 0.4
    ) * stability
    vol5m = (
        safe_float(stats5m.get("buyVolume"))
        + safe_float(stats5m.get("sellVolume"))
    )
    vol_wt = min(
        max(vol5m / 1000.0, 0.5),
        1.5,
    )
    return round(
        signal * vol_wt,
        2,
    )
# =====================================================================
# SECURITY CHECKS
# =====================================================================
def check_gmgn_batch(mints):
    """
    Evaluates liquidity and basic liquidity/FDV safety for token mints
    using DexScreener.
    Returns:
        dict mapping mint -> True (passed) / False (failed)
    """
    if not mints:
        return {}
    results = {
        mint: False
        for mint in mints
    }
    # DexScreener supports multiple token addresses in this endpoint.
    # A smaller batch size is intentionally used to reduce request size.
    chunk_size = 10
    mint_chunks = [
        mints[i:i + chunk_size]
        for i in range(
            0,
            len(mints),
            chunk_size,
        )
    ]
    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36"
        )
    }
    for chunk in mint_chunks:
        joined_mints = ",".join(chunk)
        url = (
            "https://api.dexscreener.com/"
            f"latest/dex/tokens/{joined_mints}"
        )
        max_retries = 3
        res = None
        for attempt in range(max_retries):
            try:
                time.sleep(0.5)
                res = requests.get(
                    url,
                    headers=headers,
                    timeout=5,
                )
                if res.status_code == 200:
                    break
                if res.status_code == 429:
                    wait_time = (
                        attempt + 1
                    ) * 3
                    logging.warning(
                        f"[BATCH WAIT] "
                        f"DexScreener 429 rate limit. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logging.warning(
                        f"[BATCH FAIL] "
                        f"Non-200 Status "
                        f"({res.status_code}) "
                        f"from DexScreener"
                    )
                    break
            except Exception as e:
                logging.warning(
                    f"[BATCH ERROR] "
                    f"Request failed: {e}"
                )
                break
        if not res or res.status_code != 200:
            continue
        try:
            data = res.json()
        except Exception as e:
            logging.warning(
                f"[BATCH ERROR] "
                f"Invalid JSON response: {e}"
            )
            continue
        pairs = data.get("pairs") or []
        pairs_by_mint = {}
        for pair in pairs:
            if not isinstance(pair, dict):
                continue
            base_token = (
                pair.get("baseToken")
                or {}
            )
            base_addr = base_token.get(
                "address"
            )
            if (
                base_addr
                and base_addr in chunk
            ):
                pairs_by_mint.setdefault(
                    base_addr,
                    []
                ).append(pair)
        for mint in chunk:
            token_pairs = pairs_by_mint.get(
                mint,
                []
            )
            if not token_pairs:
                logging.info(
                    f"[SECURITY FAIL] "
                    f"{mint[:8]}... "
                    f"No active pair found "
                    f"on DexScreener"
                )
                continue
            main_pair = token_pairs[0]
            liquidity_data = (
                main_pair.get("liquidity")
                or {}
            )
            liquidity = safe_float(
                liquidity_data.get("usd")
            )
            fdv = safe_float(
                main_pair.get("fdv")
            )
            if liquidity < MIN_LIQUIDITY_USD:
                logging.info(
                    f"[SECURITY FAIL] "
                    f"{mint[:8]}... Low Liquidity "
                    f"(${liquidity:,.0f} < "
                    f"${MIN_LIQUIDITY_USD:,.0f})"
                )
                continue
            if (
                fdv > 0
                and (liquidity / fdv) < 0.01
            ):
                logging.info(
                    f"[SECURITY FAIL] "
                    f"{mint[:8]}... Thin "
                    f"Liquidity Ratio "
                    f"({liquidity / fdv:.2%})"
                )
                continue
            logging.info(
                f"[SECURITY PASS] "
                f"{mint[:8]}... | "
                f"Liquidity: "
                f"${liquidity:,.0f}"
            )
            results[mint] = True
    return results
def check_top_holders(mint):
    """
    Checks whether Solana returns token largest-account data.
    NOTE:
    getTokenLargestAccounts() returns token-account balances,
    not a percentage of total supply. This function therefore does
    not make a concentration decision; it only verifies that the
    RPC endpoint can return holder data.
    """
    rpc_url = (
        "https://api.mainnet-beta.solana.com"
    )
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTokenLargestAccounts",
        "params": [mint],
    }
    try:
        rpc_response = requests.post(
            rpc_url,
            json=payload,
            timeout=5,
        )
        if rpc_response.status_code != 200:
            logging.warning(
                f"[SECURITY RPC] "
                f"{mint[:8]}... HTTP "
                f"{rpc_response.status_code}"
            )
            return False
        rpc_res = rpc_response.json()
        accounts = (
            rpc_res
            .get("result", {})
            .get("value", [])
        )
        if accounts:
            top_10_sum = sum(
                safe_float(
                    acc.get("uiAmount")
                )
                for acc in accounts[:10]
            )
            logging.info(
                f"[DEBUG RPC] "
                f"{mint[:8]}... Top 10 Accounts "
                f"Total: {top_10_sum}"
            )
            return True
        logging.warning(
            f"[SECURITY RPC] "
            f"{mint[:8]}... No holder data"
        )
        return False
    except Exception as e:
        logging.warning(
            f"[SECURITY EXCEPTION] {e}"
        )
        return False
def check_rugcheck(mint):
    try:
        r = requests.get(
            f"https://api.rugcheck.xyz/v1/"
            f"tokens/{mint}/report/summary",
            timeout=5,
        )
        if r.status_code != 200:
            return False
        data = r.json()
        if data.get("riskLevel") in (
            "Danger",
            "High",
        ):
            return False
        bad = {
            "Single holder ownership",
            "High holder concentration",
            "Mint Authority Enabled",
            "Freeze Authority Enabled",
        }
        for risk in data.get("risks", []):
            if not isinstance(risk, dict):
                continue
            if risk.get("name") in bad:
                return False
        return True
    except Exception:
        return False
# =====================================================================
# REAL-TIME PRICE
# =====================================================================
def get_price(mint):
    # 1. Jupiter Primary Price Endpoint
    if jup_available():
        try:
            r = requests.get(
                f"https://api.jup.ag/price/v2?ids={mint}",
                headers=jup_headers(),
                timeout=2,
            )
            if r.status_code == 200:
                data = r.json()
                p = (
                    data
                    .get("data", {})
                    .get(mint, {})
                    .get("price")
                )
                price = safe_float(
                    p,
                    default=0.0,
                )
                if price > 0:
                    return price
            elif r.status_code == 429:
                jup_mark_limited()
        except Exception:
            pass
    # 2. Jupiter Search API fallback
    if jup_available():
        try:
            r = requests.get(
                "https://api.jup.ag/tokens/v2/"
                f"search?query={mint}",
                headers=jup_headers(),
                timeout=3,
            )
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list):
                    items = data
                elif isinstance(data, dict):
                    items = (
                        data.get("data")
                        or data.get("tokens")
                        or []
                    )
                else:
                    items = []
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    address = (
                        item.get("address")
                        or item.get("mint")
                        or item.get("id")
                    )
                    if address != mint:
                        continue
                    stats24h = (
                        item.get("stats24h")
                        or {}
                    )
                    p = (
                        item.get("price")
                        or item.get("usdPrice")
                        or stats24h.get("price")
                    )
                    price = safe_float(
                        p,
                        default=0.0,
                    )
                    if price > 0:
                        return price
            elif r.status_code == 429:
                jup_mark_limited()
        except Exception:
            pass
    # 3. DexScreener emergency fallback
    try:
        r = requests.get(
            "https://api.dexscreener.com/"
            f"latest/dex/tokens/{mint}",
            timeout=3,
        )
        if r.status_code == 200:
            pairs = (
                r.json().get("pairs")
                or []
            )
            if pairs:
                price = safe_float(
                    pairs[0].get("priceUsd"),
                    default=0.0,
                )
                if price > 0:
                    return price
    except Exception:
        pass
    return None
# =====================================================================
# POSITION MONITOR — 1-second background thread
# =====================================================================
def run_monitor():
    logging.info(
        "⚡ [MONITOR] Position monitor started (1s loop)"
    )
    heartbeat = 0
    while True:
        heartbeat += 1
        try:
            with state_lock:
                position_mints = list(
                    active_positions.keys()
                )
            if position_mints:
                to_close = []
                pnl_lines = []
                for mint in position_mints:
                    with state_lock:
                        info = active_positions.get(mint)
                    if not info:
                        continue
                    price = get_price(mint)
                    entry = safe_float(
                        info.get("entry_price")
                    )
                    if not price or entry <= 0:
                        continue
                    symbol = info.get(
                        "symbol",
                        "UNKNOWN",
                    )
                    gross = (
                        price - entry
                    ) / entry
                    net = (
                        gross
                        - 2 * FEE_SLIPPAGE_PCT
                    )
                    size = safe_float(
                        info.get(
                            "trade_size",
                            SOL_TRADE_SIZE,
                        ),
                        SOL_TRADE_SIZE,
                    )
                    pnl_lines.append(
                        f"${symbol} "
                        f"{gross * 100:+.1f}% gross / "
                        f"{net * 100:+.1f}% net"
                    )
                    # -------------------------------------------------
                    # TAKE PROFIT
                    # -------------------------------------------------
                    if gross >= TAKE_PROFIT_PCT:
                        net_sol = size * net
                        with state_lock:
                            trade_stats[
                                "total_closed"
                            ] += 1
                            trade_stats[
                                "wins"
                            ] += 1
                            trade_stats[
                                "net_sol_pnl"
                            ] += net_sol
                            total_closed = (
                                trade_stats[
                                    "total_closed"
                                ]
                            )
                            wins = (
                                trade_stats[
                                    "wins"
                                ]
                            )
                            total_pnl = (
                                trade_stats[
                                    "net_sol_pnl"
                                ]
                            )
                        wr = (
                            wins
                            / total_closed
                        ) * 100
                        logging.info(
                            f"🎯 [TP] ${symbol} "
                            f"gross +{gross * 100:.1f}% | "
                            f"net {net * 100:.1f}% | "
                            f"+{net_sol:.4f} SOL | "
                            f"WR={wr:.1f}% | "
                            f"Net={total_pnl:+.4f} SOL"
                        )
                        risk_on_win()
                        log_trade(
                            symbol,
                            mint,
                            "TP",
                            entry,
                            price,
                            info.get(
                                "signal_src",
                                "?"
                            ),
                            safe_float(
                                info.get(
                                    "signal",
                                    0
                                )
                            ),
                            trade_size=size,
                        )
                        try:
                            familiars_post(
                                "trade",
                                (
                                    f"TP "
                                    f"+{gross * 100:.1f}% "
                                    f"(net "
                                    f"{net * 100:.1f}%) "
                                    f"on ${symbol}. "
                                    f"Net="
                                    f"{total_pnl:+.4f} SOL"
                                ),
                                mint=mint,
                            )
                        except Exception as post_err:
                            logging.error(
                                f"⚠️ [MONITOR] "
                                f"TP webhook error: "
                                f"{post_err}"
                            )
                        to_close.append(mint)
                    # -------------------------------------------------
                    # STOP LOSS
                    # -------------------------------------------------
                    elif gross <= -STOP_LOSS_PCT:
                        net_sol = size * net
                        with state_lock:
                            trade_stats[
                                "total_closed"
                            ] += 1
                            trade_stats[
                                "losses"
                            ] += 1
                            trade_stats[
                                "net_sol_pnl"
                            ] += net_sol
                            total_closed = (
                                trade_stats[
                                    "total_closed"
                                ]
                            )
                            wins = (
                                trade_stats[
                                    "wins"
                                ]
                            )
                            total_pnl = (
                                trade_stats[
                                    "net_sol_pnl"
                                ]
                            )
                        wr = (
                            wins
                            / total_closed
                        ) * 100
                        logging.info(
                            f"🔴 [SL] ${symbol} "
                            f"gross {gross * 100:.1f}% | "
                            f"net {net * 100:.1f}% | "
                            f"{net_sol:.4f} SOL | "
                            f"WR={wr:.1f}% | "
                            f"Net={total_pnl:+.4f} SOL"
                        )
                        risk_on_loss(
                            abs(net_sol)
                        )
                        log_trade(
                            symbol,
                            mint,
                            "SL",
                            entry,
                            price,
                            info.get(
                                "signal_src",
                                "?"
                            ),
                            safe_float(
                                info.get(
                                    "signal",
                                    0
                                )
                            ),
                            trade_size=size,
                        )
                        try:
                            familiars_post(
                                "trade",
                                (
                                    f"SL "
                                    f"{gross * 100:.1f}% "
                                    f"(net "
                                    f"{net * 100:.1f}%) "
                                    f"on ${symbol}. "
                                    f"Blacklisting 2h."
                                ),
                                mint=mint,
                            )
                        except Exception as post_err:
                            logging.error(
                                f"⚠️ [MONITOR] "
                                f"SL webhook error: "
                                f"{post_err}"
                            )
                        with state_lock:
                            stopped_out_tokens[
                                mint
                            ] = time.time()
                        to_close.append(mint)
                # -----------------------------------------------------
                # Clean up closed positions
                # -----------------------------------------------------
                if to_close:
                    with state_lock:
                        for mint in to_close:
                            active_positions.pop(
                                mint,
                                None
                            )
                            coin_trackers.pop(
                                mint,
                                None
                            )
                if (
                    pnl_lines
                    and heartbeat % 10 == 0
                ):
                    logging.info(
                        "📟 [PNL] "
                        + " | ".join(
                            pnl_lines
                        )
                    )
        except Exception as e:
            logging.error(
                f"❌ [MONITOR] {e}"
            )
        time.sleep(1)
# =====================================================================
# MAIN TRADING LOOP
# =====================================================================
def run_bot():
    logging.info(
        f"🚀 [BOT] Markov 2.0 Engine active | "
        f"Paper={PAPER_TRADING} | "
        f"Stride={STRIDE_BARS}×5m | "
        f"MinWindows={MIN_WINDOWS}"
    )
    while True:
        cycle_start = time.time()
        try:
            # ---------------------------------------------------------
            # Position limit
            # ---------------------------------------------------------
            with state_lock:
                current_positions = len(
                    active_positions
                )
            if current_positions >= MAX_POSITIONS:
                time.sleep(
                    LOOP_INTERVAL
                )
                continue
            # ---------------------------------------------------------
            # SOL macro regime
            # ---------------------------------------------------------
            sol_regime = get_sol_regime()
            if sol_regime == "BEAR":
                logging.info(
                    "🚫 [MACRO] SOL=BEAR — "
                    "sitting out this cycle"
                )
                time.sleep(
                    LOOP_INTERVAL
                )
                continue
            # ---------------------------------------------------------
            # Remove stale trackers
            # ---------------------------------------------------------
            now = time.time()
            with state_lock:
                stale = [
                    mint
                    for mint, tracker
                    in coin_trackers.items()
                    if (
                        mint not in active_positions
                        and (
                            tracker.pool_abandoned
                            or (
                                now
                                - tracker.last_seen
                                > TRACKER_STALE_SECONDS
                            )
                        )
                    )
                ]
                for mint in stale:
                    coin_trackers.pop(
                        mint,
                        None
                    )
            if stale:
                logging.info(
                    f"🧹 [PRUNE] "
                    f"Dropped {len(stale)} "
                    f"stale tracker(s)"
                )
            # ---------------------------------------------------------
            # Refresh existing trackers
            # ---------------------------------------------------------
            with state_lock:
                trackers_snapshot = list(
                    coin_trackers.values()
                )
            for tracker in trackers_snapshot:
                tracker.refresh()
            with state_lock:
                tracking_count = len(
                    coin_trackers
                )
                active_count = len(
                    active_positions
                )
            logging.info(
                f"🔎 [SCAN] SOL={sol_regime} | "
                f"Tracking={tracking_count} coins | "
                f"Active={active_count}"
            )
            # ---------------------------------------------------------
            # Candidate discovery
            # ---------------------------------------------------------
            token_map = (
                fetch_jupiter_candidates()
            )
            mints = list(
                token_map.keys()
            )
            # ---------------------------------------------------------
            # Batch security check
            # ---------------------------------------------------------
            security_results = (
                check_gmgn_batch(mints)
            )
            candidates = []
            tally = {
                "total": len(mints),
                "in_position_or_blacklist": 0,
                "security_blacklist_skip": 0,
                "no_token_data": 0,
                "failed_filters": 0,
                "security_rejected": 0,
                "no_signal": 0,
                "below_threshold": 0,
                "qualified": 0,
            }
            # =========================================================
            # TOKEN EVALUATION
            # =========================================================
            for mint in mints:
                with state_lock:
                    already_active = (
                        mint
                        in active_positions
                    )
                    stopped_time = (
                        stopped_out_tokens.get(
                            mint
                        )
                    )
                if already_active:
                    tally[
                        "in_position_or_blacklist"
                    ] += 1
                    continue
                # -----------------------------------------------------
                # Stop-loss blacklist
                # -----------------------------------------------------
                if stopped_time is not None:
                    if (
                        time.time()
                        - stopped_time
                        < BLACKLIST_COOLDOWN
                    ):
                        tally[
                            "in_position_or_blacklist"
                        ] += 1
                        continue
                    with state_lock:
                        stopped_out_tokens.pop(
                            mint,
                            None
                        )
                # -----------------------------------------------------
                # Token data
                # -----------------------------------------------------
                token = token_map.get(
                    mint
                )
                if not token:
                    tally[
                        "no_token_data"
                    ] += 1
                    continue
                # -----------------------------------------------------
                # Basic token filters
                # -----------------------------------------------------
                if not validate_token(token):
                    tally[
                        "failed_filters"
                    ] += 1
                    continue
                symbol = token.get(
                    "symbol",
                    "?"
                )
                graduated_pool_hint = (
                    token.get(
                        "graduatedPool"
                    )
                )
                price = safe_float(
                    token.get("usdPrice")
                )
                mc = safe_float(
                    token.get("mcap")
                    or token.get("fdv")
                )
                # -----------------------------------------------------
                # Security rejection cooldown
                # -----------------------------------------------------
                with state_lock:
                    security_reject_time = (
                        security_rejected.get(
                            mint
                        )
                    )
                if security_reject_time is not None:
                    if (
                        time.time()
                        - security_reject_time
                        < SECURITY_REJECT_COOLDOWN
                    ):
                        tally[
                            "security_blacklist_skip"
                        ] += 1
                        continue
                    with state_lock:
                        security_rejected.pop(
                            mint,
                            None
                        )
                # -----------------------------------------------------
                # RugCheck
                # -----------------------------------------------------
                if not check_rugcheck(mint):
                    logging.info(
                        f"🛡️ [REJECTED] "
                        f"${symbol} — "
                        f"RugCheck fail"
                    )
                    tally[
                        "security_rejected"
                    ] += 1
                    with state_lock:
                        security_rejected[
                            mint
                        ] = time.time()
                    continue
                # -----------------------------------------------------
                # DexScreener security check
                # -----------------------------------------------------
                if not security_results.get(
                    mint,
                    False
                ):
                    logging.info(
                        f"🛡️ [REJECTED] "
                        f"${symbol} - "
                        f"Failed DexScreener check"
                    )
                    tally[
                        "security_rejected"
                    ] += 1
                    with state_lock:
                        security_rejected[
                            mint
                        ] = time.time()
                    continue
                # -----------------------------------------------------
                # Create tracker if necessary
                # -----------------------------------------------------
                with state_lock:
                    tracker = coin_trackers.get(
                        mint
                    )
                    if tracker is None:
                        tracker = CoinMarkovTracker(
                            mint,
                            symbol,
                            graduated_pool_hint,
                        )
                        coin_trackers[
                            mint
                        ] = tracker
                    tracker.last_seen = time.time()
                tracker.refresh()
                # -----------------------------------------------------
                # Markov / momentum signal
                # -----------------------------------------------------
                if tracker.ready:
                    signal = (
                        tracker.signal
                    )
                    signal_src = (
                        f"Markov("
                        f"conf="
                        f"{tracker.confidence:.0%}"
                        f")"
                    )
                    threshold = (
                        MARKOV_THRESHOLD
                    )
                else:
                    signal = (
                        compute_momentum_signal(
                            token
                        )
                    )
                    signal_src = (
                        f"Momentum("
                        f"cold,w="
                        f"{tracker.windows}"
                        f")"
                    )
                    threshold = (
                        MOMENTUM_THRESHOLD
                    )
                if signal is None:
                    tally[
                        "no_signal"
                    ] += 1
                    continue
                logging.info(
                    f"📊 [EVAL] "
                    f"${symbol} | "
                    f"MC=${mc:,.0f} | "
                    f"S={signal:+.3f} "
                    f"[{signal_src}] | "
                    f"Need>{threshold}"
                )
                if signal < threshold:
                    tally[
                        "below_threshold"
                    ] += 1
                    continue
                tally[
                    "qualified"
                ] += 1
                candidates.append({
                    "mint": mint,
                    "symbol": symbol,
                    "price": price,
                    "mc": mc,
                    "signal": signal,
                    "signal_src": signal_src,
                    "ready": tracker.ready,
                    "state": (
                        tracker.current_state
                        if tracker.ready
                        else None
                    ),
                    "stickiness": (
                        tracker.stickiness
                        if tracker.ready
                        else None
                    ),
                })
            # =========================================================
            # FUNNEL REPORT
            # =========================================================
            if mints:
                logging.info(
                    f"🔍 [FUNNEL] "
                    f"{tally['total']} candidates → "
                    f"skip="
                    f"{tally['in_position_or_blacklist']} | "
                    f"security_blacklist="
                    f"{tally['security_blacklist_skip']} | "
                    f"no_token_data="
                    f"{tally['no_token_data']} | "
                    f"failed_filters="
                    f"{tally['failed_filters']} | "
                    f"security_rejected="
                    f"{tally['security_rejected']} | "
                    f"no_signal="
                    f"{tally['no_signal']} | "
                    f"below_threshold="
                    f"{tally['below_threshold']} | "
                    f"qualified="
                    f"{tally['qualified']}"
                )
            # =========================================================
            # RANKING
            # =========================================================
            if candidates:
                candidates.sort(
                    key=lambda c: (
                        c["ready"],
                        c["signal"],
                    ),
                    reverse=True,
                )
                top = " | ".join(
                    (
                        f"${c['symbol']} "
                        f"{c['signal']:+.3f}"
                        f"{'[M]' if c['ready'] else '[mom]'}"
                    )
                    for c in candidates[:5]
                )
                logging.info(
                    f"🏆 [RANKING] "
                    f"{len(candidates)} "
                    f"qualified this cycle | "
                    f"Top: {top}"
                )
            # =========================================================
            # OPEN SLOTS
            # =========================================================
            with state_lock:
                slots_open = (
                    MAX_POSITIONS
                    - len(active_positions)
                )
            # =========================================================
            # ENTRY LOOP
            # =========================================================
            for cand in candidates:
                if slots_open <= 0:
                    break
                mint = cand["mint"]
                with state_lock:
                    if mint in active_positions:
                        continue
                # -----------------------------------------------------
                # Hard risk gate
                # -----------------------------------------------------
                if not risk_entry_ok():
                    with state_lock:
                        daily_loss = (
                            risk_state[
                                "daily_loss_sol"
                            ]
                        )
                    logging.info(
                        f"🚫 [RISK] "
                        f"Entry blocked by "
                        f"kill-switch "
                        f"(daily loss: "
                        f"{daily_loss:.4f} SOL)"
                    )
                    break
                # -----------------------------------------------------
                # Familiars owner limit
                # -----------------------------------------------------
                limits = (
                    familiars_limits()
                )
                max_pos_usd = (
                    limits.get(
                        "maxPositionUsd"
                    )
                )
                size = (
                    effective_trade_size()
                )
                if max_pos_usd:
                    sol_price = (
                        get_price(
                            SOL_MINT
                        )
                        or 150
                    )
                    trade_usd = (
                        size
                        * sol_price
                    )
                    max_pos_usd_float = (
                        safe_float(
                            max_pos_usd
                        )
                    )
                    if (
                        max_pos_usd_float > 0
                        and trade_usd
                        > max_pos_usd_float
                    ):
                        logging.warning(
                            f"⛔ [LIMITS] "
                            f"Trade ~"
                            f"${trade_usd:.0f} "
                            f"exceeds owner cap "
                            f"${max_pos_usd_float:.0f}"
                        )
                        continue
                # -----------------------------------------------------
                # Live price check
                # -----------------------------------------------------
                live_price = get_price(
                    mint
                )
                if (
                    not live_price
                    or live_price <= 0
                ):
                    logging.info(
                        f"⚠️ [ENTRY] "
                        f"${cand['symbol']} — "
                        f"no live price, "
                        f"skipping"
                    )
                    continue
                snapshot_price = safe_float(
                    cand["price"]
                )
                if snapshot_price > 0:
                    drift = (
                        abs(
                            live_price
                            - snapshot_price
                        )
                        / snapshot_price
                    )
                    if (
                        drift
                        > MAX_ENTRY_DRIFT_PCT
                    ):
                        logging.info(
                            f"⚠️ [ENTRY] "
                            f"${cand['symbol']} "
                            f"skipped — "
                            f"price drifted "
                            f"{drift * 100:.1f}% "
                            f"since snapshot "
                            f"("
                            f"${snapshot_price:.8f}"
                            f" → "
                            f"${live_price:.8f}"
                            f")"
                        )
                        continue
                # -----------------------------------------------------
                # Simulated entry price
                # -----------------------------------------------------
                entry_price = (
                    live_price
                    * (
                        1
                        + FEE_SLIPPAGE_PCT
                    )
                )
                # -----------------------------------------------------
                # Entry reason
                # -----------------------------------------------------
                reason_parts = [
                    f"${cand['symbol']} "
                    f"({mint})",
                    f"MC="
                    f"${cand['mc']:,.0f}",
                    f"SOL="
                    f"{sol_regime}",
                    f"Signal="
                    f"{cand['signal']:+.3f} "
                    f"[{cand['signal_src']}]",
                    f"size="
                    f"{size:.3f} SOL",
                ]
                if cand["ready"]:
                    reason_parts += [
                        f"State="
                        f"{STATE_NAME[cand['state']]}",
                        f"Stickiness="
                        f"{cand['stickiness']}",
                    ]
                reason = " | ".join(
                    reason_parts
                )
                logging.info(
                    f"\n🚀 [ENTRY] {reason}"
                )
                familiars_post(
                    "callout",
                    f"Entering {reason}",
                    mint=mint,
                )
                # -----------------------------------------------------
                # PAPER TRADE
                # -----------------------------------------------------
                if PAPER_TRADING:
                    with state_lock:
                        # Re-check position count while holding lock.
                        if (
                            len(active_positions)
                            >= MAX_POSITIONS
                        ):
                            break
                        active_positions[
                            mint
                        ] = {
                            "symbol": (
                                cand["symbol"]
                            ),
                            "entry_price": (
                                entry_price
                            ),
                            "trade_size": size,
                            "signal_src": (
                                cand["signal_src"]
                            ),
                            "signal": (
                                cand["signal"]
                            ),
                        }
                    logging.info(
                        f"💰 [PAPER] BUY "
                        f"{size:.3f} SOL → "
                        f"${cand['symbol']} "
                        f"({mint}) @ "
                        f"${entry_price:.8f} "
                        f"(live "
                        f"${live_price:.8f} + "
                        f"{FEE_SLIPPAGE_PCT * 100:.1f}% "
                        f"cost)"
                    )
                # -----------------------------------------------------
                # Live execution stub
                # -----------------------------------------------------
                # When ready:
                # set PAPER_TRADING = False
                # and add Jupiter swap execution here.
                slots_open -= 1
        except Exception as e:
            logging.error(
                f"❌ [LOOP] "
                f"Main loop iteration failed: {e}"
            )
        # -------------------------------------------------------------
        # Maintain the requested loop interval
        # -------------------------------------------------------------
        elapsed = (
            time.time()
            - cycle_start
        )
        sleep_time = max(
            0.0,
            LOOP_INTERVAL - elapsed,
        )
        time.sleep(
            sleep_time
        )
# =====================================================================
# FLASK HEALTH CHECK & SERVER ENTRY POINT
# =====================================================================
app = Flask(__name__)
@app.route("/")
@app.route("/health")
def health():
    with state_lock:
        positions = len(
            active_positions
        )
        stats = dict(
            trade_stats
        )
    return (
        f"Markov 2.0 Active | "
        f"Positions={positions} | "
        f"Stats={stats}",
        200,
    )
def run_flask():
    import logging as _log
    _log.getLogger(
        "werkzeug"
    ).setLevel(
        _log.ERROR
    )
    # Launch trading bot and monitor threads
    # inside the execution context.
    monitor_thread = threading.Thread(
        target=run_monitor,
        daemon=True,
        name="position-monitor",
    )
    bot_thread = threading.Thread(
        target=run_bot,
        daemon=True,
        name="trading-bot",
    )
    monitor_thread.start()
    bot_thread.start()
    port = int(
        os.environ.get(
            "PORT",
            10000,
        )
    )
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False,
    )
# =====================================================================
# APPLICATION ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    run_flask()