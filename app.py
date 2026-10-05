import os
import csv
import time
import logging
import logging.handlers
import queue
import threading
import requests
from flask import Flask

# ── Non-blocking logging queue ────────────────────────────────────────
# Python's logging module holds one internal lock while writing to stdout.
# If that write ever stalls (pipe back-pressure under load), every thread
# waiting to log blocks too — silently freezing the bot while gunicorn's
# HTTP handling keeps returning 200. Routing through a queue makes every
# logging call a fast non-blocking queue.put(); a single dedicated thread
# does the actual write so only that thread can stall.
_log_queue = queue.Queue(-1)
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
_queue_listener = logging.handlers.QueueListener(_log_queue, _stream_handler)
_queue_listener.start()
_root_logger = logging.getLogger()
_root_logger.setLevel(logging.INFO)
_root_logger.addHandler(logging.handlers.QueueHandler(_log_queue))

# =====================================================================
# CONFIGURATION
# =====================================================================
PAPER_TRADING = True
SOL_TRADE_SIZE = 0.1

MIN_LIQUIDITY_USD        = 3_000.0
MIN_MARKET_CAP           = 10_000.0
MAX_MARKET_CAP           = 250_000.0
MIN_5M_VOLUME            = 500.0
MAX_MACRO_DRAWDOWN       = -25.0

STRIDE_BARS              = 4
MIN_WINDOWS              = 12
ATR_BULL_MULT            = 0.5
ATR_BEAR_MULT            = 0.5
MARKOV_THRESHOLD         = 0.20
MOMENTUM_THRESHOLD       = 0.25

SOL_BEAR_H24             = -15.0
SOL_BULL_H24             =  10.0
SOL_MINT                 = "So11111111111111111111111111111111111111112"

TAKE_PROFIT_PCT          = 0.35
STOP_LOSS_PCT            = 0.12
BLACKLIST_COOLDOWN       = 7200
SECURITY_REJECT_COOLDOWN = 14400
MAX_POSITIONS            = 1
LOOP_INTERVAL            = 30
OHLCV_TTL                = 300
OHLCV_RETRY_COOLDOWN     = 45
TRACKER_STALE_SECONDS    = 3600
POOL_RESOLVE_MAX_ATTEMPTS= 5

MAX_ENTRY_DRIFT_PCT      = 0.03
FEE_SLIPPAGE_PCT         = 0.01

MAX_DAILY_LOSS_SOL       = 0.50
MAX_CONSECUTIVE_LOSSES   = 3

TRADE_LOG_PATH           = "/tmp/trades.csv"

FAMILIARS_KEY   = os.environ.get("FAMILIARS_API_KEY", "")
FAMILIARS_URL   = "https://familiars.family"
JUPITER_API_KEY = os.environ.get("JUPITER_API_KEY", "")
JUP_BASE_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def jup_headers():
    h = dict(JUP_BASE_HEADERS)
    if JUPITER_API_KEY:
        h["x-api-key"] = JUPITER_API_KEY
    return h

_jup_backoff    = {"until": 0}
JUP_BACKOFF_SECONDS = 30

def jup_available():
    return time.time() >= _jup_backoff["until"]

def jup_mark_limited():
    _jup_backoff["until"] = time.time() + JUP_BACKOFF_SECONDS

_gecko_backoff  = {"until": 0}
GECKO_BACKOFF_SECONDS = 60

def gecko_available():
    return time.time() >= _gecko_backoff["until"]

def gecko_mark_limited():
    _gecko_backoff["until"] = time.time() + GECKO_BACKOFF_SECONDS

# =====================================================================
# STATE
# =====================================================================
active_positions  = {}
stopped_out_tokens= {}
security_rejected = {}
coin_trackers     = {}
_pool_cache       = {}
trade_stats = {"total_closed": 0, "wins": 0, "losses": 0, "net_sol_pnl": 0.0}
_sol_cache  = {"state": "SIDEWAYS", "last_check": 0}

risk_state = {
    "consecutive_losses": 0,
    "daily_loss_sol":     0.0,
    "daily_loss_date":    "",
    "kill_switch":        False,
}

BULL, BEAR, SIDEWAYS = 0, 1, 2
STATE_NAME = {BULL: "BULL", BEAR: "BEAR", SIDEWAYS: "SIDEWAYS"}

# =====================================================================
# TRADE LOG
# =====================================================================
def _ensure_trade_log():
    try:
        if not os.path.exists(TRADE_LOG_PATH) or os.path.getsize(TRADE_LOG_PATH) == 0:
            with open(TRADE_LOG_PATH, "w", newline="") as f:
                csv.writer(f).writerow([
                    "utc", "symbol", "mint", "outcome",
                    "entry_price", "exit_price",
                    "gross_pnl_pct", "net_pnl_pct", "net_pnl_sol",
                    "signal_src", "signal",
                ])
    except Exception as e:
        logging.warning(f"⚠️ [LOG] Could not create trade log: {e}")

def log_trade(symbol, mint, outcome, entry_price, exit_price, signal_src, signal):
    try:
        gross   = (exit_price - entry_price) / entry_price
        net     = gross - 2 * FEE_SLIPPAGE_PCT
        net_sol = SOL_TRADE_SIZE * net
        with open(TRADE_LOG_PATH, "a", newline="") as f:
            csv.writer(f).writerow([
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                symbol, mint, outcome,
                f"{entry_price:.10f}", f"{exit_price:.10f}",
                f"{gross*100:.2f}", f"{net*100:.2f}", f"{net_sol:.5f}",
                signal_src, f"{signal:.4f}",
            ])
    except Exception as e:
        logging.warning(f"⚠️ [LOG] Trade write failed: {e}")

_ensure_trade_log()

# =====================================================================
# RISK ENGINE
# =====================================================================
def risk_check_daily_reset():
    today = time.strftime("%Y-%m-%d", time.gmtime())
    if risk_state["daily_loss_date"] != today:
        risk_state["daily_loss_date"] = today
        risk_state["daily_loss_sol"]  = 0.0
        if risk_state["kill_switch"]:
            logging.info("🔓 [RISK] New UTC day — daily kill-switch cleared")
            risk_state["kill_switch"] = False

def risk_on_loss(sol_lost):
    risk_state["consecutive_losses"] += 1
    risk_state["daily_loss_sol"]     += abs(sol_lost)
    if risk_state["consecutive_losses"] >= MAX_CONSECUTIVE_LOSSES:
        logging.warning(
            f"⚠️ [RISK] {risk_state['consecutive_losses']} consecutive losses — "
            f"trade size halved until next win"
        )
    if risk_state["daily_loss_sol"] >= MAX_DAILY_LOSS_SOL:
        risk_state["kill_switch"] = True
        logging.error(
            f"🛑 [RISK] Daily loss limit hit "
            f"({risk_state['daily_loss_sol']:.4f} SOL ≥ {MAX_DAILY_LOSS_SOL} SOL) — "
            f"kill-switch latched, no new entries until UTC midnight"
        )

def risk_on_win():
    risk_state["consecutive_losses"] = 0

def effective_trade_size():
    if risk_state["consecutive_losses"] >= MAX_CONSECUTIVE_LOSSES:
        return SOL_TRADE_SIZE * 0.5
    return SOL_TRADE_SIZE

def risk_entry_ok():
    risk_check_daily_reset()
    return not risk_state["kill_switch"]

# =====================================================================
# FAMILIARS INTEGRATION
# =====================================================================
def familiars_post(kind, text, mint=None, signature=None):
    if not FAMILIARS_KEY:
        return
    payload = {"kind": kind, "text": text[:500]}
    if mint:
        payload["mint"] = mint
    if signature:
        payload["signature"] = signature
    try:
        requests.post(
            f"{FAMILIARS_URL}/api/posts",
            headers={"Authorization": f"Bearer {FAMILIARS_KEY}",
                     "Content-Type": "application/json"},
            json=payload, timeout=5
        )
    except Exception:
        pass

def familiars_limits():
    if not FAMILIARS_KEY:
        return {}
    try:
        r = requests.get(
            f"{FAMILIARS_URL}/api/agent/me",
            headers={"Authorization": f"Bearer {FAMILIARS_KEY}"},
            timeout=5
        )
        if r.status_code == 200:
            return r.json().get("settings", {})
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
            headers=jup_headers(), timeout=5
        )
        if r.status_code == 429:
            jup_mark_limited()
        if r.status_code == 200:
            data  = r.json()
            items = data if isinstance(data, list) else (data.get("data") or data.get("tokens") or [])
            token = next(
                (t for t in items
                 if (t.get("id") or t.get("address") or t.get("mint")) == SOL_MINT),
                None
            )
            if token:
                h24 = float((token.get("stats24h") or {}).get("priceChange") or 0)
                h6  = float((token.get("stats6h")  or {}).get("priceChange") or 0)
                h1  = float((token.get("stats1h")  or {}).get("priceChange") or 0)
                if h24 <= SOL_BEAR_H24 or (h6 <= -10 and h1 <= -5):
                    state = "BEAR"
                elif h24 >= SOL_BULL_H24 and h6 >= 5:
                    state = "BULL"
                else:
                    state = "SIDEWAYS"
                _sol_cache.update({"state": state, "last_check": now})
                logging.info(
                    f"🌐 [SOL MACRO] {state} | "
                    f"24h={h24:+.1f}%  6h={h6:+.1f}%  1h={h1:+.1f}%"
                )
    except Exception as e:
        logging.error(f"❌ [SOL REGIME] {e}")
    return _sol_cache["state"]

# =====================================================================
# OHLCV — GECKOTERMINAL
# =====================================================================
def fetch_ohlcv(pool_address, agg_min=5, limit=200):
    if not gecko_available():
        return []
    url = (
        f"https://api.geckoterminal.com/api/v2/networks/solana"
        f"/pools/{pool_address}/ohlcv/minute"
        f"?aggregate={agg_min}&limit={limit}&currency=usd"
    )
    try:
        r = requests.get(url, headers={"Accept": "application/json"}, timeout=8)
        if r.status_code == 404:
            return []
        if r.status_code == 429:
            gecko_mark_limited()
            logging.warning(f"⚠️ [GECKO] Rate limited on {pool_address[:8]}… backing off 60s")
            return []
        if r.status_code != 200:
            return []
        raw = r.json().get("data", {}).get("attributes", {}).get("ohlcv_list", [])
        if not raw:
            return []
        return [
            {"ts": c[0], "o": float(c[1]), "h": float(c[2]),
             "l": float(c[3]), "c": float(c[4]), "v": float(c[5])}
            for c in reversed(raw)
            if c[1] and c[4]
        ]
    except Exception as e:
        logging.error(f"❌ [GECKO OHLCV] {pool_address[:8]}…: {e}")
        return []

def resolve_pool_address(mint, graduated_pool_hint=None):
    """
    Returns (pool_address_or_None, was_rate_limited).
    was_rate_limited=True means the failure was transient — don't count
    it toward the give-up counter on the tracker.
    """
    if graduated_pool_hint:
        return graduated_pool_hint, False

    if mint in _pool_cache:
        return _pool_cache[mint], False

    if not gecko_available():
        return None, True

    url = f"https://api.geckoterminal.com/api/v2/networks/solana/tokens/{mint}/pools"
    try:
        r = requests.get(url, headers={"Accept": "application/json"}, timeout=8)
        if r.status_code == 200:
            data = r.json().get("data", [])
            if data:
                pool_addr = data[0].get("attributes", {}).get("address")
                if pool_addr:
                    _pool_cache[mint] = pool_addr
                    return pool_addr, False
            return None, False
        if r.status_code == 429:
            gecko_mark_limited()
            logging.warning(
                f"⚠️ [GECKO POOL] Rate limited resolving pool for {mint[:8]}… backing off 60s"
            )
            return None, True
        return None, False
    except Exception as e:
        logging.error(f"❌ [GECKO POOL] {mint[:8]}…: {e}")
        return None, True

# =====================================================================
# MARKOV 2.0 ENGINE
# =====================================================================
def _atr_pct(candles, period=14):
    if len(candles) < period + 1:
        return 3.0
    trs = []
    for i in range(1, len(candles)):
        h, l, pc = candles[i]["h"], candles[i]["l"], candles[i-1]["c"]
        if pc == 0:
            continue
        trs.append(max(h - l, abs(h - pc), abs(l - pc)) / pc * 100)
    tail = trs[-period:]
    return sum(tail) / len(tail) if tail else 3.0

def _label_states(candles, stride, bull_mult, bear_mult):
    """FIX 1 — Stride sampling: non-overlapping windows, ATR-adaptive thresholds."""
    atr         = _atr_pct(candles)
    bull_thresh =  atr * bull_mult
    bear_thresh = -atr * bear_mult
    states = []
    for i in range(0, len(candles) - stride + 1, stride):
        w = candles[i:i + stride]
        o, c = w[0]["o"], w[-1]["c"]
        if o == 0:
            continue
        ret = (c - o) / o * 100
        if   ret >= bull_thresh: states.append(BULL)
        elif ret <= bear_thresh: states.append(BEAR)
        else:                    states.append(SIDEWAYS)
    return states, bull_thresh, bear_thresh

def _build_matrix(states):
    counts = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(len(states) - 1):
        counts[states[i]][states[i+1]] += 1
    matrix = []
    for row in counts:
        total = sum(row)
        matrix.append([v / total if total else 1/3 for v in row])
    stickiness = {STATE_NAME[s]: round(matrix[s][s], 3) for s in (BULL, BEAR, SIDEWAYS)}
    return matrix, stickiness

def _verify_labels(states, candles, stride):
    """FIX 2 — Label verification: spot-check first, middle, last windows."""
    errors = 0
    for idx in [0, len(states) // 2, len(states) - 1]:
        start = idx * stride
        w = candles[start:start + stride]
        if not w or w[0]["o"] == 0:
            continue
        ret = (w[-1]["c"] - w[0]["o"]) / w[0]["o"] * 100
        if ret >  5 and states[idx] == BEAR: errors += 1
        if ret < -5 and states[idx] == BULL: errors += 1
    if errors:
        logging.warning(
            f"⚠️ [MARKOV FIX2] {errors} label anomaly(s) — "
            f"matrix built but confidence is lower"
        )
    return errors == 0

def _markov_signal(matrix, current_state):
    """P(BULL|current) − P(BEAR|current). Range −1 to +1."""
    return round(matrix[current_state][BULL] - matrix[current_state][BEAR], 4)

# =====================================================================
# PER-COIN MARKOV TRACKER
# =====================================================================
class CoinMarkovTracker:
    def __init__(self, mint, symbol, graduated_pool_hint=None):
        self.mint                  = mint
        self.symbol                = symbol
        self.graduated_pool_hint   = graduated_pool_hint
        self.pool_address          = None
        self.pool_last_try         = 0
        self.pool_resolve_attempts = 0
        self.pool_abandoned        = False
        self.candles               = []
        self.states                = []
        self.matrix                = None
        self.stickiness            = None
        self.signal                = None
        self.current_state         = SIDEWAYS
        self.windows               = 0
        self.last_fetch            = 0
        self.verified              = False
        self.last_seen             = time.time()

    @property
    def ready(self):
        return self.windows >= MIN_WINDOWS and self.signal is not None

    @property
    def confidence(self):
        if self.windows < MIN_WINDOWS:
            return 0.0
        return min((self.windows - MIN_WINDOWS) / (MIN_WINDOWS * 3), 1.0)

    def refresh(self):
        if self.pool_abandoned:
            return

        if not self.pool_address:
            if time.time() - self.pool_last_try < OHLCV_RETRY_COOLDOWN:
                return
            self.pool_last_try = time.time()
            addr, was_rate_limited = resolve_pool_address(
                self.mint, self.graduated_pool_hint
            )
            if addr:
                self.pool_address = addr
            else:
                if not was_rate_limited:
                    self.pool_resolve_attempts += 1
                    if self.pool_resolve_attempts >= POOL_RESOLVE_MAX_ATTEMPTS:
                        self.pool_abandoned = True
                        logging.info(
                            f"📉 [POOL] Giving up on ${self.symbol} — "
                            f"no pool after {self.pool_resolve_attempts} checks, "
                            f"momentum-only from here"
                        )
                return

        ttl = OHLCV_TTL if self.candles else OHLCV_RETRY_COOLDOWN
        if time.time() - self.last_fetch < ttl:
            return

        candles = fetch_ohlcv(self.pool_address)
        self.last_fetch = time.time()

        if not candles:
            return

        self.candles = candles
        states, bull_t, bear_t = _label_states(
            candles, STRIDE_BARS, ATR_BULL_MULT, ATR_BEAR_MULT
        )
        if len(states) < 3:
            return

        self.verified      = _verify_labels(states, candles, STRIDE_BARS)
        self.states        = states
        self.windows       = len(states)
        self.matrix, self.stickiness = _build_matrix(states)
        self.current_state = states[-1]
        self.signal        = _markov_signal(self.matrix, self.current_state)

        logging.info(
            f"📈 [MARKOV] ${self.symbol} | "
            f"Windows={self.windows} | State={STATE_NAME[self.current_state]} | "
            f"S={self.signal:+.4f} | Conf={self.confidence:.0%} | "
            f"Stick={self.stickiness} | "
            f"Thresh=[+{bull_t:.2f}% / {bear_t:.2f}%] | "
            f"Labels={'✅' if self.verified else '⚠️'}"
        )

# =====================================================================
# CANDIDATE DISCOVERY — JUPITER TOKEN API V2 ONLY
# =====================================================================
def fetch_jupiter_candidates():
    """Returns {mint: token_object} for up to 40 candidates (20 per source)."""
    token_map     = {}
    source_report = []
    sources = [
        ("recent",     "https://api.jup.ag/tokens/v2/recent"),
        ("trending5m", "https://api.jup.ag/tokens/v2/toptrending/5m"),
    ]
    for label, url in sources:
        if not jup_available():
            source_report.append((label, 0, "skipped(backoff)"))
            continue
        try:
            r = requests.get(url, headers=jup_headers(), timeout=5)
            if r.status_code == 200:
                before = len(token_map)
                data   = r.json()
                if isinstance(data, list):
                    items = data
                elif isinstance(data, dict):
                    items = data.get("data") or data.get("tokens") or []
                else:
                    items = []
                for item in items[:20]:
                    addr = item.get("address") or item.get("mint") or item.get("id")
                    if addr and addr not in token_map:
                        token_map[addr] = item
                added = len(token_map) - before
                note  = "ok" if added else (
                    f"0 items, keys={list(data.keys()) if isinstance(data, dict) else 'list'}"
                )
                source_report.append((label, added, note))
            elif r.status_code == 429:
                jup_mark_limited()
                source_report.append((label, 0, "HTTP 429→backoff"))
            else:
                source_report.append((label, 0, f"HTTP {r.status_code}"))
        except Exception as e:
            source_report.append((label, 0, f"exc: {e}"))
    breakdown = " | ".join(f"{label}:{count}({note})" for label, count, note in source_report)
    logging.info(f"📡 [SCRAPER] {len(token_map)} candidate tokens | {breakdown}")
    return token_map

# =====================================================================
# TOKEN FILTERS
# =====================================================================
def validate_token(token):
    if not token:
        return False
    liq = float(token.get("liquidity") or 0)
    if liq < MIN_LIQUIDITY_USD:
        return False
    mc = float(token.get("mcap") or token.get("fdv") or 0)
    if mc < MIN_MARKET_CAP or mc > MAX_MARKET_CAP:
        return False
    stats5m  = token.get("stats5m")  or {}
    stats6h  = token.get("stats6h")  or {}
    stats24h = token.get("stats24h") or {}
    vol5m = (float(stats5m.get("buyVolume")  or 0)
           + float(stats5m.get("sellVolume") or 0))
    if vol5m < MIN_5M_VOLUME:
        return False
    if float(stats6h.get("priceChange")  or 0) < MAX_MACRO_DRAWDOWN:
        return False
    if float(stats24h.get("priceChange") or 0) < MAX_MACRO_DRAWDOWN:
        return False
    buys  = int(stats5m.get("numBuys")  or 0)
    sells = int(stats5m.get("numSells") or 0)
    if (buys + sells) < 12 or buys < (sells * 1.2):
        return False
    return True

# =====================================================================
# LAYER 2 — MOMENTUM SIGNAL (cold-start fallback)
# =====================================================================
def compute_momentum_signal(token):
    stats5m = token.get("stats5m") or {}
    stats1h = token.get("stats1h") or {}
    stats6h = token.get("stats6h") or {}
    m5 = float(stats5m.get("priceChange") or 0)
    h1 = float(stats1h.get("priceChange") or 0)
    h6 = float(stats6h.get("priceChange") or 0)
    if h1 > 50 or m5 > 30:
        return None
    v_m5      = m5  / 5.0
    v_h1      = h1  / 60.0
    v_h6      = h6  / 360.0
    delta_v   = v_m5 - v_h1
    stability = 1.0 if abs(v_h1 - v_h6) < 0.5 else 0.5
    S         = (delta_v * 0.6 + v_m5 * 0.4) * stability
    vol5m     = (float(stats5m.get("buyVolume")  or 0)
               + float(stats5m.get("sellVolume") or 0))
    vol_wt    = min(max(vol5m / 1000.0, 0.5), 1.5)
    return round(S * vol_wt, 2)

# =====================================================================
# SECURITY CHECKS
# =====================================================================
def check_gmgn_batch(mints):
    """
    DexScreener batch liquidity/safety check.
    Called AFTER validate_token() so we only query tokens already past
    the basic filters — no wasted API calls or blocking sleeps.
    Single attempt per chunk, no retry sleeps. On API error tokens
    default to True (pass) so an outage doesn't stop all trading.
    """
    if not mints:
        return {}

    results = {mint: True for mint in mints}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    for i in range(0, len(mints), 10):
        chunk = mints[i:i + 10]
        url   = f"https://api.dexscreener.com/latest/dex/tokens/{','.join(chunk)}"
        try:
            res = requests.get(url, headers=headers, timeout=5)
            if res.status_code != 200:
                logging.warning(
                    f"⚠️ [SECURITY] DexScreener batch returned {res.status_code} — "
                    f"skipping chunk, tokens default to pass"
                )
                continue

            pairs_by_mint = {}
            for pair in (res.json().get("pairs") or []):
                base_addr = pair.get("baseToken", {}).get("address")
                if base_addr and base_addr in chunk:
                    pairs_by_mint.setdefault(base_addr, []).append(pair)

            for mint in chunk:
                token_pairs = pairs_by_mint.get(mint, [])
                if not token_pairs:
                    results[mint] = False
                    logging.info(f"🛡️ [SECURITY] {mint[:8]}… — no active pair on DexScreener")
                    continue

                main_pair = token_pairs[0]
                liquidity = float(main_pair.get("liquidity", {}).get("usd", 0) or 0)
                fdv       = float(main_pair.get("fdv", 0) or 0)

                if liquidity < 3000:
                    results[mint] = False
                    logging.info(f"🛡️ [SECURITY] {mint[:8]}… — low liquidity ${liquidity:,.0f}")
                    continue

                if fdv > 0 and (liquidity / fdv) < 0.01:
                    results[mint] = False
                    logging.info(
                        f"🛡️ [SECURITY] {mint[:8]}… — "
                        f"thin liquidity ratio {liquidity/fdv:.2%}"
                    )
                    continue

        except Exception as e:
            logging.warning(f"⚠️ [SECURITY] DexScreener batch error: {e}")

    return results


def check_rugcheck(mint):
    """
    Returns True on any exception or non-200 response so a RugCheck
    outage doesn't halt all trading. The DexScreener batch check and
    validate_token() filters still provide a safety net.
    """
    try:
        r = requests.get(
            f"https://api.rugcheck.xyz/v1/tokens/{mint}/report/summary",
            timeout=5
        )
        if r.status_code == 200:
            data = r.json()
            if data.get("riskLevel") in ("Danger", "High"):
                return False
            bad = {
                "Single holder ownership",
                "High holder concentration",
                "Mint Authority Enabled",
                "Freeze Authority Enabled",
            }
            for risk in data.get("risks", []):
                if risk.get("name") in bad:
                    return False
    except Exception:
        pass
    return True

# =====================================================================
# REAL-TIME PRICE
# =====================================================================
def get_price(mint):
    # 1. Jupiter Price API v2 (fastest)
    if jup_available():
        try:
            r = requests.get(
                f"https://api.jup.ag/price/v2?ids={mint}",
                headers=jup_headers(), timeout=2
            )
            if r.status_code == 200:
                p = r.json().get("data", {}).get(mint, {}).get("price")
                if p:
                    return float(p)
            elif r.status_code == 429:
                jup_mark_limited()
        except Exception:
            pass

    # 2. Jupiter Token Search API (fallback)
    if jup_available():
        try:
            r = requests.get(
                f"https://api.jup.ag/tokens/v2/search?query={mint}",
                headers=jup_headers(), timeout=3
            )
            if r.status_code == 200:
                data  = r.json()
                items = data if isinstance(data, list) else (
                    data.get("data") or data.get("tokens") or []
                )
                for item in items:
                    if (item.get("address") or item.get("mint") or item.get("id")) == mint:
                        p = (item.get("price")
                             or item.get("usdPrice")
                             or (item.get("stats24h") or {}).get("price"))
                        if p:
                            return float(p)
            elif r.status_code == 429:
                jup_mark_limited()
        except Exception:
            pass

    # 3. DexScreener emergency fallback
    try:
        r = requests.get(
            f"https://api.dexscreener.com/latest/dex/tokens/{mint}",
            timeout=3
        )
        if r.status_code == 200:
            pairs = r.json().get("pairs") or []
            if pairs:
                p = pairs[0].get("priceUsd")
                if p:
                    return float(p)
    except Exception:
        pass

    return None

# =====================================================================
# POSITION MONITOR — 1-second background thread
# =====================================================================
def run_monitor():
    logging.info("⚡ [MONITOR] Position monitor started (1s loop)")
    heartbeat = 0
    while True:
        heartbeat += 1
        try:
            if active_positions:
                to_close  = []
                pnl_lines = []

                for mint in list(active_positions.keys()):
                    info = active_positions.get(mint)
                    if not info:
                        continue
                    price = get_price(mint)
                    entry = info.get("entry_price", 0)
                    if not price or entry == 0:
                        continue

                    symbol = info.get("symbol", "UNKNOWN")
                    gross  = (price - entry) / entry
                    net    = gross - 2 * FEE_SLIPPAGE_PCT
                    size   = info.get("trade_size", SOL_TRADE_SIZE)

                    pnl_lines.append(
                        f"${symbol} {gross*100:+.1f}% gross / {net*100:+.1f}% net"
                    )

                    if gross >= TAKE_PROFIT_PCT:
                        net_sol = size * net
                        trade_stats["total_closed"] += 1
                        trade_stats["wins"]         += 1
                        trade_stats["net_sol_pnl"]  += net_sol
                        wr = trade_stats["wins"] / trade_stats["total_closed"] * 100
                        logging.info(
                            f"🎯 [TP] ${symbol} gross +{gross*100:.1f}% | "
                            f"net {net*100:.1f}% | +{net_sol:.4f} SOL | "
                            f"WR={wr:.1f}% | Net={trade_stats['net_sol_pnl']:+.4f} SOL"
                        )
                        risk_on_win()
                        log_trade(symbol, mint, "TP", entry, price,
                                  info.get("signal_src", "?"), info.get("signal", 0))
                        try:
                            familiars_post(
                                "trade",
                                f"TP +{gross*100:.1f}% (net {net*100:.1f}%) on ${symbol}. "
                                f"Net={trade_stats['net_sol_pnl']:+.4f} SOL",
                                mint=mint
                            )
                        except Exception as post_err:
                            logging.warning(f"⚠️ [MONITOR] TP post error: {post_err}")
                        to_close.append(mint)

                    elif gross <= -STOP_LOSS_PCT:
                        net_sol = size * net
                        trade_stats["total_closed"] += 1
                        trade_stats["losses"]        += 1
                        trade_stats["net_sol_pnl"]   += net_sol
                        wr = trade_stats["wins"] / trade_stats["total_closed"] * 100
                        logging.info(
                            f"🛑 [SL] ${symbol} gross {gross*100:.1f}% | "
                            f"net {net*100:.1f}% | {net_sol:.4f} SOL | "
                            f"WR={wr:.1f}% | Net={trade_stats['net_sol_pnl']:+.4f} SOL"
                        )
                        risk_on_loss(abs(net_sol))
                        log_trade(symbol, mint, "SL", entry, price,
                                  info.get("signal_src", "?"), info.get("signal", 0))
                        try:
                            familiars_post(
                                "trade",
                                f"SL {gross*100:.1f}% (net {net*100:.1f}%) on ${symbol}. "
                                f"Blacklisting 2h.",
                                mint=mint
                            )
                        except Exception as post_err:
                            logging.warning(f"⚠️ [MONITOR] SL post error: {post_err}")
                        stopped_out_tokens[mint] = time.time()
                        to_close.append(mint)

                for mint in to_close:
                    active_positions.pop(mint, None)
                    coin_trackers.pop(mint, None)

                if pnl_lines and heartbeat % 10 == 0:
                    logging.info("📟 [PNL] " + " | ".join(pnl_lines))

        except Exception as e:
            logging.error(f"❌ [MONITOR] {e}")
        time.sleep(1)

# =====================================================================
# MAIN TRADING LOOP
# =====================================================================
def run_bot():
    logging.info(
        f"🚀 [BOT] Markov 2.0 Engine active | "
        f"Paper={PAPER_TRADING} | Stride={STRIDE_BARS}×5m | "
        f"MinWindows={MIN_WINDOWS}"
    )
    while True:
        cycle_start = time.time()
        try:
            if len(active_positions) >= MAX_POSITIONS:
                time.sleep(LOOP_INTERVAL)
                continue

            sol_regime = get_sol_regime()
            if sol_regime == "BEAR":
                logging.info("🚫 [MACRO] SOL=BEAR — sitting out this cycle")
                time.sleep(LOOP_INTERVAL)
                continue

            stale = [
                mint for mint, tracker in coin_trackers.items()
                if mint not in active_positions
                and (tracker.pool_abandoned
                     or time.time() - tracker.last_seen > TRACKER_STALE_SECONDS)
            ]
            for mint in stale:
                del coin_trackers[mint]
            if stale:
                logging.info(f"🧹 [PRUNE] Dropped {len(stale)} stale tracker(s)")

            for tracker in list(coin_trackers.values()):
                tracker.refresh()

            logging.info(
                f"🔎 [SCAN] SOL={sol_regime} | "
                f"Tracking={len(coin_trackers)} coins | "
                f"Active={len(active_positions)}"
            )

            token_map = fetch_jupiter_candidates()
            mints     = list(token_map.keys())

            # Pass 1: pre-filter without API calls
            pre_filtered = []
            tally = {
                "total":                    len(mints),
                "in_position_or_blacklist": 0,
                "no_token_data":            0,
                "failed_filters":           0,
                "security_blacklist_skip":  0,
                "security_rejected":        0,
                "no_signal":                0,
                "below_threshold":          0,
                "qualified":                0,
            }

            for mint in mints:
                if mint in active_positions:
                    tally["in_position_or_blacklist"] += 1
                    continue
                if mint in stopped_out_tokens:
                    if time.time() - stopped_out_tokens[mint] < BLACKLIST_COOLDOWN:
                        tally["in_position_or_blacklist"] += 1
                        continue
                    del stopped_out_tokens[mint]
                token = token_map.get(mint)
                if not token:
                    tally["no_token_data"] += 1
                    continue
                if not validate_token(token):
                    tally["failed_filters"] += 1
                    continue
                pre_filtered.append(mint)

            # Batch security check on pre-filtered tokens only
            security_results = check_gmgn_batch(pre_filtered) if pre_filtered else {}

            # Pass 2: security + signal evaluation
            candidates = []
            for mint in pre_filtered:
                token               = token_map[mint]
                symbol              = token.get("symbol", "?")
                graduated_pool_hint = token.get("graduatedPool")
                price               = float(token.get("usdPrice") or 0)
                mc                  = float(token.get("mcap") or token.get("fdv") or 0)

                if mint in security_rejected:
                    if time.time() - security_rejected[mint] < SECURITY_REJECT_COOLDOWN:
                        tally["security_blacklist_skip"] += 1
                        continue
                    del security_rejected[mint]

                if not check_rugcheck(mint):
                    logging.info(f"🛡️ [REJECTED] ${symbol} — RugCheck fail")
                    tally["security_rejected"] += 1
                    security_rejected[mint] = time.time()
                    continue

                if not security_results.get(mint, True):
                    logging.info(f"🛡️ [REJECTED] ${symbol} — DexScreener check failed")
                    tally["security_rejected"] += 1
                    security_rejected[mint] = time.time()
                    continue

                if mint not in coin_trackers:
                    coin_trackers[mint] = CoinMarkovTracker(
                        mint, symbol, graduated_pool_hint
                    )
                tracker = coin_trackers[mint]
                tracker.last_seen = time.time()
                tracker.refresh()

                if tracker.ready:
                    signal     = tracker.signal
                    signal_src = f"Markov(conf={tracker.confidence:.0%})"
                    threshold  = MARKOV_THRESHOLD
                else:
                    signal     = compute_momentum_signal(token)
                    signal_src = f"Momentum(cold,w={tracker.windows})"
                    threshold  = MOMENTUM_THRESHOLD

                if signal is None:
                    tally["no_signal"] += 1
                    continue

                logging.info(
                    f"📊 [EVAL] ${symbol} | MC=${mc:,.0f} | "
                    f"S={signal:+.3f} [{signal_src}] | Need>{threshold}"
                )

                if signal < threshold:
                    tally["below_threshold"] += 1
                    continue

                tally["qualified"] += 1
                candidates.append({
                    "mint":       mint,
                    "symbol":     symbol,
                    "price":      price,
                    "mc":         mc,
                    "signal":     signal,
                    "signal_src": signal_src,
                    "ready":      tracker.ready,
                    "state":      tracker.current_state if tracker.ready else None,
                    "stickiness": tracker.stickiness   if tracker.ready else None,
                })

            if mints:
                logging.info(
                    f"🔍 [FUNNEL] {tally['total']} candidates → "
                    f"skip={tally['in_position_or_blacklist']} | "
                    f"no_token_data={tally['no_token_data']} | "
                    f"failed_filters={tally['failed_filters']} | "
                    f"security_blacklist={tally['security_blacklist_skip']} | "
                    f"security_rejected={tally['security_rejected']} | "
                    f"no_signal={tally['no_signal']} | "
                    f"below_threshold={tally['below_threshold']} | "
                    f"qualified={tally['qualified']}"
                )

            if candidates:
                candidates.sort(key=lambda c: (c["ready"], c["signal"]), reverse=True)
                top = " | ".join(
                    f"${c['symbol']} {c['signal']:+.3f}"
                    f"{'[M]' if c['ready'] else '[mom]'}"
                    for c in candidates[:5]
                )
                logging.info(f"🏆 [RANKING] {len(candidates)} qualified | Top: {top}")

            slots_open = MAX_POSITIONS - len(active_positions)
            for cand in candidates:
                if slots_open <= 0:
                    break
                mint = cand["mint"]
                if mint in active_positions:
                    continue

                if not risk_entry_ok():
                    logging.info(
                        f"🚫 [RISK] Kill-switch active — "
                        f"daily loss {risk_state['daily_loss_sol']:.4f} SOL"
                    )
                    break

                limits      = familiars_limits()
                max_pos_usd = limits.get("maxPositionUsd")
                size        = effective_trade_size()
                if max_pos_usd:
                    sol_price = get_price(SOL_MINT) or 150
                    trade_usd = size * sol_price
                    if trade_usd > float(max_pos_usd):
                        logging.warning(
                            f"⛔ [LIMITS] Trade ~${trade_usd:.0f} "
                            f"exceeds owner cap ${max_pos_usd}"
                        )
                        continue

                live_price = get_price(mint)
                if not live_price or live_price == 0:
                    logging.info(f"⚠️ [ENTRY] ${cand['symbol']} — no live price, skipping")
                    continue
                snapshot_price = cand["price"]
                if snapshot_price and snapshot_price > 0:
                    drift = abs(live_price - snapshot_price) / snapshot_price
                    if drift > MAX_ENTRY_DRIFT_PCT:
                        logging.info(
                            f"⚠️ [ENTRY] ${cand['symbol']} skipped — "
                            f"price drifted {drift*100:.1f}% since snapshot "
                            f"(${snapshot_price:.8f} → ${live_price:.8f})"
                        )
                        continue

                entry_price = live_price * (1 + FEE_SLIPPAGE_PCT)

                reason_parts = [
                    f"${cand['symbol']} ({mint})",
                    f"MC=${cand['mc']:,.0f}",
                    f"SOL={sol_regime}",
                    f"Signal={cand['signal']:+.3f} [{cand['signal_src']}]",
                    f"size={size:.3f} SOL",
                ]
                if cand["ready"]:
                    reason_parts += [
                        f"State={STATE_NAME[cand['state']]}",
                        f"Stickiness={cand['stickiness']}",
                    ]
                reason = " | ".join(reason_parts)
                logging.info(f"\n🚀 [ENTRY] {reason}")
                familiars_post("callout", f"Entering {reason}", mint=mint)

                if PAPER_TRADING:
                    active_positions[mint] = {
                        "symbol":      cand["symbol"],
                        "entry_price": entry_price,
                        "trade_size":  size,
                        "signal_src":  cand["signal_src"],
                        "signal":      cand["signal"],
                    }
                    logging.info(
                        f"💰 [PAPER] BUY {size:.3f} SOL → "
                        f"${cand['symbol']} ({mint}) @ ${entry_price:.8f} "
                        f"(live ${live_price:.8f} + {FEE_SLIPPAGE_PCT*100:.1f}% cost)"
                    )
                # When ready: set PAPER_TRADING = False and wire Jupiter swap here

                slots_open -= 1

        except Exception as e:
            logging.error(f"❌ [LOOP] {e}")

        elapsed    = time.time() - cycle_start
        sleep_time = max(0.0, LOOP_INTERVAL - elapsed)
        time.sleep(sleep_time)

# =====================================================================
# FLASK HEALTH CHECK
# =====================================================================
app = Flask(__name__)

@app.route("/")
@app.route("/health")
def health():
    return (
        f"Markov 2.0 Active | "
        f"Positions={len(active_positions)} | "
        f"Stats={trade_stats}",
        200
    )

def run_flask():
    import logging as _log
    _log.getLogger("werkzeug").setLevel(_log.ERROR)
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

# =====================================================================
# ENTRY POINT — threads start at module level so gunicorn picks them up
# =====================================================================
threading.Thread(target=run_monitor, daemon=True).start()
threading.Thread(target=run_bot,     daemon=True).start()

if __name__ == "__main__":
    run_flask()
