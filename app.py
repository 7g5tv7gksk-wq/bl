import os
import time
import logging
import threading
import requests
from flask import Flask

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# =====================================================================
# CONFIGURATION
# =====================================================================
PAPER_TRADING = True
SOL_TRADE_SIZE = 0.1          # SOL per trade

# Pair quality filters
MIN_LIQUIDITY_USD  = 3_000.0
MIN_MARKET_CAP     = 10_000.0
MAX_MARKET_CAP     = 250_000.0
MIN_5M_VOLUME      = 500.0
MAX_MACRO_DRAWDOWN = -25.0    # Reject if 6h or 24h change worse than this

# ── Markov 2.0 parameters ─────────────────────────────────────────────
STRIDE_BARS  = 4              # Non-overlapping window size (4 × 5m = 20-min epochs)
MIN_WINDOWS  = 12             # Min stride windows before Markov signal is trusted (~4 h)
ATR_BULL_MULT = 0.5           # Window return > +0.5×ATR% → BULL
ATR_BEAR_MULT = 0.5           # Window return < -0.5×ATR% → BEAR
MARKOV_THRESHOLD   = 0.20     # P(bull) − P(bear) required to enter via Markov layer
MOMENTUM_THRESHOLD = 0.25     # Threshold for cold-start momentum fallback

# SOL macro regime — now read straight from Jupiter's own token stats for the
# native SOL mint, no separate pair address needed.
SOL_BEAR_H24 = -15.0
SOL_BULL_H24 =  10.0
SOL_MINT     = "So11111111111111111111111111111111111111112"

# Position management
TAKE_PROFIT_PCT      = 0.35
STOP_LOSS_PCT        = 0.12
BLACKLIST_COOLDOWN   = 7200   # 2 hours after a stop-loss
SECURITY_REJECT_COOLDOWN = 14400  # 4 hours after a RugCheck/GMGN fail — mint/freeze
                                    # authority and holder concentration rarely change
                                    # quickly, so re-checking every cycle just burns calls
MAX_POSITIONS        = 1
LOOP_INTERVAL        = 15     # seconds between scan cycles
OHLCV_TTL            = 300    # seconds between GeckoTerminal refreshes once we HAVE candles (5 min = 1 candle)
OHLCV_RETRY_COOLDOWN = 45     # seconds between retries when a fetch/pool-resolution failed or was
                               # rate-limited — short enough to pick up newly-indexed pools quickly,
                               # long enough not to hammer the API every single 15s scan cycle
TRACKER_STALE_SECONDS = 3600   # prune a coin's tracker if we haven't seen it in ANY cycle's
                                # candidate list for this long (and it isn't an open position) —
                                # without this, coin_trackers grows forever (every coin ever
                                # scanned keeps getting refreshed and cached indefinitely, even
                                # after it permanently drops out of Jupiter's recent/trending
                                # lists), which is a real memory leak on a memory-capped host
POOL_RESOLVE_MAX_ATTEMPTS = 5   # give up on a coin's pool after this many genuine "no pool
                                # found" checks (~4 min at OHLCV_RETRY_COOLDOWN spacing) —
                                # without this, a coin still on a pump.fun bonding curve (no
                                # AMM pool yet, maybe never) gets retried forever, burning a
                                # GeckoTerminal call every cycle for a check that can never
                                # succeed. Momentum signal doesn't need a tracker at all, so
                                # giving up costs nothing — only real Markov needs the pool.

# Familiars (set FAMILIARS_API_KEY env var after registering at familiars.family)
FAMILIARS_KEY = os.environ.get("FAMILIARS_API_KEY", "")
FAMILIARS_URL = "https://familiars.family"

# Jupiter (optional — set JUPITER_API_KEY after signing up free at
# https://developers.jup.ag/portal). Keyless requests share a 0.5 RPS bucket
# with every other anonymous caller on the same IP — and Render's free-tier
# IPs are shared across many unrelated services, so that shared bucket can
# get exhausted by traffic that has nothing to do with this bot. A free key
# gives this bot its own dedicated 1 RPS bucket instead.
JUPITER_API_KEY = os.environ.get("JUPITER_API_KEY", "")
JUP_BASE_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def jup_headers():
    h = dict(JUP_BASE_HEADERS)
    if JUPITER_API_KEY:
        h["x-api-key"] = JUPITER_API_KEY
    return h

# Shared backoff for Jupiter calls — one 429 anywhere pauses all Jupiter
# calls for a bit instead of continuing to hammer it every 15s cycle.
_jup_backoff = {"until": 0}
JUP_BACKOFF_SECONDS = 30

def jup_available():
    return time.time() >= _jup_backoff["until"]

def jup_mark_limited():
    _jup_backoff["until"] = time.time() + JUP_BACKOFF_SECONDS

# =====================================================================
# STATE
# =====================================================================
active_positions  = {}    # mint → {symbol, entry_price}
stopped_out_tokens= {}    # mint → timestamp of stop-out
security_rejected = {}    # mint → timestamp of RugCheck/GMGN rejection
coin_trackers     = {}    # mint → CoinMarkovTracker
trade_stats = {"total_closed": 0, "wins": 0, "losses": 0, "net_sol_pnl": 0.0}
_sol_cache = {"state": "SIDEWAYS", "last_check": 0}

# Markov state constants
BULL, SIDEWAYS, BEAR = 0, 1, 2
STATE_NAME = {BULL: "BULL", SIDEWAYS: "SIDEWAYS", BEAR: "BEAR"}


# =====================================================================
# FAMILIARS INTEGRATION
# =====================================================================
def familiars_post(kind, text, mint=None, signature=None):
    """
    Post a callout or trade explanation to the Familiars public feed.
    kind: "callout" | "trade" | "note"
    Trades on-chain show up automatically; this adds our reasoning.
    """
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
    """
    Read owner-set limits (maxPositionUsd, dailyLimitUsd, instructions)
    before every entry. Returns {} if not configured.
    """
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
# Reads Jupiter's own rolling stats for the SOL mint directly — no separate
# pair address needed, same Token API v2 schema as everything else now.
# =====================================================================
def get_sol_regime():
    """Returns "BULL" | "SIDEWAYS" | "BEAR". Cached for OHLCV_TTL seconds."""
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
            data = r.json()
            items = data if isinstance(data, list) else (data.get("data") or data.get("tokens") or [])
            token = next(
                (t for t in items if (t.get("id") or t.get("address") or t.get("mint")) == SOL_MINT),
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
                    f"🌐 [SOL MACRO] {state} | 24h={h24:+.1f}%  6h={h6:+.1f}%  1h={h1:+.1f}%"
                )
    except Exception as e:
        logging.error(f"❌ [SOL REGIME] {e}")

    return _sol_cache["state"]


# =====================================================================
# OHLCV — GECKOTERMINAL  (free, keyless) — still the only source with actual
# historical candle series; Jupiter's stats are snapshots, not a time series.
# =====================================================================
def fetch_ohlcv(pool_address, agg_min=5, limit=200):
    """
    Fetch 5m OHLCV candles for a Solana pool from GeckoTerminal.
    Returns list[dict] oldest-first, or [] if pool not yet indexed.
    """
    url = (
        f"https://api.geckoterminal.com/api/v2/networks/solana"
        f"/pools/{pool_address}/ohlcv/minute"
        f"?aggregate={agg_min}&limit={limit}&currency=usd"
    )
    try:
        r = requests.get(url, headers={"Accept": "application/json"}, timeout=8)
        if r.status_code == 404:
            return []      # Pool not indexed yet — very new pair, fall back to momentum
        if r.status_code == 429:
            # Do NOT block the shared bot thread here — a synchronous sleep
            # inside this function freezes the entire scan loop (SOL regime
            # check, other trackers, everything) for as long as we wait.
            logging.warning(f"⚠️ [GECKO] Rate limited on {pool_address[:8]}…")
            return []
        if r.status_code != 200:
            return []

        raw = r.json().get("data", {}).get("attributes", {}).get("ohlcv_list", [])
        if not raw:
            return []

        # GeckoTerminal returns newest-first; reverse to oldest-first for the engine
        return [
            {"ts": c[0], "o": float(c[1]), "h": float(c[2]),
             "l": float(c[3]), "c": float(c[4]), "v": float(c[5])}
            for c in reversed(raw)
            if c[1] and c[4]   # skip zero/null candles
        ]
    except Exception as e:
        logging.error(f"❌ [GECKO OHLCV] {pool_address[:8]}…: {e}")
        return []


def resolve_pool_address(mint, graduated_pool_hint=None):
    """
    GeckoTerminal's OHLCV endpoint needs a POOL address, not a mint address.
    Jupiter's token object gives us `graduatedPool` for free once a pump.fun
    token has graduated to a full AMM pool — use that with zero extra calls.
    Otherwise, ask GeckoTerminal directly which pool(s) trade this mint and
    take the most liquid one (their tokens/{address}/pools endpoint, already
    ranked by liquidity + volume).

    Returns (pool_address_or_None, was_rate_limited). The caller uses
    was_rate_limited to avoid penalizing a coin just because OUR OWN rate
    limit tripped during the check — only a genuine "checked successfully,
    no pool exists" result should count toward giving up on a coin.
    """
    if graduated_pool_hint:
        return graduated_pool_hint, False

    url = f"https://api.geckoterminal.com/api/v2/networks/solana/tokens/{mint}/pools"
    try:
        r = requests.get(url, headers={"Accept": "application/json"}, timeout=8)
        if r.status_code == 200:
            data = r.json().get("data", [])
            if data:
                return data[0].get("attributes", {}).get("address"), False
            return None, False    # Checked successfully — genuinely no pool (yet)
        if r.status_code == 429:
            logging.warning(f"⚠️ [GECKO POOL] Rate limited resolving pool for {mint[:8]}…")
            return None, True     # Transient — don't count against the give-up counter
        return None, False        # Other error status — count it as a real check
    except Exception as e:
        logging.error(f"❌ [GECKO POOL] {mint[:8]}…: {e}")
        return None, True         # Network hiccup — transient, don't count


# =====================================================================
# MARKOV 2.0 ENGINE — THREE FIXES IMPLEMENTED
# (unchanged — operates purely on candle data regardless of where it came from)
# =====================================================================

def _atr_pct(candles, period=14):
    """
    Average True Range as % of close.
    Used to set adaptive BULL/BEAR thresholds so the same parameters work
    across coins ranging from 0.001% to 50% daily vol.
    """
    if len(candles) < period + 1:
        return 3.0     # Fallback: 3% if not enough history
    trs = []
    for i in range(1, len(candles)):
        h, l, pc = candles[i]["h"], candles[i]["l"], candles[i-1]["c"]
        if pc == 0:
            continue
        trs.append(max(h - l, abs(h - pc), abs(l - pc)) / pc * 100)
    tail = trs[-period:]
    return sum(tail) / len(tail) if tail else 3.0


def _label_states(candles, stride, bull_mult, bear_mult):
    """
    FIX 1 — Stride sampling.

    NEVER build the matrix from overlapping windows.
    Consecutive 20-day windows share 19 days → fakes persistence on the diagonal.
    Here we step through NON-OVERLAPPING windows of `stride` bars.

    Threshold is ATR-adaptive: a coin moving 50% daily needs much wider bands
    than a coin moving 2% daily. Same parameters, different coins.

    Returns: (states[], bull_threshold%, bear_threshold%)
    """
    atr        = _atr_pct(candles)
    bull_thresh = atr * bull_mult
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
    """
    Build a 3×3 transition probability matrix.
    Rows = from-state, Cols = to-state, each row sums to 1.
    Returns (matrix, stickiness_dict).
    Stickiness = diagonal values — how likely each regime is to persist.
    """
    counts = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(len(states) - 1):
        counts[states[i]][states[i+1]] += 1

    matrix = []
    for row in counts:
        total = sum(row)
        matrix.append([v / total if total else 1/3 for v in row])

    stickiness = {STATE_NAME[s]: round(matrix[s][s], 3) for s in (BULL, SIDEWAYS, BEAR)}
    return matrix, stickiness


def _verify_labels(states, candles, stride):
    """
    FIX 2 — Label verification.

    After building any matrix, self-check the state labels against
    three known windows (first, middle, last). A large positive return
    labelled BEAR — or vice versa — means the threshold calibration is off.
    Logs a warning; the engine continues but flags lower confidence.
    """
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
    """
    Signal = P(BULL tomorrow | current state) − P(BEAR tomorrow | current state).
    Range: −1.0 to +1.0. Positive = bullish conviction.
    """
    return round(matrix[current_state][BULL] - matrix[current_state][BEAR], 4)


# =====================================================================
# PER-COIN MARKOV TRACKER
# =====================================================================
class CoinMarkovTracker:
    """
    Accumulates 5m OHLCV history for one coin and maintains a live Markov signal.
    Cold starts gracefully: signal is None until MIN_WINDOWS stride windows exist.
    The bot falls back to the momentum signal in the meantime.
    """

    def __init__(self, mint, symbol, graduated_pool_hint=None):
        self.mint            = mint
        self.symbol          = symbol
        self.graduated_pool_hint = graduated_pool_hint
        self.pool_address    = None
        self.pool_last_try   = 0
        self.pool_resolve_attempts = 0
        self.pool_abandoned  = False
        self.candles         = []
        self.states          = []
        self.matrix          = None
        self.stickiness      = None
        self.signal          = None
        self.current_state   = SIDEWAYS
        self.windows         = 0
        self.last_fetch       = 0
        self.verified         = False
        self.last_seen        = time.time()   # updated whenever this mint appears in a scan cycle

    @property
    def ready(self):
        """True once the matrix has enough data to be meaningful."""
        return self.windows >= MIN_WINDOWS and self.signal is not None

    @property
    def confidence(self):
        """
        0 → MIN_WINDOWS: 0.0 (not ready)
        MIN_WINDOWS → 4×MIN_WINDOWS: 0.0 → 1.0 (growing confidence)
        """
        if self.windows < MIN_WINDOWS:
            return 0.0
        return min((self.windows - MIN_WINDOWS) / (MIN_WINDOWS * 3), 1.0)

    def refresh(self):
        """
        First resolve a pool address if we don't have one yet (free if the
        token already graduated on pump.fun; otherwise one GeckoTerminal
        lookup, retried on OHLCV_RETRY_COOLDOWN so it doesn't hammer the API
        every single 15s cycle while waiting). Gives up after
        POOL_RESOLVE_MAX_ATTEMPTS genuine "no pool" checks — a coin still on
        a bonding curve may never have one, and momentum fallback doesn't
        need this tracker at all, so continuing to retry forever is pure waste.

        Then pull candles on OHLCV_TTL once we have working data, or
        OHLCV_RETRY_COOLDOWN while we don't (still-indexing pool or a
        rate limit).
        """
        if self.pool_abandoned:
            return    # Gave up — no pool after several genuine checks; momentum-only forever

        if not self.pool_address:
            if time.time() - self.pool_last_try < OHLCV_RETRY_COOLDOWN:
                return
            self.pool_last_try = time.time()
            addr, was_rate_limited = resolve_pool_address(self.mint, self.graduated_pool_hint)
            if addr:
                self.pool_address = addr
            else:
                if not was_rate_limited:
                    self.pool_resolve_attempts += 1
                    if self.pool_resolve_attempts >= POOL_RESOLVE_MAX_ATTEMPTS:
                        self.pool_abandoned = True
                        logging.info(
                            f"📉 [POOL] Giving up on ${self.symbol} — "
                            f"no pool after {self.pool_resolve_attempts} checks, momentum-only"
                        )
                return    # Still no pool — retry after cooldown (unless just abandoned)

        ttl = OHLCV_TTL if self.candles else OHLCV_RETRY_COOLDOWN
        if time.time() - self.last_fetch < ttl:
            return

        candles = fetch_ohlcv(self.pool_address)
        self.last_fetch = time.time()   # stamp the attempt whether it succeeded or not

        if not candles:
            return    # Pool not indexed yet or rate-limited — retry after cooldown

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
# Both scraping AND market data come from the same call now: Jupiter's
# discovery endpoints return full token objects (liquidity, mcap, volume,
# price change, buy/sell counts, mint/freeze authority) — no separate
# batch pair-data fetch needed like the old DexScreener pipeline required.
# =====================================================================
def fetch_jupiter_candidates():
    """Returns dict: {mint: token_object}."""
    token_map = {}
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
                data = r.json()
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
                if added:
                    note = "ok"
                else:
                    shape = list(data.keys()) if isinstance(data, dict) else "list"
                    note = f"0 items, keys={shape}"
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
# TOKEN FILTERS  (Jupiter Token API v2 schema)
# =====================================================================
def validate_token(token):
    """Enforces liquidity, MC band, volume, drawdown, and buy/sell ratio."""
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

    vol5m = float(stats5m.get("buyVolume") or 0) + float(stats5m.get("sellVolume") or 0)
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
# LAYER 2 — MOMENTUM SIGNAL  (cold-start fallback)
# Used when GeckoTerminal hasn't indexed the pool yet or data < MIN_WINDOWS.
# =====================================================================
def compute_momentum_signal(token):
    """
    Velocity-acceleration signal built from Jupiter's rolling stats.
    Works on any token immediately with no OHLCV history.
    Returns float signal or None (rejects anti-top-blast conditions).
    """
    stats5m = token.get("stats5m") or {}
    stats1h = token.get("stats1h") or {}
    stats6h = token.get("stats6h") or {}

    m5 = float(stats5m.get("priceChange") or 0)
    h1 = float(stats1h.get("priceChange") or 0)
    h6 = float(stats6h.get("priceChange") or 0)

    if h1 > 50 or m5 > 30:
        return None   # Anti-top-blast: already ripping — skip

    v_m5 = m5 / 5.0
    v_h1 = h1 / 60.0
    v_h6 = h6 / 360.0

    delta_v   = v_m5 - v_h1
    stability = 1.0 if abs(v_h1 - v_h6) < 0.5 else 0.5
    S = (delta_v * 0.6 + v_m5 * 0.4) * stability

    vol5m  = float(stats5m.get("buyVolume") or 0) + float(stats5m.get("sellVolume") or 0)
    vol_wt = min(max(vol5m / 1000.0, 0.5), 1.5)
    return round(S * vol_wt, 2)


# =====================================================================
# SECURITY CHECKS  (unchanged — independent of the data-source migration)
# =====================================================================
def check_gmgn(mint):
    """Reject if bundler cluster > 10% or rug ratio > 0.30."""
    try:
        r = requests.get(
            f"https://gmgn.ai/defi/quotation/v1/tokens/sol/{mint}",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=5
        )
        if r.status_code == 200:
            tok = r.json().get("data", {}).get("token", {})
            if float(tok.get("bundler_pct", 0) or 0) > 10:
                return False
            if float(tok.get("rug_ratio",   0) or 0) > 0.30:
                return False
    except Exception:
        pass
    return True


def check_rugcheck(mint):
    """Reject Danger/High risk or mint/freeze/concentration flags."""
    try:
        r = requests.get(
            f"https://api.rugcheck.xyz/v1/tokens/{mint}/report/summary",
            timeout=5
        )
        if r.status_code == 200:
            data = r.json()
            if data.get("riskLevel") in ("Danger", "High"):
                return False
            bad = {"Single holder ownership", "High holder concentration",
                   "Mint Authority Enabled", "Freeze Authority Enabled"}
            for risk in data.get("risks", []):
                if risk.get("name") in bad:
                    return False
    except Exception:
        pass
    return True


# =====================================================================
# REAL-TIME PRICE  (Jupiter Price API v2 → Jupiter Token API v2 fallback)
# =====================================================================
def get_price(mint):
    if jup_available():
        try:
            r = requests.get(f"https://api.jup.ag/price/v2?ids={mint}",
                              headers=jup_headers(), timeout=2)
            if r.status_code == 200:
                p = r.json().get("data", {}).get(mint, {}).get("price")
                if p:
                    return float(p)
            elif r.status_code == 429:
                jup_mark_limited()
        except Exception:
            pass
    if jup_available():
        try:
            r = requests.get(
                f"https://api.jup.ag/tokens/v2/search?query={mint}",
                headers=jup_headers(), timeout=3
            )
            if r.status_code == 200:
                data = r.json()
                items = data if isinstance(data, list) else (data.get("data") or data.get("tokens") or [])
                for item in items:
                    if (item.get("address") or item.get("mint") or item.get("id")) == mint:
                        p = item.get("usdPrice")
                        if p:
                            return float(p)
            elif r.status_code == 429:
                jup_mark_limited()
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
                to_close   = []
                pnl_lines  = []
                for mint, info in list(active_positions.items()):
                    price = get_price(mint)
                    if not price or info["entry_price"] == 0:
                        continue

                    pnl    = (price - info["entry_price"]) / info["entry_price"]
                    symbol = info["symbol"]
                    pnl_lines.append(
                        f"${symbol} {pnl*100:+.1f}% "
                        f"(entry ${info['entry_price']:.8f} → ${price:.8f})"
                    )

                    if pnl >= TAKE_PROFIT_PCT:
                        sol_gain = SOL_TRADE_SIZE * pnl
                        trade_stats["total_closed"] += 1
                        trade_stats["wins"]         += 1
                        trade_stats["net_sol_pnl"]  += sol_gain
                        wr = trade_stats["wins"] / trade_stats["total_closed"] * 100
                        logging.info(
                            f"🎯 [TP] ${symbol} +{pnl*100:.1f}% | "
                            f"+{sol_gain:.4f} SOL | WR={wr:.1f}% | "
                            f"Net={trade_stats['net_sol_pnl']:+.4f} SOL"
                        )
                        familiars_post(
                            "trade",
                            f"TP +{pnl*100:.1f}% on ${symbol}. "
                            f"Net={trade_stats['net_sol_pnl']:+.4f} SOL",
                            mint=mint
                        )
                        to_close.append(mint)

                    elif pnl <= -STOP_LOSS_PCT:
                        sol_loss = SOL_TRADE_SIZE * pnl
                        trade_stats["total_closed"] += 1
                        trade_stats["losses"]        += 1
                        trade_stats["net_sol_pnl"]   += sol_loss
                        wr = trade_stats["wins"] / trade_stats["total_closed"] * 100
                        logging.info(
                            f"🛑 [SL] ${symbol} {pnl*100:.1f}% | "
                            f"{sol_loss:.4f} SOL | WR={wr:.1f}% | "
                            f"Net={trade_stats['net_sol_pnl']:+.4f} SOL"
                        )
                        familiars_post(
                            "trade",
                            f"SL {pnl*100:.1f}% on ${symbol}. "
                            f"Blacklisting for 2h.",
                            mint=mint
                        )
                        stopped_out_tokens[mint] = time.time()
                        to_close.append(mint)

                for mint in to_close:
                    active_positions.pop(mint, None)
                    coin_trackers.pop(mint, None)    # Clear stale tracker on exit

                # Unrealized PnL heartbeat — every 10s, not every 1s (avoid log spam)
                if pnl_lines and heartbeat % 10 == 0:
                    logging.info("📟 [PNL] " + " | ".join(pnl_lines))

        except Exception as e:
            logging.error(f"❌ [MONITOR] {e}")
        time.sleep(1)


# =====================================================================
# MAIN TRADING LOOP
# =====================================================================
def run_bot():
    """
    Three-layer entry logic:

    Layer 1 — SOL macro regime (Jupiter's own stats for the SOL mint, cached 5 min)
        BEAR → sit out this cycle entirely

    Layer 2 — Momentum signal (instant, no OHLCV needed)
        Used during cold start or when GeckoTerminal hasn't indexed the pool yet

    Layer 3 — Markov 2.0 signal (GeckoTerminal OHLCV, 4-bar stride, ATR-adaptive)
        Replaces Layer 2 once MIN_WINDOWS stride windows have accumulated

    FIX 3 — STANDALONE mode:
        The Markov/momentum differential IS the strategy.
        There is no separate user strategy being filtered.
    """
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

            # ── Layer 1: SOL macro gate ──────────────────────────────
            sol_regime = get_sol_regime()
            if sol_regime == "BEAR":
                logging.info("🚫 [MACRO] SOL=BEAR — sitting out this cycle")
                time.sleep(LOOP_INTERVAL)
                continue

            # Prune trackers we're done with: either genuinely stale (not seen
            # in a candidate list for a while) or abandoned (gave up on ever
            # finding a pool — no point keeping those around at all).
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

            # Refresh Markov on coins already in our tracker pool
            for tracker in list(coin_trackers.values()):
                tracker.refresh()

            logging.info(
                f"🔎 [SCAN] SOL={sol_regime} | "
                f"Tracking={len(coin_trackers)} coins | "
                f"Active={len(active_positions)}"
            )

            # Discovery — Jupiter gives us candidates AND their market data
            # in the same call, so there's no separate batch-fetch step.
            token_map = fetch_jupiter_candidates()
            mints     = list(token_map.keys())

            # Pass 1 — evaluate every candidate in the batch and collect the
            # ones that clear their threshold. We don't enter yet: we want to
            # rank the whole cycle first and take the strongest signal(s),
            # not just whichever mint happened to come first in the list.
            candidates = []

            # Tally WHY candidates drop out each cycle — without this, "no
            # entries" and "no token data at all" look identical in the logs.
            tally = {
                "total": len(mints), "in_position_or_blacklist": 0,
                "security_blacklist_skip": 0,
                "no_token_data": 0, "failed_filters": 0,
                "security_rejected": 0, "no_signal": 0,
                "below_threshold": 0, "qualified": 0,
            }

            for mint in mints:
                if mint in active_positions:
                    tally["in_position_or_blacklist"] += 1
                    continue

                # Blacklist check
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

                symbol              = token.get("symbol", "?")
                graduated_pool_hint = token.get("graduatedPool")
                price               = float(token.get("usdPrice") or 0)
                mc                  = float(token.get("mcap") or token.get("fdv") or 0)

                # Security blacklist check — skip re-querying RugCheck/GMGN for
                # a mint we already know failed recently.
                if mint in security_rejected:
                    if time.time() - security_rejected[mint] < SECURITY_REJECT_COOLDOWN:
                        tally["security_blacklist_skip"] += 1
                        continue
                    del security_rejected[mint]

                # Security screen
                if not check_rugcheck(mint):
                    logging.info(f"🛡️ [REJECTED] ${symbol} — RugCheck fail")
                    tally["security_rejected"] += 1
                    security_rejected[mint] = time.time()
                    continue
                if not check_gmgn(mint):
                    logging.info(f"🛡️ [REJECTED] ${symbol} — GMGN fail")
                    tally["security_rejected"] += 1
                    security_rejected[mint] = time.time()
                    continue

                # ── Layers 2 / 3: signal selection ──────────────────
                if mint not in coin_trackers:
                    coin_trackers[mint] = CoinMarkovTracker(mint, symbol, graduated_pool_hint)

                tracker = coin_trackers[mint]
                tracker.last_seen = time.time()   # still showing up in Jupiter's lists
                tracker.refresh()

                if tracker.ready:
                    # Layer 3: real Markov 2.0
                    signal     = tracker.signal
                    signal_src = f"Markov(conf={tracker.confidence:.0%})"
                    threshold  = MARKOV_THRESHOLD
                else:
                    # Layer 2: momentum fallback (cold start)
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
                    "mint": mint, "symbol": symbol, "price": price, "mc": mc,
                    "signal": signal, "signal_src": signal_src,
                    "ready": tracker.ready,
                    "state": tracker.current_state if tracker.ready else None,
                    "stickiness": tracker.stickiness if tracker.ready else None,
                })

            # Always log the funnel — this is what tells us WHY a cycle
            # produced no entries: no candidates, no market data, strict
            # filters, security rejections, or just no signal yet.
            if mints:
                logging.info(
                    f"🔍 [FUNNEL] {tally['total']} candidates → "
                    f"skip={tally['in_position_or_blacklist']} | "
                    f"security_blacklist={tally['security_blacklist_skip']} | "
                    f"no_token_data={tally['no_token_data']} | "
                    f"failed_filters={tally['failed_filters']} | "
                    f"security_rejected={tally['security_rejected']} | "
                    f"no_signal={tally['no_signal']} | "
                    f"below_threshold={tally['below_threshold']} | "
                    f"qualified={tally['qualified']}"
                )

            # Pass 2 — rank qualifying candidates.
            # IMPORTANT: Markov signal is bounded to [-1, +1] by construction
            # (it's a probability differential). Momentum signal is NOT bounded
            # — it can read -3 or +3 depending on how sharp the move was. Sorting
            # both on raw signal value would let a noisy cold-start momentum
            # read (e.g. +2.5) outrank a real, statistically-grounded Markov
            # signal (e.g. +0.35), which is backwards: Markov is the trusted
            # layer, momentum is only a stand-in until enough history exists.
            # So we sort in two tiers: all Markov-ready candidates first
            # (by signal, strongest first), then momentum-only candidates
            # (by signal, strongest first).
            if candidates:
                candidates.sort(key=lambda c: (c["ready"], c["signal"]), reverse=True)
                top = " | ".join(
                    f"${c['symbol']} {c['signal']:+.3f}"
                    f"{'[M]' if c['ready'] else '[mom]'}"
                    for c in candidates[:5]
                )
                logging.info(
                    f"🏆 [RANKING] {len(candidates)} qualified this cycle | Top: {top}"
                )

            slots_open = MAX_POSITIONS - len(active_positions)

            for cand in candidates:
                if slots_open <= 0:
                    break
                mint = cand["mint"]
                if mint in active_positions:      # could've filled since ranking
                    continue

                # ── Familiars owner limit check ──────────────────────
                limits      = familiars_limits()
                max_pos_usd = limits.get("maxPositionUsd")
                if max_pos_usd:
                    sol_price = get_price(SOL_MINT) or 150   # live price; rough fallback
                    trade_usd = SOL_TRADE_SIZE * sol_price
                    if trade_usd > float(max_pos_usd):
                        logging.warning(
                            f"⛔ [LIMITS] Trade ~${trade_usd:.0f} "
                            f"exceeds owner cap ${max_pos_usd}"
                        )
                        continue

                # ── Entry ────────────────────────────────────────────
                reason_parts = [
                    f"${cand['symbol']}", f"MC=${cand['mc']:,.0f}",
                    f"SOL={sol_regime}", f"Signal={cand['signal']:+.3f} [{cand['signal_src']}]"
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
                        "symbol": cand["symbol"], "entry_price": cand["price"]
                    }
                    logging.info(
                        f"💰 [PAPER] BUY {SOL_TRADE_SIZE} SOL → "
                        f"${cand['symbol']} @ ${cand['price']:.8f}"
                    )
                # ── Live execution stub ──────────────────────────────
                # When ready: set PAPER_TRADING = False and add Jupiter swap here
                # jupiter_swap(mint, SOL_TRADE_SIZE, slippage_bps=100)

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
    positions = len(active_positions)
    return f"Markov 2.0 Active | Positions={positions} | Stats={trade_stats}", 200


def run_flask():
    import logging as _log
    _log.getLogger("werkzeug").setLevel(_log.ERROR)
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)


# =====================================================================
# ENTRY POINT
#
# IMPORTANT: gunicorn imports this module and never executes the
# `if __name__ == "__main__":` block below — it just grabs the `app`
# object. So the background threads are started unconditionally here,
# at import time, so they run under gunicorn too.
#
# Render sets WEB_CONCURRENCY=1 automatically (confirmed in your deploy
# logs). Keep it at 1 worker — with more than one, gunicorn would spawn
# multiple copies of this module, meaning multiple scanner loops and
# multiple monitor threads all trading independently against the same
# limits.
# =====================================================================
threading.Thread(target=run_monitor, daemon=True).start()
threading.Thread(target=run_bot,     daemon=True).start()

if __name__ == "__main__":
    # Only reached with `python app.py` directly (local testing).
    # In production, gunicorn serves `app` itself — no need for run_flask().
    run_flask()
