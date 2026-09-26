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

# SOL macro regime (simple % change rules — no extra API needed)
SOL_BEAR_H24  = -15.0
SOL_BULL_H24  =  10.0
SOL_USDC_PAIR = "HJPjoWUrhoZzkNfRpHuieeFk9WcZWjwy6PBjZ81ngndJ"   # Raydium SOL/USDC

# Position management
TAKE_PROFIT_PCT      = 0.35
STOP_LOSS_PCT        = 0.12
BLACKLIST_COOLDOWN   = 7200   # 2 hours after a stop-loss
MAX_POSITIONS        = 1
LOOP_INTERVAL        = 15     # seconds between scan cycles
OHLCV_TTL            = 300    # seconds between GeckoTerminal refreshes (5 min = 1 candle)

# Familiars (set FAMILIARS_API_KEY env var after registering at familiars.family)
FAMILIARS_KEY = os.environ.get("FAMILIARS_API_KEY", "")
FAMILIARS_URL = "https://familiars.family"

# =====================================================================
# STATE
# =====================================================================
active_positions  = {}    # mint → {symbol, entry_price}
stopped_out_tokens= {}    # mint → timestamp of stop-out
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
# Uses DexScreener % changes on SOL/USDC (free, no key, cached 5 min).
# If SOL is in freefall the bot sits out entirely.
# =====================================================================
def get_sol_regime():
    """Returns "BULL" | "SIDEWAYS" | "BEAR". Cached for OHLCV_TTL seconds."""
    now = time.time()
    if now - _sol_cache["last_check"] < OHLCV_TTL:
        return _sol_cache["state"]

    try:
        r = requests.get(
            f"https://api.dexscreener.com/latest/dex/pairs/solana/{SOL_USDC_PAIR}",
            timeout=5
        )
        if r.status_code == 200:
            pc = r.json().get("pair", {}).get("priceChange", {})
            h24 = float(pc.get("h24", 0) or 0)
            h6  = float(pc.get("h6",  0) or 0)
            h1  = float(pc.get("h1",  0) or 0)

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
# OHLCV — GECKOTERMINAL  (free, keyless)
# =====================================================================
def fetch_ohlcv(pool_address, agg_min=5, limit=200):
    """
    Fetch 5m OHLCV candles for a Solana pool from GeckoTerminal.
    Returns list[dict] oldest-first, or [] if pool not yet indexed.
    DexScreener pairAddress == GeckoTerminal pool address for Raydium/Orca.
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
            logging.warning("⚠️ [GECKO] Rate limited — waiting 10s")
            time.sleep(10)
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


# =====================================================================
# MARKOV 2.0 ENGINE — THREE FIXES IMPLEMENTED
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
    three known windows (first, middle, last).  A large positive return
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
    Range: −1.0 to +1.0.  Positive = bullish conviction.
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

    def __init__(self, mint, pair_address, symbol):
        self.mint         = mint
        self.pair_address = pair_address
        self.symbol       = symbol
        self.candles      = []
        self.states       = []
        self.matrix       = None
        self.stickiness   = None
        self.signal       = None
        self.current_state= SIDEWAYS
        self.windows      = 0
        self.last_fetch   = 0
        self.verified     = False

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
        """Pull latest candles and recompute the Markov signal. No-op if TTL hasn't passed."""
        if time.time() - self.last_fetch < OHLCV_TTL:
            return

        candles = fetch_ohlcv(self.pair_address)
        if not candles:
            return    # Pool not indexed yet — keep existing signal (or None)

        self.candles    = candles
        self.last_fetch = time.time()

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
# CANDIDATE SCRAPER  (multi-source, rate-limit safe)
# =====================================================================
def fetch_organic_candidates():
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    mints = []

    # Source A: DexScreener trending SOL search
    try:
        r = requests.get(
            "https://api.dexscreener.com/latest/dex/search?q=sol",
            headers=headers, timeout=5
        )
        if r.status_code == 200:
            for p in r.json().get("pairs", [])[:20]:
                if p.get("chainId") == "solana":
                    addr = p.get("baseToken", {}).get("address")
                    if addr and addr not in mints:
                        mints.append(addr)
    except Exception as e:
        logging.warning(f"⚠️ [SCRAPER] DexScreener: {e}")

    # Source B: Jupiter recent mints V2
    try:
        r = requests.get(
            "https://api.jup.ag/tokens/v2/recent",
            headers=headers, timeout=5
        )
        if r.status_code == 200:
            data  = r.json()
            items = data if isinstance(data, list) else data.get("tokens", [])
            for item in items[:20]:
                addr = item.get("address") or item.get("mint")
                if addr and addr not in mints:
                    mints.append(addr)
    except Exception as e:
        logging.warning(f"⚠️ [SCRAPER] Jupiter: {e}")

    # Source C: DexScreener token profiles (fallback if A+B thin)
    if len(mints) < 10:
        try:
            r = requests.get(
                "https://api.dexscreener.com/token-profiles/latest/v1",
                headers=headers, timeout=5
            )
            if r.status_code == 200:
                for item in r.json()[:20]:
                    if item.get("chainId") == "solana":
                        addr = item.get("tokenAddress")
                        if addr and addr not in mints:
                            mints.append(addr)
        except Exception as e:
            logging.warning(f"⚠️ [SCRAPER] DexScreener profiles: {e}")

    logging.info(f"📡 [SCRAPER] {len(mints[:30])} candidate mints")
    return mints[:30]


# =====================================================================
# BATCH DEX DATA  (single DexScreener call for up to 30 mints)
# =====================================================================
def fetch_dex_batch(mints):
    if not mints:
        return {}
    url = f"https://api.dexscreener.com/latest/dex/tokens/{','.join(mints[:30])}"
    try:
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            pair_map = {}
            for p in r.json().get("pairs", []) or []:
                addr = p.get("baseToken", {}).get("address")
                if addr and addr not in pair_map:
                    pair_map[addr] = p
            return pair_map
        if r.status_code == 429:
            logging.warning("⚠️ [DEX BATCH] 429 — backing off 15s")
            time.sleep(15)
    except Exception as e:
        logging.error(f"❌ [DEX BATCH] {e}")
    return {}


# =====================================================================
# PAIR FILTERS
# =====================================================================
def validate_pair(pair):
    """Enforces liquidity, MC band, volume, drawdown, and buy/sell ratio."""
    if not pair:
        return False
    liq = float(pair.get("liquidity", {}).get("usd", 0) or 0)
    if liq < MIN_LIQUIDITY_USD:
        return False
    mc = float(pair.get("marketCap") or pair.get("fdv", 0) or 0)
    if mc < MIN_MARKET_CAP or mc > MAX_MARKET_CAP:
        return False
    v5m = float(pair.get("volume", {}).get("m5", 0) or 0)
    if v5m < MIN_5M_VOLUME:
        return False
    pc = pair.get("priceChange", {})
    if float(pc.get("h6",  0) or 0) < MAX_MACRO_DRAWDOWN:
        return False
    if float(pc.get("h24", 0) or 0) < MAX_MACRO_DRAWDOWN:
        return False
    txns = pair.get("txns", {}).get("m5", {})
    buys  = int(txns.get("buys",  0) or 0)
    sells = int(txns.get("sells", 0) or 0)
    if (buys + sells) < 12 or buys < (sells * 1.2):
        return False
    return True


# =====================================================================
# LAYER 2 — MOMENTUM SIGNAL  (cold-start fallback)
# Used when GeckoTerminal hasn't indexed the pair yet or data < MIN_WINDOWS.
# This is the original compute_markov_differential, renamed to be honest.
# =====================================================================
def compute_momentum_signal(pair):
    """
    Velocity-acceleration signal built from DexScreener snapshot % changes.
    Works on any pair immediately with no OHLCV history.
    Returns float signal or None (rejects anti-top-blast conditions).
    """
    pc  = pair.get("priceChange", {})
    m5  = float(pc.get("m5",  0) or 0)
    h1  = float(pc.get("h1",  0) or 0)
    h6  = float(pc.get("h6",  0) or 0)

    if h1 > 50 or m5 > 30:
        return None   # Anti-top-blast: already ripping — skip

    v_m5 = m5  / 5.0
    v_h1 = h1  / 60.0
    v_h6 = h6  / 360.0

    delta_v   = v_m5 - v_h1
    stability = 1.0 if abs(v_h1 - v_h6) < 0.5 else 0.5
    S = (delta_v * 0.6 + v_m5 * 0.4) * stability
    vol_wt = min(max(float(pair.get("volume", {}).get("m5", 0) or 0) / 1000.0, 0.5), 1.5)
    return round(S * vol_wt, 2)


# =====================================================================
# SECURITY CHECKS
# =====================================================================
def check_gmgn(mint):
    """Reject if bundler cluster > 10% or rug ratio > 0.30."""
    try:
        r = requests.get(
            f"https://gmgn.ai/defi/quotation/v1/tokens/sol/{mint}",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=5
        )
        if r.status_code == 200:
            token = r.json().get("data", {}).get("token", {})
            if float(token.get("bundler_pct", 0) or 0) > 10:
                return False
            if float(token.get("rug_ratio",   0) or 0) > 0.30:
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
# REAL-TIME PRICE  (Jupiter V2 → DexScreener fallback)
# =====================================================================
def get_price(mint):
    try:
        r = requests.get(f"https://api.jup.ag/price/v2?ids={mint}", timeout=2)
        if r.status_code == 200:
            p = r.json().get("data", {}).get(mint, {}).get("price")
            if p:
                return float(p)
    except Exception:
        pass
    try:
        r = requests.get(
            f"https://api.dexscreener.com/latest/dex/tokens/{mint}",
            timeout=3
        )
        if r.status_code == 200:
            pairs = r.json().get("pairs", [])
            if pairs:
                return float(pairs[0].get("priceUsd", 0) or 0)
    except Exception:
        pass
    return None


# =====================================================================
# POSITION MONITOR — 1-second background thread
# =====================================================================
def run_monitor():
    logging.info("⚡ [MONITOR] Position monitor started (1s loop)")
    while True:
        try:
            if active_positions:
                to_close = []
                for mint, info in list(active_positions.items()):
                    price = get_price(mint)
                    if not price or info["entry_price"] == 0:
                        continue

                    pnl    = (price - info["entry_price"]) / info["entry_price"]
                    symbol = info["symbol"]

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

        except Exception as e:
            logging.error(f"❌ [MONITOR] {e}")
        time.sleep(1)


# =====================================================================
# MAIN TRADING LOOP
# =====================================================================
def run_bot():
    """
    Three-layer entry logic:

    Layer 1 — SOL macro regime (DexScreener, cached 5 min)
        BEAR → sit out this cycle entirely

    Layer 2 — Momentum signal (instant, no OHLCV needed)
        Used during cold start or when GeckoTerminal hasn't indexed the pair yet

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

            # Refresh Markov on coins already in our tracker pool
            for tracker in list(coin_trackers.values()):
                tracker.refresh()

            logging.info(
                f"🔎 [SCAN] SOL={sol_regime} | "
                f"Tracking={len(coin_trackers)} coins | "
                f"Active={len(active_positions)}"
            )

            # Scrape + batch fetch
            mints    = fetch_organic_candidates()
            pair_map = fetch_dex_batch(mints) if mints else {}

            for mint in mints:
                if len(active_positions) >= MAX_POSITIONS:
                    break
                if mint in active_positions:
                    continue

                # Blacklist check
                if mint in stopped_out_tokens:
                    if time.time() - stopped_out_tokens[mint] < BLACKLIST_COOLDOWN:
                        continue
                    del stopped_out_tokens[mint]

                pair = pair_map.get(mint)
                if not pair or not validate_pair(pair):
                    continue

                symbol       = pair.get("baseToken", {}).get("symbol", "?")
                pair_address = pair.get("pairAddress", "")
                price        = float(pair.get("priceUsd", 0) or 0)
                mc           = float(pair.get("marketCap") or pair.get("fdv", 0) or 0)

                # Security screen
                if not check_rugcheck(mint):
                    logging.info(f"🛡️ [REJECTED] ${symbol} — RugCheck fail")
                    continue
                if not check_gmgn(mint):
                    logging.info(f"🛡️ [REJECTED] ${symbol} — GMGN fail")
                    continue

                # ── Layers 2 / 3: signal selection ──────────────────
                if mint not in coin_trackers:
                    coin_trackers[mint] = CoinMarkovTracker(mint, pair_address, symbol)

                tracker = coin_trackers[mint]
                tracker.refresh()

                if tracker.ready:
                    # Layer 3: real Markov 2.0
                    signal     = tracker.signal
                    signal_src = f"Markov(conf={tracker.confidence:.0%})"
                    threshold  = MARKOV_THRESHOLD
                else:
                    # Layer 2: momentum fallback (cold start)
                    signal     = compute_momentum_signal(pair)
                    signal_src = f"Momentum(cold,w={tracker.windows})"
                    threshold  = MOMENTUM_THRESHOLD

                if signal is None:
                    continue

                logging.info(
                    f"📊 [EVAL] ${symbol} | MC=${mc:,.0f} | "
                    f"S={signal:+.3f} [{signal_src}] | Need>{threshold}"
                )

                if signal < threshold:
                    continue

                # ── Familiars owner limit check ──────────────────────
                limits      = familiars_limits()
                max_pos_usd = limits.get("maxPositionUsd")
                if max_pos_usd:
                    sol_price_est = 150    # rough estimate for limit check
                    trade_usd = SOL_TRADE_SIZE * sol_price_est
                    if trade_usd > float(max_pos_usd):
                        logging.warning(
                            f"⛔ [LIMITS] Trade ~${trade_usd:.0f} "
                            f"exceeds owner cap ${max_pos_usd}"
                        )
                        continue

                # ── Entry ────────────────────────────────────────────
                reason_parts = [
                    f"${symbol}", f"MC=${mc:,.0f}",
                    f"SOL={sol_regime}", f"Signal={signal:+.3f} [{signal_src}]"
                ]
                if tracker.ready:
                    reason_parts += [
                        f"State={STATE_NAME[tracker.current_state]}",
                        f"Stickiness={tracker.stickiness}",
                    ]
                reason = " | ".join(reason_parts)

                logging.info(f"\n🚀 [ENTRY] {reason}")
                familiars_post("callout", f"Entering {reason}", mint=mint)

                if PAPER_TRADING:
                    active_positions[mint] = {"symbol": symbol, "entry_price": price}
                    logging.info(
                        f"💰 [PAPER] BUY {SOL_TRADE_SIZE} SOL → "
                        f"${symbol} @ ${price:.8f}"
                    )
                # ── Live execution stub ──────────────────────────────
                # When ready: set PAPER_TRADING = False and add Jupiter swap here
                # jupiter_swap(mint, SOL_TRADE_SIZE, slippage_bps=100)

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
# =====================================================================
if __name__ == "__main__":
    threading.Thread(target=run_flask,  daemon=True).start()
    threading.Thread(target=run_monitor, daemon=True).start()
    run_bot()
