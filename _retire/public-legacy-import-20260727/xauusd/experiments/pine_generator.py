"""
Generates Pine Script v6 strategy files for all 20 experiments.
Each file is a self-contained TradingView strategy with:
    - Same SL/TP/time-exit logic as the Python engine
    - Group and description in the header
    - Ready to copy into TradingView for full historical backtesting

Output: XAUUSD-Long-Experiments/pine/E##_<name>.pine  (20 files)
"""
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "XAUUSD-Long-Experiments" / "pine"

HEADER = """\
// © Tittan — XAUUSD Long-Only Experiment Series
// Strategy: {title}
// Group   : {group}
// Logic   : {description}
// Risk    : SL 0.5% / TP 1.0% / Max hold 48 bars (24h)
// Version : 1.0  (proof-of-concept — validate with full TradingView history)
//@version=6
strategy(title="{title}",
         shorttitle="{shortid}",
         overlay=true,
         format=format.price,
         precision=2,
         default_qty_type=strategy.cash,
         initial_capital=10000,
         default_qty_value=10000,
         commission_value=0.0,
         slippage=1)

// ── Risk parameters ──────────────────────────────────────────────────────────
SL_PCT = input.float(0.5, "Stop Loss %", step=0.1) / 100
TP_PCT = input.float(1.0, "Take Profit %", step=0.1) / 100
MAX_BARS = input.int(48, "Max Hold Bars")

// ── Position tracking ─────────────────────────────────────────────────────────
var int entry_bar = na
MP = strategy.position_size > 0 ? 1 : 0

// Time-based exit
if MP == 1 and not na(entry_bar) and (bar_index - entry_bar) >= MAX_BARS
    strategy.close("Long", comment="TIME")
    entry_bar := na

"""

FOOTER = """\

// ── Entry execution ───────────────────────────────────────────────────────────
if longSignal and MP == 0
    strategy.entry("Long", strategy.long, comment="LE")
    entry_bar := bar_index

// ── SL / TP ───────────────────────────────────────────────────────────────────
if MP == 1
    ep = strategy.position_avg_price
    strategy.exit("LX", "Long",
                  stop  = ep * (1 - SL_PCT),
                  limit = ep * (1 + TP_PCT),
                  comment_loss = "SL",
                  comment_profit = "TP")

// ── Signal marker ─────────────────────────────────────────────────────────────
plotshape(longSignal and MP == 0, style=shape.triangleup,
          location=location.belowbar, color=color.new(color.green, 0),
          size=size.small, title="Entry Signal")
"""

STRATEGIES_PINE: dict[str, dict] = {
    "E01_EMA_Cross": {
        "group": "Trend",
        "description": "Fast EMA(8) crosses above Slow EMA(21)",
        "body": """\
// ── Indicators ────────────────────────────────────────────────────────────────
fast_ema = ta.ema(close, 8)
slow_ema = ta.ema(close, 21)

plot(fast_ema, "Fast EMA(8)",  color=color.new(color.blue,   0), linewidth=1)
plot(slow_ema, "Slow EMA(21)", color=color.new(color.orange, 0), linewidth=1)

// ── Signal ────────────────────────────────────────────────────────────────────
longSignal = ta.crossover(fast_ema, slow_ema)
""",
    },
    "E02_Triple_EMA": {
        "group": "Trend",
        "description": "Triple EMA alignment (EMA8>EMA21>EMA55) with pullback to EMA8",
        "body": """\
// ── Indicators ────────────────────────────────────────────────────────────────
e8  = ta.ema(close, 8)
e21 = ta.ema(close, 21)
e55 = ta.ema(close, 55)

plot(e8,  "EMA8",  color=color.new(color.blue,   0), linewidth=1)
plot(e21, "EMA21", color=color.new(color.orange, 0), linewidth=1)
plot(e55, "EMA55", color=color.new(color.red,    0), linewidth=1)

// ── Signal ────────────────────────────────────────────────────────────────────
aligned   = e8 > e21 and e21 > e55
pullback  = close[1] <= e8[1] * 1.002
bounce    = close > close[1]
longSignal = aligned and pullback and bounce
""",
    },
    "E03_MACD_Signal": {
        "group": "Momentum",
        "description": "MACD histogram crosses from negative to positive",
        "body": """\
// ── Indicators ────────────────────────────────────────────────────────────────
[ml, sl, hist] = ta.macd(close, 12, 26, 9)

hline(0, "Zero", color=color.gray)
plot(hist, "MACD Hist", style=plot.style_histogram,
     color=hist >= 0 ? color.new(color.green,40) : color.new(color.red,40))

// ── Signal ────────────────────────────────────────────────────────────────────
longSignal = ta.crossover(hist, 0) and ml > -5
""",
    },
    "E04_Supertrend": {
        "group": "Trend",
        "description": "Supertrend(10,3) flips from bearish to bullish",
        "body": """\
// ── Indicators ────────────────────────────────────────────────────────────────
[st_line, st_dir] = ta.supertrend(3.0, 10)
bullish = st_dir < 0

plot(bullish ? st_line : na, "ST Bull", color=color.new(color.green,0), linewidth=2)
plot(bullish ? na : st_line, "ST Bear", color=color.new(color.red,0),   linewidth=2)

// ── Signal ────────────────────────────────────────────────────────────────────
longSignal = bullish and not bullish[1]   // just flipped bullish
""",
    },
    "E05_ADX_EMA_Dip": {
        "group": "Trend",
        "description": "ADX>25 strong trend + EMA20 dip-and-bounce entry",
        "body": """\
// ── Indicators ────────────────────────────────────────────────────────────────
[adx_val, di_plus, di_minus] = ta.dmi(14, 14)
e20 = ta.ema(close, 20)
e50 = ta.ema(close, 50)

plot(e20, "EMA20", color=color.new(color.blue,  0), linewidth=1)
plot(e50, "EMA50", color=color.new(color.orange,0), linewidth=1)

// ── Signal ────────────────────────────────────────────────────────────────────
trending   = adx_val > 25
above_e50  = close > e50
dip_bounce = close[1] < e20[1] and close > e20
longSignal = trending and above_e50 and dip_bounce
""",
    },
    "E06_RSI_Oversold": {
        "group": "Oscillator",
        "description": "RSI(14) crosses above 30 from oversold zone",
        "body": """\
// ── Indicators ────────────────────────────────────────────────────────────────
rsi_val = ta.rsi(close, 14)

// ── Signal ────────────────────────────────────────────────────────────────────
longSignal = ta.crossover(rsi_val, 30)
""",
    },
    "E07_Stoch_Cross": {
        "group": "Oscillator",
        "description": "Stochastic K crosses above D from below 25",
        "body": """\
// ── Indicators ────────────────────────────────────────────────────────────────
k = ta.sma(ta.stoch(close, high, low, 14), 3)
d = ta.sma(k, 3)

// ── Signal ────────────────────────────────────────────────────────────────────
longSignal = ta.crossover(k, d) and k[1] < 25
""",
    },
    "E08_CCI_Bounce": {
        "group": "Oscillator",
        "description": "CCI(20) crosses above -100 from oversold",
        "body": """\
// ── Indicators ────────────────────────────────────────────────────────────────
cci_val = ta.cci(high, low, close, 20)

// ── Signal ────────────────────────────────────────────────────────────────────
longSignal = ta.crossover(cci_val, -100)
""",
    },
    "E09_Williams_R": {
        "group": "Oscillator",
        "description": "Williams %R(14) exits oversold zone (crosses above -80)",
        "body": """\
// ── Indicators ────────────────────────────────────────────────────────────────
wr = ta.wpr(14)

// ── Signal ────────────────────────────────────────────────────────────────────
longSignal = ta.crossover(wr, -80)
""",
    },
    "E10_RSI_Divergence": {
        "group": "Oscillator",
        "description": "Bullish RSI divergence proxy: price lower low, RSI higher low",
        "body": """\
// ── Indicators ────────────────────────────────────────────────────────────────
rsi_val = ta.rsi(close, 14)
look = 5

price_ll = close < ta.lowest(close, look)[1]
rsi_hl   = rsi_val > ta.lowest(rsi_val, look)[1]
rsi_rise = rsi_val > rsi_val[1]

// ── Signal ────────────────────────────────────────────────────────────────────
longSignal = price_ll and rsi_hl and rsi_rise
""",
    },
    "E11_BB_Lower_Touch": {
        "group": "BB",
        "description": "Price touches BB lower band then closes above it (mean reversion)",
        "body": """\
// ── Indicators ────────────────────────────────────────────────────────────────
[basis, upper, lower] = ta.bb(close, 20, 2.0)
rsi_val = ta.rsi(close, 14)

plot(basis, "BB Basis", color=color.new(color.gray,  50))
plot(upper, "BB Upper", color=color.new(color.blue,  50))
plot(lower, "BB Lower", color=color.new(color.orange,50))

// ── Signal ────────────────────────────────────────────────────────────────────
touched  = low  <= lower
recovery = close > lower
bullish  = close > close[1]
longSignal = touched and recovery and bullish and rsi_val < 50
""",
    },
    "E12_BB_Squeeze_Break": {
        "group": "BB",
        "description": "BB squeeze (narrow width) followed by upward expansion",
        "body": """\
// ── Indicators ────────────────────────────────────────────────────────────────
[basis, upper, lower] = ta.bb(close, 20, 2.0)
width     = upper - lower
avg_width = ta.sma(width, 20)

plot(basis, "BB Basis", color=color.new(color.gray,50))
plot(upper, "BB Upper", color=color.new(color.blue,50))
plot(lower, "BB Lower", color=color.new(color.orange,50))

// ── Signal ────────────────────────────────────────────────────────────────────
was_squeeze = width[1] < avg_width[1] * 0.85
expanding   = width    > width[1]
above_mid   = close    > basis
longSignal = was_squeeze and expanding and above_mid
""",
    },
    "E13_BB_Basis_Walk": {
        "group": "BB",
        "description": "Pullback to BB basis in an uptrend, then recovery above basis",
        "body": """\
// ── Indicators ────────────────────────────────────────────────────────────────
[basis, upper, lower] = ta.bb(close, 20, 2.0)

plot(basis, "BB Basis", color=color.new(color.gray,  50))
plot(upper, "BB Upper", color=color.new(color.blue,  50))
plot(lower, "BB Lower", color=color.new(color.orange,50))

// ── Signal ────────────────────────────────────────────────────────────────────
slope_up = basis > basis[3]
dip      = close[1] <= basis[1] * 1.001
recovery = close    >  basis
longSignal = slope_up and dip and recovery
""",
    },
    "E14_Donchian_Break": {
        "group": "Breakout",
        "description": "Price breaks above the 20-bar Donchian channel high",
        "body": """\
// ── Indicators ────────────────────────────────────────────────────────────────
dc_high = ta.highest(high[1], 20)   // prior 20 bars (exclude current)

plot(dc_high, "Donchian High", color=color.new(color.purple, 30), linewidth=1)

// ── Signal ────────────────────────────────────────────────────────────────────
longSignal = close > dc_high and close > close[1]
""",
    },
    "E15_Inside_Bar_Break": {
        "group": "Breakout",
        "description": "Inside bar consolidation breakout to the upside",
        "body": """\
// ── Signal ────────────────────────────────────────────────────────────────────
// bar[1] is inside bar[2] (mother bar)
inside   = high[1] <= high[2] and low[1] >= low[2]
breakout = close > high[2]
longSignal = inside and breakout
""",
    },
    "E16_ATR_Vol_Break": {
        "group": "Breakout",
        "description": "Price closes >0.8× ATR above prior close with EMA21 trending up",
        "body": """\
// ── Indicators ────────────────────────────────────────────────────────────────
atr_val = ta.atr(14)
e21     = ta.ema(close, 21)

plot(e21, "EMA21", color=color.new(color.orange,0), linewidth=1)

// ── Signal ────────────────────────────────────────────────────────────────────
strong_move = (close - close[1]) > atr_val * 0.8
trend_up    = e21 > e21[3]
longSignal = strong_move and trend_up and close > close[1]
""",
    },
    "E17_AO_Saucer": {
        "group": "Pattern",
        "description": "Awesome Oscillator saucer pattern while AO is positive",
        "body": """\
// ── Indicators ────────────────────────────────────────────────────────────────
ao = ta.sma(hl2, 5) - ta.sma(hl2, 34)

plot(ao, "AO", style=plot.style_histogram,
     color=ao >= 0 ? color.new(color.teal,30) : color.new(color.red,30))
hline(0, color=color.gray)

// ── Signal ────────────────────────────────────────────────────────────────────
saucer = ao > 0 and ao > ao[1] and ao[1] < ao[2]   // dip-and-recovery while positive
longSignal = saucer
""",
    },
    "E18_Hammer": {
        "group": "Pattern",
        "description": "Hammer candle near 20-bar lows (lower wick ≥ 2× body)",
        "body": """\
// ── Signal ────────────────────────────────────────────────────────────────────
body        = math.abs(close - open)
lower_wick  = math.min(close, open) - low
upper_wick  = high - math.max(close, open)
is_hammer   = body > 0 and lower_wick >= 2 * body and upper_wick <= body and close > open

range_low   = ta.lowest(low,  20)
range_high  = ta.highest(high, 20)
range_size  = range_high - range_low
in_lower    = range_size > 0 ? (close - range_low) / range_size < 0.45 : false

longSignal = is_hammer and in_lower
""",
    },
    "E19_Bullish_Engulf": {
        "group": "Pattern",
        "description": "Bullish engulfing candle with RSI confirmation",
        "body": """\
// ── Indicators ────────────────────────────────────────────────────────────────
rsi_val = ta.rsi(close, 14)

// ── Signal ────────────────────────────────────────────────────────────────────
prev_bearish = close[1] < open[1]
curr_bullish = close    > open
engulfs      = open <= close[1] and close >= open[1]
longSignal = prev_bearish and curr_bullish and engulfs and rsi_val < 60
""",
    },
    "E20_Confluence": {
        "group": "Confluence",
        "description": "BB lower touch + RSI<40 + EMA21 uptrend — all three must align",
        "body": """\
// ── Indicators ────────────────────────────────────────────────────────────────
[basis, upper, lower] = ta.bb(close, 20, 2.0)
rsi_val = ta.rsi(close, 14)
e21     = ta.ema(close, 21)

plot(basis, "BB Basis", color=color.new(color.gray,  50))
plot(upper, "BB Upper", color=color.new(color.blue,  50))
plot(lower, "BB Lower", color=color.new(color.orange,50))
plot(e21,   "EMA21",    color=color.new(color.purple, 0), linewidth=1)

// ── Signal ────────────────────────────────────────────────────────────────────
bb_touch  = low <= lower and close > lower
oversold  = rsi_val < 40
trend_up  = e21 > e21[5]
longSignal = bb_touch and oversold and trend_up
""",
    },
}


def generate_all() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for strat_id, cfg in STRATEGIES_PINE.items():
        title = strat_id.replace("_", " ")
        shortid = strat_id[:12]
        content = (
            HEADER.format(
                title=title,
                shortid=shortid,
                group=cfg["group"],
                description=cfg["description"],
            )
            + cfg["body"]
            + FOOTER
        )
        out = OUT_DIR / f"{strat_id}.pine"
        out.write_text(content, encoding="utf-8")
    print(f"  Pine Script files → {OUT_DIR}/ ({len(STRATEGIES_PINE)} files)")


if __name__ == "__main__":
    generate_all()
