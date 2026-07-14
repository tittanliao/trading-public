"""
Generates Pine Script v6 strategy files for all 20 short-side experiments.
Output: XAUUSD-Short-Experiments/pine/S##_<name>.pine  (20 files)
"""
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "XAUUSD-Short-Experiments" / "pine"

HEADER = """\
// © Tittan — XAUUSD Short-Side Experiment Series
// Strategy: {title}
// Group   : {group}
// Logic   : {description}
// Risk    : SL 0.5% above entry / TP 1.0% below entry / Max hold 48 bars (24h)
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
SL_PCT  = input.float(0.5, "Stop Loss %",   step=0.1) / 100
TP_PCT  = input.float(1.0, "Take Profit %", step=0.1) / 100
MAX_BARS = input.int(48,   "Max Hold Bars")

// ── Position tracking ─────────────────────────────────────────────────────────
var int entry_bar = na
MP = strategy.position_size < 0 ? 1 : 0

// Time-based exit
if MP == 1 and not na(entry_bar) and (bar_index - entry_bar) >= MAX_BARS
    strategy.close("Short", comment="TIME")
    entry_bar := na

"""

FOOTER = """\

// ── Entry execution ───────────────────────────────────────────────────────────
if shortSignal and MP == 0
    strategy.entry("Short", strategy.short, comment="SE")
    entry_bar := bar_index

// ── SL / TP ───────────────────────────────────────────────────────────────────
if MP == 1
    ep = strategy.position_avg_price
    strategy.exit("SX", "Short",
                  stop  = ep * (1 + SL_PCT),
                  limit = ep * (1 - TP_PCT),
                  comment="SX")

// ── Chart markers ─────────────────────────────────────────────────────────────
plotshape(shortSignal, style=shape.triangledown, location=location.abovebar,
          color=color.red, size=size.small, title="Short Signal")
"""

PINE_BODIES: dict[str, str] = {

"S01_EMA_Cross": """\
// Fast EMA(8) crosses below Slow EMA(21)
fastEMA = ta.ema(close, 8)
slowEMA = ta.ema(close, 21)
shortSignal = ta.crossunder(fastEMA, slowEMA)
plot(fastEMA, color=color.orange, linewidth=1, title="Fast EMA 8")
plot(slowEMA, color=color.red,    linewidth=1, title="Slow EMA 21")
""",

"S02_Triple_EMA": """\
// Bearish triple EMA: E8 < E21 < E55, rejection at E8
e8  = ta.ema(close, 8)
e21 = ta.ema(close, 21)
e55 = ta.ema(close, 55)
aligned   = e8 < e21 and e21 < e55
rejection = close[1] >= e8[1] * 0.998 and close < close[1]
shortSignal = aligned and rejection
plot(e8,  color=color.orange, linewidth=1, title="EMA 8")
plot(e21, color=color.red,    linewidth=1, title="EMA 21")
plot(e55, color=color.maroon, linewidth=2, title="EMA 55")
""",

"S03_MACD_Signal": """\
// MACD histogram turns negative
[macdLine, signalLine, hist] = ta.macd(close, 12, 26, 9)
shortSignal = hist < 0 and hist[1] >= 0 and macdLine < 5
hColor = hist >= 0 ? color.green : color.red
plot(hist, style=plot.style_histogram, color=hColor, title="MACD Hist")
""",

"S04_Supertrend": """\
// Supertrend(10,3) flips bearish
atrMult = 3.0
atrLen  = 10
[st, dir] = ta.supertrend(atrMult, atrLen)
shortSignal = dir == 1 and dir[1] == -1   // Pine: 1=bearish, -1=bullish
plot(st, color=dir == 1 ? color.red : color.green, linewidth=2, title="Supertrend")
""",

"S05_ADX_EMA_Reject": """\
// ADX>25 downtrend + EMA20 rejection
[diPlus, diMinus, adxVal] = ta.dmi(14, 14)
e20  = ta.ema(close, 20)
e50  = ta.ema(close, 50)
trending  = adxVal > 25
belowE50  = close < e50
rejection = close[1] > e20[1] and close < e20
shortSignal = trending and belowE50 and rejection
plot(e20, color=color.orange, linewidth=1, title="EMA 20")
plot(e50, color=color.red,    linewidth=2, title="EMA 50")
""",

"S06_RSI_Overbought": """\
// RSI(14) crosses below 70
rsiVal  = ta.rsi(close, 14)
shortSignal = ta.crossunder(rsiVal, 70)
hline(70, "OB", color=color.red,   linestyle=hline.style_dashed)
hline(50, "Mid", color=color.gray, linestyle=hline.style_dotted)
""",

"S07_Stoch_Cross": """\
// Stochastic K crosses below D from overbought zone
kVal = ta.sma(ta.stoch(close, high, low, 14), 3)
dVal = ta.sma(kVal, 3)
shortSignal = ta.crossunder(kVal, dVal) and kVal[1] > 75
""",

"S08_CCI_Overbought": """\
// CCI(20) crosses below +100
cciVal  = ta.cci(high, low, close, 20)
shortSignal = ta.crossunder(cciVal, 100)
hline(100,  "+100", color=color.red,   linestyle=hline.style_dashed)
hline(-100, "-100", color=color.green, linestyle=hline.style_dashed)
""",

"S09_Williams_R": """\
// Williams %R exits overbought (crosses below -20)
wrVal = ta.wpr(14)
shortSignal = ta.crossunder(wrVal, -20)
hline(-20, "OB", color=color.red,   linestyle=hline.style_dashed)
hline(-80, "OS", color=color.green, linestyle=hline.style_dashed)
""",

"S10_RSI_Divergence": """\
// Bearish RSI divergence proxy: price higher high, RSI lower high
rsiVal = ta.rsi(close, 14)
look   = 5
priceHH = close > ta.highest(close, look)[1]
rsiLH   = rsiVal < ta.highest(rsiVal, look)[1]
rsiFall = rsiVal < rsiVal[1]
shortSignal = priceHH and rsiLH and rsiFall
""",

"S11_BB_Upper_Touch": """\
// BB upper band touch + bearish close
[basis, upper, lower] = ta.bb(close, 20, 2.0)
rsiVal   = ta.rsi(close, 14)
touched  = high >= upper
recovery = close < upper
bearish  = close < close[1]
shortSignal = touched and recovery and bearish and rsiVal > 50
plot(upper, color=color.red,   linewidth=1, title="BB Upper")
plot(basis, color=color.gray,  linewidth=1, title="BB Basis")
plot(lower, color=color.green, linewidth=1, title="BB Lower")
""",

"S12_BB_Squeeze_Break": """\
// BB squeeze followed by downward expansion
[basis, upper, lower] = ta.bb(close, 20, 2.0)
width    = upper - lower
avgWidth = ta.sma(width, 20)
squeeze  = width[1] < avgWidth[1] * 0.85
expand   = width > width[1]
belowMid = close < basis
shortSignal = squeeze and expand and belowMid
plot(upper, color=color.red,  linewidth=1, title="BB Upper")
plot(basis, color=color.gray, linewidth=1, title="BB Basis")
""",

"S13_BB_Basis_Reject": """\
// Rejection at BB basis in downtrend
[basis, upper, lower] = ta.bb(close, 20, 2.0)
slopeDown  = basis < basis[3]
bounce     = close[1] >= basis[1] * 0.999
rejection  = close < basis
shortSignal = slopeDown and bounce and rejection
plot(basis, color=color.red, linewidth=1, title="BB Basis")
""",

"S14_Donchian_Break": """\
// Price breaks below 20-bar Donchian low
dcLow = ta.lowest(low, 20)[1]
shortSignal = close < dcLow and close < close[1]
plot(dcLow, color=color.red, linewidth=1, style=plot.style_stepline, title="DC Low")
""",

"S15_Inside_Bar_Break": """\
// Inside bar breakdown to downside
motherHigh = high[2]
motherLow  = low[2]
isInside   = high[1] <= motherHigh and low[1] >= motherLow
breakdown  = close < motherLow
shortSignal = isInside and breakdown
""",

"S16_ATR_Vol_Break": """\
// ATR-sized bearish move + EMA21 declining
atrVal = ta.atr(14)
move   = close[1] - close   // positive when price drops
e21    = ta.ema(close, 21)
shortSignal = move > atrVal * 0.8 and e21 < e21[3] and move > 0
plot(e21, color=color.red, linewidth=1, title="EMA 21")
""",

"S17_AO_Saucer": """\
// Bearish AO saucer (negative AO, hill pattern)
ao = ta.sma(hl2, 5) - ta.sma(hl2, 34)
bearSaucer = ao < 0 and ao < ao[1] and ao[1] > ao[2]
shortSignal = bearSaucer
hcolor = ao >= 0 ? color.green : color.red
plot(ao, style=plot.style_histogram, color=hcolor, title="AO")
""",

"S18_Shooting_Star": """\
// Shooting star / hanging man near recent highs
body      = math.abs(close - open)
upperWick = high - math.max(close, open)
lowerWick = math.min(close, open) - low
isStar    = body > 0 and upperWick >= 2 * body and lowerWick <= body and close < open
rangeHigh = ta.highest(high, 20)[1]
rangeLow  = ta.lowest(low, 20)[1]
inUpper   = rangeHigh != rangeLow and (close - rangeLow) / (rangeHigh - rangeLow) > 0.55
shortSignal = isStar and inUpper
""",

"S19_Bearish_Engulf": """\
// Bearish engulfing confirmation
prevBull = close[1] > open[1]
currBear = close < open
engulfs  = open >= close[1] and close <= open[1]
rsiVal   = ta.rsi(close, 14)
rangeHigh = ta.highest(high, 20)[1]
rangeLow  = ta.lowest(low, 20)[1]
inUpper  = rangeHigh != rangeLow and (close - rangeLow) / (rangeHigh - rangeLow) > 0.40
shortSignal = prevBull and currBear and engulfs and rsiVal > 40 and inUpper
""",

"S20_Confluence": """\
// Multi-factor short: BB upper touch + RSI>60 + EMA21 declining
[basis, upper, lower] = ta.bb(close, 20, 2.0)
rsiVal    = ta.rsi(close, 14)
e21       = ta.ema(close, 21)
bbTouch   = high >= upper and close < upper
overbought = rsiVal > 60
trendDown  = e21 < e21[5]
shortSignal = bbTouch and overbought and trendDown
plot(upper, color=color.red,  linewidth=1, title="BB Upper")
plot(e21,   color=color.maroon, linewidth=1, title="EMA 21")
""",
}


def generate_all() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from experiments.strategies_short import STRATEGIES

    for strat_id, (fn, group, description) in STRATEGIES.items():
        body = PINE_BODIES.get(strat_id, "shortSignal = false  // TODO\n")
        content = (
            HEADER.format(
                title=strat_id.replace("_", " "),
                shortid=strat_id,
                group=group,
                description=description,
            )
            + body
            + FOOTER
        )
        out = OUT_DIR / f"{strat_id}.pine"
        out.write_text(content, encoding="utf-8")

    print(f"  Pine Script files -> {OUT_DIR}/ ({len(STRATEGIES)} files)")


if __name__ == "__main__":
    generate_all()
