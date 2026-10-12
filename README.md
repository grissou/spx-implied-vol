# SPX implied vol surface, risk-neutral density and local vol

What the options market on the S&P 500 said about the future on two days: 19 Feb 2020, the day
the index hit its all-time high before Covid, and 16 Mar 2020, the day the VIX fear index closed
at a record 83. From thousands of raw option quotes on each day, this project recovers the
market's implied probability distribution for the index, builds a pricing model that matches every
quoted option, and uses it to price two options whose value depends on the path the index takes.

## The idea in plain terms

- **An option** is a contract paying off if the index finishes above (call) or below (put) a
  chosen level, the strike, at a chosen date, the expiry. Its price is mostly a bet on how much the
  index will move.
- **Implied volatility** is that bet in standard units: the annualised standard deviation of index
  returns that makes the Black-Scholes formula reproduce the quoted price. One number per option.
- **The smile** is implied volatility plotted against strike. For an index it slopes down: puts
  that pay off in a crash cost more than the formula's single volatility would suggest. The slope is
  the skew, and the whole set of smiles across expiries is the volatility surface.
- **The forward** is the price at which you could agree today to buy the index at expiry. It is
  spot adjusted for interest and dividends, and every strike in this project is measured against it.
- **The risk-neutral density** is the probability distribution of the index at expiry that the
  option prices imply. It is what the market is willing to pay for each outcome, not a forecast.
- **Local volatility** is a model in which volatility depends on the index level and time. It is
  the unique such model that reproduces every quoted option price at once, so it is the natural tool
  for pricing options that quotes do not cover.
- **Monte Carlo** simulates many possible index paths under a model and averages the payoff.

## Run

```
python -m venv ~/venvs/spxvol && ~/venvs/spxvol/bin/pip install -e .
~/venvs/spxvol/bin/python scripts/build.py
```

The first run pulls two days from WRDS through `~/.pgpass` and caches them in `data/`, which is
gitignored because OptionMetrics data cannot be redistributed. The full build takes about 30 seconds.
Keep the venv outside cloud-synced folders.

## Step 1: quotes, forwards, implied vols

Raw quotes need cleaning, and the forward has to be known before any implied volatility is
meaningful. I keep quotes with bid > 0, ask > bid, a spread at most 50% of mid, and 7 to 730 days to
expiry. SPX monthlies settle on the Friday open and SPXW weeklies on the close; where both exist for
one date I keep the PM contract, and I shorten T by 6.5 hours for AM-settled expiries. I use
out-of-the-money quotes only. An in-the-money option's time value is a tiny fraction of its price,
and a deep in-the-money call at 8% vol turned out to have time value below float precision, so its
implied vol is numerically meaningless.

A call and a put at the same strike obey put-call parity, C - P = D (F - K), with D the discount
factor from the zero curve. That gives the forward F per expiry from strikes within 10% of spot
without any model. Implied vols come from Black-76 on the forward, inverted with Brent's method.

![forwards](figures/step1_forwards.png)

- A joint regression of C - P on K recovers the same forward to 0.01% but its discount factor is
  noise below three months, so I take D from the zero curve and use parity for F only.
- On 16 March the parity forward is 2.5% below spot at 1.75 years, i.e. the market still priced
  about 2% dividends, while OptionMetrics' dividend yield reads 0.4%. Their implied vols differ from
  ours by 0.7 vol points with opposite signs on puts and calls, which is what a wrong forward looks
  like: it pushes puts and calls in opposite directions. Given their forward, my solver reproduces
  their vols to 0.03 vol points, so the solver is exact and the forward is the whole disagreement.

## Step 2: SVI surface

Quotes exist only at listed strikes and expiries. A smooth curve through them is needed to take
derivatives (Step 3) and to price options at arbitrary levels (Step 4), and it must not admit
arbitrage, i.e. combinations of options with negative price.

**Model.** Each expiry is a raw SVI slice ("stochastic volatility inspired", Gatheral 2004) in
total variance w = sigma^2 T against log-moneyness k = ln(K/F):

    w(k) = a + b [ rho (k - m) + sqrt((k - m)^2 + sigma^2) ]

a sets the level, b the wing slopes, rho the asymmetry (negative for equity skew), m shifts the
smile, sigma rounds the kink. Wings are linear in k, consistent with Lee's moment bounds.

**Fit.** Bounded least squares in w on strikes from 55% to 135% of the forward, the range that
holds the density mass and the Step 4 payoffs. I weight each quote by 1/var(w) implied by its
bid-ask (half-spread divided by vega, times dw/dsigma), so tight near-the-money quotes dominate.
Unweighted fits in w do the opposite: the wings have the largest w and pull the fit away from the
smile minimum. Nine starting points over (rho, m) guard against local minima.

Two constraints, both learned the hard way. First, a >= 0 and m inside the quoted k range. My first
fit on live data came back with rho = +0.93 and m = 0.69: the optimiser had matched a steep put skew
using the far branch of the hyperbola, which fits the quotes and then extrapolates to a density with
mass 1.04 and positive skew. Second, sigma >= 0.25 sigma_ATM sqrt(T). SVI is convex in w and the
stress-day put wing is concave, so without a floor the optimiser answers with a straight line and a
zero-width kink at the edge of the range, which is a point mass in the density at 1.3 times the
forward. The floor rounds the kink at a cost of 0.3 vol points on the stress day and nothing on the
calm day, because the floor scales with the smile width.

**Checks.** Butterfly arbitrage (a negative probability somewhere): Gatheral's g(k) >= 0 on every
slice, since g is proportional to the density in k. Calendar arbitrage (a longer option cheaper
than a shorter one): w must not decrease in T at any k; adjacent slices are compared on the fitted
range and violations reported. The surface used downstream interpolates w linearly in T between
slices after a running maximum in T removes the violations.

![smiles](figures/step2_smiles_2020-03-16.png)
![surface](figures/step2_surface.png)

- Mean fit error 0.8 vol points on the calm day, 2.2 on the stress day. The residual pattern is the
  same on every stress slice: the fit sits above the market in both wings and below it near the money.
  SVI is convex in total variance and that smile is not.
- No butterfly violations. Calendar violations are small (worst 0.0014 in total variance on the
  calm day, 0.015 on the stress day) and sit at the edges of the fitted range.

## Step 3: risk-neutral density

Breeden and Litzenberger (1978): the second derivative of the call price with respect to strike is
the discounted probability density of the index at expiry. Intuitively, a tight spread of calls
pays off only if the index lands in a narrow band, so its price is the probability of that band.
From the slice's call prices C(K) on a strike grid from 0.05F to 4F, q(K) = (1/D) d2C/dK2 by
central differences. Mass integrates to 1.000 and the mean equals the forward on every slice. The
benchmark is the lognormal distribution Black-Scholes assumes, with the same forward and the ATM vol.

![density](figures/step3_density.png)

| day | T | ATM vol | log-return skew | excess kurtosis | P(S < 0.9F) RN / LN | P(S < 0.8F) RN / LN |
|---|---|---|---|---|---|---|
| calm | 1M | 11% | -3.4 | 23 | 2.6% / 0.1% | 0.5% / 0.0% |
| calm | 3M | 12% | -2.9 | 16 | 7.5% / 4.2% | 2.2% / 0.0% |
| stress | 1M | 78% | -1.9 | 5.9 | 29% / 36% | 18% / 19% |
| stress | 3M | 63% | -2.1 | 6.1 | 32% / 43% | 24% / 30% |

On the calm day the skew is the whole crash premium: the lognormal gives a 20% fall no chance, the
market gives it 2.2% over three months. On the stress day the ATM vol alone already prices the crash,
and the market density is more peaked than the lognormal with a fatter far-left tail but a lower
probability of moderate falls. The stress-day tails rest on the SVI wings, where the fit is weakest,
so I would not quote the 24% to more than one significant figure.

## Step 4: local vol and Monte Carlo

Dupire (1994) showed that a surface of vanilla prices pins down a unique local volatility
sigma_loc(index level, time). In total variance the formula is sigma_loc^2(k, T) = (dw/dT) / g(k, T):
the calendar slope over the butterfly term from Step 2. It is evaluated on the surface grid and
clipped to [5%, 300%]. Monte Carlo is log-Euler on k_t = ln(S_t / F_t), so no rates or dividends
appear; 200,000 antithetic paths, 8 steps per day. At daily steps the discretisation bias on the
stress day is 0.2 to 0.5 vol points, several standard errors.

Validation comes first: European calls priced by simulation under local vol must reproduce the
surface they were built from. They do, within 2 standard errors at every strike on both days.

![local vol](figures/step4_localvol.png)
![validation](figures/step4_validation.png)

Two path-dependent options, three months, struck at the forward. A down-and-out call is a call
that is cancelled if the index ever closes below a barrier, here 90% of spot. An arithmetic Asian
call pays on the average daily close rather than the final level. Black-Scholes and local vol agree
on every vanilla, so any difference here is the path. "BS with term structure" uses the ATM vol
curve with no skew, to separate the two effects. Control variates: the European call for the
barrier, the geometric Asian (closed form) for the Asian under Black-Scholes. Prices in % of the
forward; s.e. at most 0.02 on the stress day and 0.006 on the calm day.

| day | option | Black-Scholes | BS with ATM term structure | local vol |
|---|---|---|---|---|
| calm | European | 2.39 | 2.39 | 2.39 |
| calm | down-and-out call | 2.387 | 2.387 | 2.358 |
| calm | arithmetic Asian | 1.385 | 1.285 | 1.287 |
| stress | European | 12.72 | 12.72 | 12.71 |
| stress | down-and-out call | 8.69 | 9.06 | 8.64 |
| stress | arithmetic Asian | 7.53 | 9.04 | 9.08 |

- The barrier is the skew effect. Local vol is higher below spot, so more paths knock out: 1.2%
  cheaper than Black-Scholes on the calm day, 4.6% cheaper than the term-structure model on the stress day.
- The Asian is the term-structure effect. Calm-day vol rises with maturity, so the average fixes
  under lower early vol (7% cheaper); stress-day vol is inverted (20% dearer). Black-Scholes with the
  right ATM term structure comes within 0.5% of the local vol price.

![convergence](figures/step4_convergence.png)

Per random draw, antithetic pairs cut the standard error of the barrier price by about 1.7x. The
European control variate cuts it a further 5x on the calm day but only 1.3x on the stress day,
where most paths knock out and the vanilla payoff is a weak control.

## Layout

| Path | Contents |
|---|---|
| `src/spxvol/black.py` | Black-76 pricer, vega, Brent implied vol |
| `src/spxvol/data.py` | one-day WRDS pull with parquet cache |
| `src/spxvol/clean.py` | filters, forwards, own implied vols |
| `src/spxvol/svi.py` | raw SVI fit, butterfly function g(k) |
| `src/spxvol/surface.py` | per-expiry fits, calendar check, surface grid |
| `src/spxvol/density.py` | Breeden-Litzenberger density and moments |
| `src/spxvol/localvol.py` | Dupire local vol on the surface grid |
| `src/spxvol/mc.py` | log-Euler simulation, antithetic pairs, control variates, geometric Asian closed form |
| `scripts/build.py` | rebuilds every figure and table |

## Limitations

- Raw SVI cannot fit concave total variance; stress-day wings beyond the fitted range are extrapolated.
- Calendar consistency is imposed after the fit. SSVI would build it in, at the cost of fit quality.
- Local variance is the T-derivative of per-expiry fits and inherits their noise as horizontal bands
  between weekly expiries. Vanilla prices are unaffected; smoothing the term structure would clean it.
- Log-Euler needs 8 steps per day on the stress day. A higher-order scheme would be cheaper.
- Quotes are end-of-day closes; intraday timing of the index level and option quotes is not aligned.

## What I learned

