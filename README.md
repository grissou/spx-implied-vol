# SPX implied vol surface, risk-neutral density and local vol

SPX options from OptionMetrics (WRDS) on a calm day, 19 Feb 2020, and a stress day, 16 Mar 2020.

1. Clean quotes, forward and discount factor per expiry from put-call parity, own implied vols.
2. SVI surface with butterfly and calendar arbitrage checks.
3. Risk-neutral density by Breeden-Litzenberger, compared with a lognormal.
4. Dupire local vol and Monte Carlo for a down-and-out call and an arithmetic Asian, against Black-Scholes.

## Run

```
python -m venv ~/venvs/spxvol && ~/venvs/spxvol/bin/pip install -e .
~/venvs/spxvol/bin/python scripts/build.py
```

The first run needs WRDS access through `~/.pgpass`; data is cached in `data/`, which is
gitignored because OptionMetrics data cannot be redistributed. The full build takes about
30 seconds. Keep the venv outside any cloud-synced folder: OneDrive evicts its files.

## Step 1: forwards and implied vols

![forwards](figures/step1_forwards.png)

- Forward from put-call parity, discount factor from the zero curve. The joint regression recovers
  the same forward to 0.01% but its discount factor is noise below three months.
- On 16 March OptionMetrics' dividend yield reads 0.4%, while parity implies about 2%. Their
  implied vols inherit the error: ours differ by 0.7 vol points with opposite signs on puts and calls.
  Feeding their forward into our solver reproduces their numbers to 0.03 vol points.

## Step 2: SVI surface

![smiles](figures/step2_smiles_2020-03-16.png)
![surface](figures/step2_surface.png)

- Raw SVI per expiry, fitted on strikes 55% to 135% of the forward, each quote weighted by the
  precision implied by its bid-ask. Mean fit error 0.8 vol points on the calm day, 2.2 on the stress day.
- SVI is convex in total variance; the stress-day put wing is concave, which is the whole fit error.
  Left alone, the optimiser answers with a straight line and a zero-width kink at the edge of the
  range, which is a point mass in the density. A floor on sigma of 0.25 x sigma_ATM sqrt(T) removes it
  at a cost of 0.3 vol points on the stress day and nothing on the calm day.
- No butterfly violations. Calendar violations between adjacent expiries are small (worst 0.0014
  in total variance on the calm day, 0.015 on the stress day) and removed by a running maximum in T.

## Step 3: risk-neutral density

![density](figures/step3_density.png)

Density of S_T from the SVI slice, q(K) = (1/D) d2C/dK2 on a strike grid from 0.05F to 4F.
Mass integrates to 1.000 and the mean equals the forward on every slice.

| day | T | ATM vol | log-return skew | excess kurtosis | P(S < 0.9F) RN / LN | P(S < 0.8F) RN / LN |
|---|---|---|---|---|---|---|
| calm | 1M | 11% | -3.4 | 23 | 2.6% / 0.1% | 0.5% / 0.0% |
| calm | 3M | 12% | -2.9 | 16 | 7.5% / 4.2% | 2.2% / 0.0% |
| stress | 1M | 78% | -1.9 | 5.9 | 29% / 36% | 18% / 19% |
| stress | 3M | 63% | -2.1 | 6.1 | 32% / 43% | 24% / 30% |

- On the calm day the lognormal at ATM vol gives a 20% fall essentially zero probability; the market
  gives it 0.5% at one month and 2.2% at three. The skew is the whole crash premium.
- On the stress day the ATM vol alone already prices the crash. The market density is more peaked
  and has a fatter far-left tail than the lognormal, but puts a lower probability on moderate falls.
- Stress-day tails lean on the SVI wings, where the fit is weakest (see Step 2).

## Step 4: local vol and Monte Carlo

![local vol](figures/step4_localvol.png)
![validation](figures/step4_validation.png)

Dupire in total variance, sigma_loc^2 = (dw/dT) / g(k, T), on the surface grid, clipped to
[5%, 300%]. Monte Carlo is log-Euler on k_t = ln(S_t / F_t) with 200,000 antithetic paths and
8 steps per day; at daily steps the Euler bias on the stress day is 0.2 to 0.5 vol points, which
is several standard errors. European calls under local vol reproduce the surface within 2 s.e.
at every strike on both days.

Three-month options struck at the forward, barrier at 90% of spot, daily monitoring. Prices in
% of the forward; s.e. at most 0.02 on the stress day and 0.006 on the calm day.

| day | option | Black-Scholes | BS with ATM term structure | local vol |
|---|---|---|---|---|
| calm | European | 2.39 | 2.39 | 2.39 |
| calm | down-and-out call | 2.387 | 2.387 | 2.358 |
| calm | arithmetic Asian | 1.385 | 1.285 | 1.287 |
| stress | European | 12.72 | 12.72 | 12.71 |
| stress | down-and-out call | 8.69 | 9.06 | 8.64 |
| stress | arithmetic Asian | 7.53 | 9.04 | 9.08 |

- All three models agree on the vanilla, so the differences below are path-dependence only.
- The barrier option is the skew effect. Local vol is higher below spot, so more paths knock out:
  1.2% cheaper than Black-Scholes on the calm day, 4.6% cheaper than the term-structure model on the stress day.
- The Asian difference is the term structure, not the skew. Calm-day vol slopes up with maturity, so
  the average fixes under lower early vol (7% cheaper); stress-day vol is inverted (20% dearer).
  A Black-Scholes model with the right ATM term structure lands within 0.5% of local vol.

![convergence](figures/step4_convergence.png)

Antithetic pairs roughly halve the standard error of the barrier price; adding the European call
as a control variate cuts it by a further order of magnitude.

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
| `scripts/build.py` | rebuilds all figures and tables |

## Limitations

- Raw SVI cannot fit concave total variance, so stress-day wings beyond the fitted range are extrapolated.
- Calendar consistency is enforced after the fit rather than built into it (SSVI would do that at the cost of fit).
- Quotes are end-of-day; AM-settled expiries are handled by shortening T by 6.5 hours.
- Local vol is clipped at a 5% floor where the fitted surface is flat in T; this bites in the far wings only.
- Log-Euler needs 8 steps per day on the stress day; a higher-order scheme would be cheaper.

## What I learned

