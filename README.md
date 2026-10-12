# SPX implied vol surface, risk-neutral density and local vol

SPX options from OptionMetrics (WRDS) on a calm day, 19 Feb 2020, and a stress day, 16 Mar 2020.

1. Clean quotes, forward and discount factor per expiry from put-call parity, own implied vols.
2. SVI surface with butterfly and calendar arbitrage checks.
3. Risk-neutral density by Breeden-Litzenberger, compared with a lognormal.
4. Dupire local vol and Monte Carlo for a down-and-out call and an arithmetic Asian, against Black-Scholes.

Steps 3 and 4 are in progress.

## Run

```
python -m venv .venv && .venv/bin/pip install -e .
.venv/bin/python scripts/build.py
```

The first run needs WRDS access through `~/.pgpass`; data is cached in `data/`, which is
gitignored because OptionMetrics data cannot be redistributed.

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
  precision implied by its bid-ask. Mean fit error 0.8 vol points on the calm day, 1.8 on the stress day.
- No butterfly violations. Calendar violations between adjacent expiries are small (worst 0.0014
  in total variance on the calm day, 0.019 on the stress day) and removed by a running maximum in T.
- SVI is convex in total variance; the stress-day put wing is concave, which is the whole fit error.

## Layout

| Path | Contents |
|---|---|
| `src/spxvol/black.py` | Black-76 pricer, vega, Brent implied vol |
| `src/spxvol/data.py` | one-day WRDS pull with parquet cache |
| `src/spxvol/clean.py` | filters, forwards, own implied vols |
| `src/spxvol/svi.py` | raw SVI fit, butterfly function g(k) |
| `src/spxvol/surface.py` | per-expiry fits, calendar check, surface grid |
| `src/spxvol/density.py` | Breeden-Litzenberger density and moments |
| `scripts/build.py` | rebuilds all figures |

## Limitations

- Raw SVI cannot fit concave total variance, so stress-day wings beyond the fitted range are extrapolated.
- Calendar consistency is enforced after the fit rather than built into it (SSVI would do that at the cost of fit).
- Quotes are end-of-day; AM-settled expiries are handled by shortening T by 6.5 hours.

## What I learned

