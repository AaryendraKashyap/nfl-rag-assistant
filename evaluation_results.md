# Evaluation results

Seasons tested: 2020

## Summary

| Metric | Result |
|---|---|
| Exact-count accuracy | 13/13 |
| Play-type precision, this system | 100% |
| Play-type precision, semantic baseline | 80% |
| Score lookup accuracy | 2/2 |
| Median latency, exact count | 5,496 ms |
| Median latency, play-type precision | 6,980 ms |
| Median latency, score lookup | 1,626 ms |

## Detail

| Area | Question | Expected | Got | Pass |
|---|---|---|---|---|
| Exact count | How many touchdowns did Kansas City have in 2020? | 63 | 63 | yes |
| Exact count | How many field goals did Kansas City have in 2020? | 32 | 32 | yes |
| Exact count | How many interceptions did Kansas City have in 2020? | 10 | 10 | yes |
| Exact count | How many touchdowns did Denver have in 2020? | 35 | 35 | yes |
| Exact count | How many field goals did Denver have in 2020? | 28 | 28 | yes |
| Exact count | How many interceptions did Denver have in 2020? | 21 | 21 | yes |
| Exact count | How many touchdowns did Detroit have in 2020? | 45 | 45 | yes |
| Exact count | How many field goals did Detroit have in 2020? | 21 | 21 | yes |
| Exact count | How many interceptions did Detroit have in 2020? | 10 | 10 | yes |
| Exact count | How many touchdowns did Dallas have in 2020? | 42 | 42 | yes |
| Exact count | How many field goals did Dallas have in 2020? | 34 | 34 | yes |
| Exact count | How many interceptions did Dallas have in 2020? | 11 | 11 | yes |
| Exact count | How many touchdowns did Patrick Mahomes have in 2020? | warning, no count | warning | yes |
| Play-type precision | Show me touchdown plays from 2020 | 100% Touchdown | 100% (baseline 100%) | yes |
| Play-type precision | Show me field goals from 2020 | 100% Field Goal | 100% (baseline 80%) | yes |
| Play-type precision | Show me interceptions in the 2020 season | 100% Interception | 100% (baseline 60%) | yes |
| Score lookup | Who won the 2020 Super Bowl? | TB | TB | yes |
| Score lookup | What are the final scores of week 1 in 2020? | 16 | 16 | yes |
