# SMB Bench

**SMB Bench by ORO AI** (Small-to-Midsize Business Bench) is a benchmark for AI agents. A frontier model runs the buying desk of a simulated produce wholesaler for a 90-day quarter and is scored on the profit the business makes, against a fixed buying rule.

[![SMB Bench leaderboard: profit by day of the quarter for each model](https://oroagents.com/images/benchmarks/social/smb-bench.jpeg)](https://oroagents.com/benchmarks/smb-bench)

**Report and leaderboard: [oroagents.com/benchmarks/smb-bench](https://oroagents.com/benchmarks/smb-bench)**

## What it measures

SMB Bench measures whether an agent learns from the delayed consequences of its own decisions.

- The agent buys for one business function, the wholesaler's buying desk. Everything else runs on fixed rules: staff receive deliveries, sell, price, mark down, discard, and pay the bills.
- Demand is censored: once the shelf is empty, shoppers leave and the sale is never recorded, so the books show what was bought, never what was wanted.
- Suppliers and accounts change their behaviour without notice, and the cost of a bad decision arrives weeks later.
- The score is profit in dollars over the quarter. Every model is read against the same fixed buying rule.

## Where everything is

| | |
|---|---|
| Report and leaderboard | https://oroagents.com/benchmarks/smb-bench |
| Every run on its own page: profit by day, orders and the model's notes | https://oroagents.com/smb-bench/runs |
| Every scored run and its trace (CC BY 4.0) | https://huggingface.co/datasets/oro-agents/smb-bench |
| Grader, to regrade any run (access on request) | https://huggingface.co/datasets/oro-agents/smb-bench-grader |
| Harbor Hub listing | https://hub.harborframework.com/datasets/oro-agents/smb-bench |

This repository is the benchmark's home on GitHub. It holds the scripts that recompute the published numbers and the buyer prompt every model works under. The runs, their traces and the grader are at the links above.

## Reproduce the published numbers

[![Reproduce the published numbers](https://github.com/ORO-AI/smb-bench/actions/workflows/reproduce.yml/badge.svg)](https://github.com/ORO-AI/smb-bench/actions/workflows/reproduce.yml)

Every number and table on the [dataset card](https://huggingface.co/datasets/oro-agents/smb-bench) recomputes from one public file, `data/runs.parquet`, which has one row per run. It needs Python 3.12 or later and takes about a minute, most of it installing the packages.

```bash
git clone https://github.com/ORO-AI/smb-bench.git
cd smb-bench
python3 -m venv .venv                              # or: uv venv --seed
. .venv/bin/activate
python -m pip install -r requirements.txt
python fetch_data.py                               # the runs table and the dataset card, about 150 KB
python -I -B recompute.py                          # every number and table on the dataset card
python -I -B recompute.py --check dataset-card.md  # exits 1 if the card and the data disagree
python -I -B plot.py                               # plots/score_vs_cost.png, profit against inference cost
```

The check prints how many of the card's numbers and tables it compared and ends with `0 mismatches`. The same commands run on every change to this repository and once a week.

To read a single run, `python examples/read_trace.py` downloads one run's trace and prints what the model ordered each morning, the cash and stock the business had the next morning, and the graded profit beside the fixed buying rule's. A trace does not hold profit by day; that is on the [run pages](https://oroagents.com/smb-bench/runs).

### What is here

| File | What it is |
|---|---|
| `recompute.py` | Recomputes the leaderboard, the intervals, the pairwise tests and every other number on the dataset card |
| `plot.py` | Draws the chart of profit against inference cost |
| `requirements.txt` | The Python packages the two scripts need |
| `task/buyer-prompt-v11.json` | The exact buyer prompt every model starts each morning with |
| `fetch_data.py` | Downloads `data/runs.parquet` and the dataset card, and confirms the four files above match the dataset |
| `examples/read_trace.py` | Reads one published run from its trace |

The first four files are byte-for-byte copies from the dataset [`oro-agents/smb-bench`](https://huggingface.co/datasets/oro-agents/smb-bench/tree/e6635ff9514e11383ad0e4240712ad8aae20ce0b) at revision `e6635ff9514e11383ad0e4240712ad8aae20ce0b`, where they are published under CC BY 4.0. The dataset is their source, and `fetch_data.py` fails if a copy here differs from it. The simulator and the grader are not in this repository: ORO hosts the environment, and a run with your own agent goes through it as described below.

## Run it on your own agent

```bash
harbor run -d oro-agents/smb-bench@1.0.0 -e e2b -a <agent> -m <model>
```

Runs need a grant: write to team@oroagents.com and we'll send a token, typically for 10 episodes with 3 running at once, under the [lab access terms](https://lab-api.oroagents.com/v1/terms). Results from your own agent are yours to publish, but they are not leaderboard results.

## Citation

```bibtex
@misc{oro2026smbbench,
  title        = {SMB Bench: measuring AI's ability to learn from delayed consequences},
  author       = {Bansal, Shardul and Schilbe, Seth and Barnes, Jarrod and {ORO AI}},
  year         = {2026},
  howpublished = {Hugging Face dataset, oro-agents/smb-bench},
  note         = {Version smb-bench-v1}
}
```

The task worlds use FreshRetailNet-50K; please also cite it, as described in the [dataset card](https://huggingface.co/datasets/oro-agents/smb-bench#citation).

## ORO AI

ORO AI is an applied AI research lab. ORO builds high-fidelity simulation environments grounded in real data to teach AI the tacit knowledge required to run safe and profitable autonomous businesses. More at [oroagents.com](https://oroagents.com/).
