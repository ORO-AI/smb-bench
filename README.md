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

This repository is the benchmark's home on GitHub. It holds no task data and no grader; those are at the links above.

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
