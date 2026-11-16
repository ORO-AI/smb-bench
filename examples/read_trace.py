"""Read one published run: what the model bought each morning and how the quarter was graded.

    python examples/read_trace.py                                   # claude-opus-5-5 on smb-bench-public-1, run 1
    python examples/read_trace.py gpt-6-astra smb-bench-public-9 2  # model slug, task, repeat

Downloads that run's tool-call trace (2 to 19 MB) and its grade from the public dataset into traces/, laid out as the
dataset lays them out, then prints one line per trading day and the graded result. Standard library only.

A trace holds every tool call and every result as the model saw it. It does not hold the business's profit by day:
profit is graded once, after the last day. Profit by day for every run is at https://oroagents.com/smb-bench/runs
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from fetch_data import get

slug, task, repeat = sys.argv[1:4] or ["claude-opus-5-5", "smb-bench-public-1", "1"]
trial = f"traces/runs/{slug.replace('.', '-')}/{task}__r{repeat}"  # the dataset's folders spell dots as hyphens
files = {}
for name in ("agent/trajectory.json", "verifier/reward.json"):
    out = HERE / trial / name
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(get(f"{trial}/{name}"))
    files[name] = json.loads(out.read_text())


def usd(cents):
    return f"{'-' if cents < 0 else ''}${abs(cents) / 100:,.0f}"


print(f"{slug}, {task}, run {repeat}")
row = "{:>3} {:>10} {:>6} {:>9} {:>17} {:>22}".format
print(row("day", "tool calls", "orders", "ordered", "cash next morning", "stock next morning, kg"))
day, calls, orders, ordered = 0, 0, 0, 0
for step in files["agent/trajectory.json"]["steps"]:
    # A provider can reuse tool-call ids across steps, so a call is joined to its result within its own step.
    results = {r["source_call_id"]: r["content"] for r in (step.get("observation") or {}).get("results", [])}
    for call in step.get("tool_calls") or []:
        result = json.loads(results[call["tool_call_id"]])
        name = call["function_name"]
        calls += name != "advance_day"  # advance_day does not count against the daily limit
        if name == "place_purchase_order" and result.get("status") == "accepted":
            orders += 1
            ordered += result["goods_value_cents"]
        if name == "advance_day" and result.get("ok"):
            status = result.get("status")  # absent after the last day
            cash = usd(status["cash_cents"]) if status else ""
            stock = f"{status['on_hand_grams'] / 1000:,.0f}" if status else ""
            print(row(day, calls, orders, usd(ordered), cash, stock))
            day, calls, orders, ordered = day + 1, 0, 0, 0

grade = files["verifier/reward.json"]
print(f"\nProfit over the quarter:        {usd(grade['business_return_cents'])}")
print(f"The fixed buying rule's profit: {usd(grade['reference_business_return_cents'])}")
print(f"Against the fixed rule:         {usd(grade['gain_cents'])} ({grade['reward']:+.2f} margin points)")
