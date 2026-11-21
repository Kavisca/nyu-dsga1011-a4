"""
Error analysis for a model's dev predictions (EC Table 5).

Every dev example whose returned records differ from the gold records is put into
exactly one category, checked in this order:
  1. Execution error       - the predicted SQL fails to run (split into truncated vs. other)
  2. Wrong query type      - the SELECT target differs from gold (e.g. flight_id vs. transport_type)
  3. Wrong constant value  - same SQL skeleton as gold, only literals differ (city, airline, date, time)
  4. Wrong constraints     - same SELECT target but different tables/conditions (missing or extra)

Usage (from part2/):
  python error_analysis.py --sql results/t5_scr_ec_lr3_dev.sql --records records/t5_scr_ec_lr3_dev.pkl
"""
import argparse, pickle, re
from collections import defaultdict

import numpy as np
from load_data import load_lines


def norm(q):
    return re.sub(r"\s+", "", q.lower())


def skeleton(q):
    q = q.lower()
    q = re.sub(r"'[^']*'", "'v'", q)          # string literals
    q = re.sub(r"\b\d+\b", "n", q)            # numbers (dates, times, years)
    return re.sub(r"\s+", "", q)


def select_target(q):
    m = re.search(r"select\s+distinct\s+(.+?)\s+from\s", q.lower())
    return re.sub(r"\s+", "", m.group(1)) if m else None


def cities(q):
    return dict(re.findall(r"(city_\d)\.city_name\s*=\s*'([^']*)'", q, flags=re.I))


def looks_truncated(q):
    q = q.strip().lower()
    return q.count("(") != q.count(")") or bool(re.search(r"(\band|\bor|=|,|\.|_|')$", q)) \
        or q.count("'") % 2 == 1


def f1(gold, pred):
    g, p = set(gold), set(pred)
    prec = 1 if not p else len(p & g) / len(p)
    rec = 1 if not g else len(p & g) / len(g)
    return 2 * prec * rec / (prec + rec + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sql", default="results/t5_scr_ec_lr3_dev.sql")
    ap.add_argument("--records", default="records/t5_scr_ec_lr3_dev.pkl")
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()

    nl = load_lines("data/dev.nl")
    gold_q = load_lines("data/dev.sql")
    pred_q = load_lines(args.sql)
    with open("records/ground_truth_dev.pkl", "rb") as f:
        gold_r, _ = pickle.load(f)
    with open(args.records, "rb") as f:
        pred_r, pred_err = pickle.load(f)
    n = len(gold_q)
    assert len(pred_q) == n == len(pred_r), "prediction file length does not match dev set"

    f1s = [f1(g, p) for g, p in zip(gold_r, pred_r)]
    print(f"Record F1 = {100*np.mean(f1s):.2f}  Record EM = {100*np.mean([set(g)==set(p) for g,p in zip(gold_r,pred_r)]):.2f}  "
          f"SQL EM = {100*np.mean([norm(g)==norm(p) for g,p in zip(gold_q,pred_q)]):.2f}  "
          f"SQL errors = {sum(bool(e) for e in pred_err)}/{n}\n")

    cats = defaultdict(list)
    dup_city = []
    for i in range(n):
        if set(gold_r[i]) == set(pred_r[i]):
            continue
        p, g = pred_q[i], gold_q[i]
        if pred_err[i]:
            cats["1a. Execution error: truncated / incomplete SQL" if looks_truncated(p)
                 else "1b. Execution error: malformed SQL (bad alias/column/table)"].append(i)
        elif select_target(p) != select_target(g):
            cats["2. Wrong query type (SELECT target differs)"].append(i)
        elif skeleton(p) == skeleton(g):
            cats["3. Wrong constant value (city/airline/date/time)"].append(i)
            pc, gc = cities(p), cities(g)
            if pc.get("city_1") and pc.get("city_1") == pc.get("city_2") and gc.get("city_1") != gc.get("city_2"):
                dup_city.append(i)
        else:
            sub = "missing condition(s)" if len(p) < len(g) else "extra/different condition(s)"
            cats[f"4. Wrong constraints: {sub}"].append(i)

    wrong = sum(len(v) for v in cats.values())
    print(f"Incorrect (record set != gold): {wrong}/{n} ({100*wrong/n:.1f}%)\n")
    for name in sorted(cats):
        idx = cats[name]
        print(f"=== {name}: {len(idx)} ({100*len(idx)/n:.1f}% of dev, {100*len(idx)/max(wrong,1):.1f}% of errors), "
              f"mean F1 within category {100*np.mean([f1s[i] for i in idx]):.1f}")
        for i in idx[:args.examples]:
            print(f"  [dev #{i}] NL:   {nl[i]}")
            print(f"           GOLD: {gold_q[i][:300]}")
            print(f"           PRED: {pred_q[i][:300]}")
            if pred_err[i]:
                print(f"           ERR:  {str(pred_err[i])[:150]}")
        print()
    if dup_city:
        print(f"Within category 3: {len(dup_city)} predictions use the same city for origin and destination "
              f"where gold uses different cities (e.g. dev #{', #'.join(map(str, dup_city[:5]))})")


if __name__ == "__main__":
    main()
