# -*- coding: utf-8 -*-
"""
33_cocolofa_multimodel_run.py
Launcher for the pre-registered multi-model CoCoLoFa study (pre-registration
draft v5: "The Explicit-Implicit Gap in Zero-Shot False-Dilemma Detection
Across Three Language Models").

WHAT IS FIXED (nothing here may change after the pre-registration commit)
  Detector  the frozen v2 prompt, system message and label parser, IMPORTED from
            23_logic_fd_eval.py (not retyped). Each comment is passed whole in the
            "Sentence" slot, exactly as in the first study (29_cocolofa_fd_run.py).
  Sample    cocolofa_fd_sample_manifest.csv; its SHA-256 must equal MANIFEST_SHA256.
  Models    three models of the Gemini API free tier (ARMS below), all with
            thinking set to "minimal" and every sampling parameter left at the
            provider default (no temperature, top_p or top_k is sent).
  Runs      three runs per model. Within a run the 440 comments are processed in
            an order fixed by a seed, and for each comment the three models are
            called in an order that rotates from one comment to the next. Run 1 is
            confirmatory; runs 2 and 3 only measure stability. A fourth run is
            refused.
  Calls     a call that returns any model output is never repeated. Only failures
            that return no model output are retried, at most three times with the
            identical request. Unparseable answers, empty answers, answers cut by
            the token limit and blocked answers are recorded as abstentions, each
            with its own code.

FREE-TIER QUOTA
  The free tier has a daily request limit per model. If it is reached during a
  run, the launcher saves everything done so far and stops. Starting it again on
  a later day continues the SAME run where it stopped, without repeating any call
  that returned output. A run is finalised (and locked) only when all 1,320 calls
  are recorded.

FILES (benchmark/)
  mm_run{r}_progress.csv          one line per finished call, no comment text
  mm_pred_{arm}_run{r}.csv        final predictions per arm and run, no comment text
  mm_usage_log.csv                one line per API attempt, no comment text
  mm_raw_local.jsonl              raw answers, LOCAL ONLY (git-ignored)
  mm_smoke_log.csv                smoke test on the author's own pilot items

USAGE (from the repository root, with GEMINI_API_KEY set)
  python benchmark/33_cocolofa_multimodel_run.py --check      no calls; verifies files
  python benchmark/33_cocolofa_multimodel_run.py --smoke      10 pilot items per model
  python benchmark/33_cocolofa_multimodel_run.py --count-tokens
  python benchmark/33_cocolofa_multimodel_run.py --run 1      (then --run 2, --run 3)

Author: Șerban, G. F. (2026). ORCID 0009-0000-2266-3356. Code: MIT.
"""
import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "cocolofa_fd_sample_manifest.csv"
LOCAL = HERE / "cocolofa_fd_eval_local.csv"
PILOT = HERE / "contrastive_pairs_pilot.csv"
USAGE = HERE / "mm_usage_log.csv"
RAW = HERE / "mm_raw_local.jsonl"
SMOKE = HERE / "mm_smoke_log.csv"
MANIFEST_SHA256 = "e35b540ec55962eb9de92b9887bd56647f563944b31050163c8fad531b2b9975"

# ---------------------------------------------------------------- frozen design
ARMS = {                       # arm -> model identifier sent to the API
    "M1": "gemini-3.1-flash-lite",
    "M2": "gemini-3.5-flash-lite",
    "M3": "gemma-4-31b-it",
}
THINKING_LEVEL = "minimal"     # Flash-Lite: lowest level; Gemma 4: thinking off
MAX_OUTPUT_TOKENS = 1024       # room for any residual thinking plus a one-word label
N_RUNS = 3
ORDER_SEED = 20261118          # run r uses ORDER_SEED + r
TRANSPORT_RETRIES = 3          # retries for failures that return no model output
MIN_SECONDS_PER_MODEL = 13.0   # at most about 4.6 requests per minute per model (limits may be 5 RPM)
TIMEOUT_MS = 120_000

ABSTAIN_CODES = ("ABSTAIN_PARSE", "ABSTAIN_EMPTY", "ABSTAIN_MAXTOKENS", "ABSTAIN_BLOCKED",
                 "ABSTAIN_FAILED")


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


det = load("fd_eval", "23_logic_fd_eval.py")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def config_fingerprint():
    """Hash of everything that defines a call, printed and logged for the record."""
    frozen = {"arms": ARMS, "thinking_level": THINKING_LEVEL, "max_output_tokens": MAX_OUTPUT_TOKENS,
              "system": det.SYSTEM, "prompt": det.PROMPT, "defs": det.DEFS,
              "outputs": det.OUTPUTS, "order_seed": ORDER_SEED, "retries": TRANSPORT_RETRIES}
    return hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------- label mapping
def map_label(raw, finish, blocked):
    """Frozen parser of the first study, plus explicit abstention codes."""
    if blocked:
        return "ABSTAIN_BLOCKED"
    text = (raw or "").strip()
    if not text:
        return "ABSTAIN_MAXTOKENS" if finish == "MAX_TOKENS" else "ABSTAIN_EMPTY"
    t = text.lower().replace(" ", "_").replace("-", "_")
    if not any(lab in t for lab in det.OUTPUTS):
        return "ABSTAIN_PARSE"          # the frozen parser would have said "none"
    return det.parse_label(text)


# ---------------------------------------------------------------- API
class DailyQuota(Exception):
    pass


def make_client():
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise SystemExit("google-genai is not installed: python -m pip install google-genai")
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY is not set")
    client = genai.Client(http_options=types.HttpOptions(
        timeout=TIMEOUT_MS, retry_options=types.HttpRetryOptions(attempts=1)))
    return client, types


def request_config(types):
    return types.GenerateContentConfig(
        system_instruction=det.SYSTEM,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
    )


_last_call = {}


def pace(model):
    wait = MIN_SECONDS_PER_MODEL - (time.monotonic() - _last_call.get(model, -1e9))
    if wait > 0:
        time.sleep(wait)
    _last_call[model] = time.monotonic()


def append_csv(path, row):
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()), lineterminator="\n")
        if new:
            w.writeheader()
        w.writerow(row)


def call_once(client, types, arm, item_id, text, run, log_path=USAGE):
    """One logical call: returns (label, raw). Retries only failures without output."""
    model = ARMS[arm]
    prompt = det.PROMPT.format(defs=det.DEFS, text=text)
    attempt, minute_waits = 0, 0
    while attempt <= TRANSPORT_RETRIES:
        attempt += 1
        pace(model)
        t0 = time.monotonic()
        base = {"comment_id": item_id, "arm": arm, "model_sent": model, "run": run,
                "time_utc": now(), "attempt": attempt}
        try:
            resp = client.models.generate_content(model=model, contents=prompt,
                                                  config=request_config(types))
        except Exception as exc:          # no model output came back
            msg = str(exc)
            code = "429_DAY" if ("429" in msg and "day" in msg.lower()) else \
                   "429" if ("429" in msg or "RESOURCE_EXHAUSTED" in msg) else type(exc).__name__
            append_csv(log_path, {**base, "model_returned": "", "input_tokens": "",
                                  "output_tokens": "", "thought_tokens": "", "finish_reason": "",
                                  "latency_s": round(time.monotonic() - t0, 2),
                                  "outcome": f"ERROR_{code}"})
            if code == "429_DAY":
                raise DailyQuota(f"{model}: daily free quota reached")
            if code == "429" and minute_waits < 5:
                minute_waits += 1         # per-minute limit: not a failure; wait and resend
                attempt -= 1
                time.sleep(65)
                continue
            if attempt <= TRANSPORT_RETRIES:
                time.sleep(10 * attempt)
            continue
        cand = (resp.candidates or [None])[0]
        finish = getattr(getattr(cand, "finish_reason", None), "name", "") if cand else ""
        blocked = bool(getattr(resp, "prompt_feedback", None) and resp.prompt_feedback.block_reason) \
            or finish in ("SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "RECITATION")
        try:
            raw = resp.text or ""
        except Exception:
            raw = ""
        u = resp.usage_metadata
        label = map_label(raw, finish, blocked)
        append_csv(log_path, {**base, "model_returned": resp.model_version or "",
                              "input_tokens": getattr(u, "prompt_token_count", "") if u else "",
                              "output_tokens": getattr(u, "candidates_token_count", "") if u else "",
                              "thought_tokens": getattr(u, "thoughts_token_count", "") if u else "",
                              "finish_reason": finish,
                              "latency_s": round(time.monotonic() - t0, 2), "outcome": label})
        return label, raw
    return "ABSTAIN_FAILED", ""


# ---------------------------------------------------------------- inputs
def load_items():
    if sha256(MANIFEST) != MANIFEST_SHA256:
        raise SystemExit("sample manifest differs from the pre-registered one; stopping")
    if not LOCAL.exists():
        raise SystemExit(f"{LOCAL.name} not found; run 28_cocolofa_sample.py first")
    items = list(csv.DictReader(open(LOCAL, encoding="utf-8")))
    man = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    if [r["id"] for r in items] != [r["id"] for r in man]:
        raise SystemExit("local evaluation file does not match the manifest; stopping")
    return items


def final_path(arm, run):
    return HERE / f"mm_pred_{arm}_run{run}.csv"


def progress_path(run):
    return HERE / f"mm_run{run}_progress.csv"


def schedule(items, run):
    """Fixed comment order for this run and a model order rotating per comment."""
    order = list(range(len(items)))
    random.Random(ORDER_SEED + run).shuffle(order)
    arms = list(ARMS)
    calls = []
    for k, i in enumerate(order):
        rot = arms[k % 3:] + arms[:k % 3]
        calls.extend((i, a) for a in rot)
    return calls


# ---------------------------------------------------------------- commands
def cmd_check():
    print(f"config fingerprint: {config_fingerprint()}")
    print(f"manifest SHA-256 ok: {sha256(MANIFEST) == MANIFEST_SHA256}")
    print(f"local comment file present: {LOCAL.exists()}")
    for r in range(1, N_RUNS + 1):
        done = [a for a in ARMS if final_path(a, r).exists()]
        prog = progress_path(r)
        n = sum(1 for _ in open(prog, encoding="utf-8")) - 1 if prog.exists() else 0
        print(f"run {r}: finalised arms {done or 'none'}; calls recorded {max(n, 0)} of {3 * 440}")


def cmd_smoke():
    client, types = make_client()
    rows = list(csv.DictReader(open(PILOT, encoding="utf-8")))[:10]
    for arm in ARMS:
        labels = []
        for r in rows:
            label, _ = call_once(client, types, arm, f"pilot_{r['pair_id']}_{r['member']}",
                                 r["text"], 0, log_path=SMOKE)
            labels.append(label)
        print(f"{arm} {ARMS[arm]}: {labels}")
    print(f"smoke test logged in {SMOKE.name} (no CoCoLoFa text was sent)")


def cmd_count_tokens():
    client, types = make_client()
    items = load_items()
    for arm, model in ARMS.items():
        total = 0
        for r in items:
            prompt = det.PROMPT.format(defs=det.DEFS, text=r["text"])
            total += client.models.count_tokens(model=model, contents=prompt).total_tokens
        print(f"{arm} {model}: {total} input tokens for 440 requests (system message excluded)")


def cmd_run(run):
    if run < 1 or run > N_RUNS:
        raise SystemExit(f"only runs 1 to {N_RUNS} exist; a further run is refused")
    for r in range(1, run):
        if not all(final_path(a, r).exists() for a in ARMS):
            raise SystemExit(f"run {r} is not finished; finish it before starting run {run}")
    if all(final_path(a, run).exists() for a in ARMS):
        raise SystemExit(f"run {run} is already finalised and locked")
    items = load_items()
    client, types = make_client()
    prog = progress_path(run)
    done = set()
    if prog.exists():
        done = {(r["comment_id"], r["arm"]) for r in csv.DictReader(open(prog, encoding="utf-8"))}
    calls = schedule(items, run)
    todo = [(i, a) for i, a in calls if (items[i]["id"], a) not in done]
    print(f"run {run}: {len(calls) - len(todo)} calls already recorded, {len(todo)} to go "
          f"(config {config_fingerprint()[:12]})")
    try:
        for n, (i, arm) in enumerate(todo, 1):
            r = items[i]
            label, raw = call_once(client, types, arm, r["id"], r["text"], run)
            append_csv(prog, {"comment_id": r["id"], "arm": arm, "run": run, "time_utc": now(),
                              "stage1_pred": label})
            with open(RAW, "a", encoding="utf-8") as f:
                f.write(json.dumps({"comment_id": r["id"], "arm": arm, "run": run,
                                    "raw": raw[:500]}, ensure_ascii=False) + "\n")
            if n % 30 == 0:
                print(f"   {len(calls) - len(todo) + n}/{len(calls)} calls recorded")
    except DailyQuota as exc:
        print(f"\nSTOPPED: {exc}. Everything so far is saved. Start the same command again "
              f"tomorrow after 09:00 (German time) to continue run {run}.")
        return
    finalise(items, run)


def finalise(items, run):
    prog = {(r["comment_id"], r["arm"]): r["stage1_pred"]
            for r in csv.DictReader(open(progress_path(run), encoding="utf-8"))}
    for arm in ARMS:
        out = []
        for r in items:
            pred = prog[(r["id"], arm)]
            out.append({"id": r["id"], "split": r["split"], "cell": r["cell"],
                        "gold_false_dilemma": r["gold_false_dilemma"],
                        "source_label": r["source_label"], "marker": r["marker"],
                        "worker_id": r["worker_id"], "arm": arm, "model": ARMS[arm], "run": run,
                        "stage1_pred": pred, "flag_fd": int(pred == det.TARGET),
                        "abstain": int(pred == det.ABSTAIN or pred in ABSTAIN_CODES)})
        path = final_path(arm, run)
        with open(path, "x", newline="", encoding="utf-8") as f:    # "x": never overwrite
            w = csv.DictWriter(f, fieldnames=list(out[0].keys()), lineterminator="\n")
            w.writeheader()
            w.writerows(out)
        print(f"saved {path.name}  SHA-256 {sha256(path)}")
    print(f"run {run} finalised. Commit these files before starting the next run.")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--count-tokens", action="store_true")
    g.add_argument("--run", type=int)
    a = ap.parse_args()
    if a.check:
        cmd_check()
    elif a.smoke:
        cmd_smoke()
    elif a.count_tokens:
        cmd_count_tokens()
    else:
        cmd_run(a.run)


if __name__ == "__main__":
    sys.exit(main())
