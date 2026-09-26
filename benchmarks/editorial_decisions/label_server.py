"""Local labeling page for lead-pick cases. Standard library only.

    python3 -m benchmarks.editorial_decisions.label_server
    open http://127.0.0.1:8917

Reads data/lead_pick_cases.jsonl. Appends every save to
data/lead_pick_labels.jsonl; the latest line per case_id is the label of record,
and earlier lines stay as history.
"""
import argparse
import json
import os
import random
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

from benchmarks.editorial_decisions.hypotheticals import load_hypotheticals

HERE = os.path.dirname(os.path.abspath(__file__))
CASES_PATH = os.path.join(HERE, "data", "lead_pick_cases.jsonl")
LABELS_PATH = os.path.join(HERE, "data", "lead_pick_labels.jsonl")
RELABELS_PATH = os.path.join(HERE, "data", "lead_pick_relabels.jsonl")
RELABEL = {"on": False, "only": None}  # set by --relabel
PAGE_PATH = os.path.join(HERE, "label.html")


def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_labels(path=None):
    """Latest label per case_id."""
    latest = {}
    for row in _read_jsonl(path or LABELS_PATH):
        latest[row["case_id"]] = row
    return latest


def display_order(case):
    """Shuffle briefs with a per-case seed so the labeler does not see the
    position Opus saw, and sees the same order on every visit."""
    indexes = [b["index"] for b in case["briefs"]]
    random.Random(case["case_id"]).shuffle(indexes)
    return indexes


def cases_payload(cases_path=None, labels_path=None, hypotheticals=True):
    """Real mornings first, then the invented ones.

    In relabel mode only mornings that already have a label are served, the old
    label is hidden, and saves go to a separate file. Comparing the two files
    gives the labeler's agreement with himself, which is the ceiling any model
    can be scored against.
    """
    first_pass = load_labels(LABELS_PATH) if RELABEL["on"] else None
    labels = load_labels(RELABELS_PATH if RELABEL["on"] else labels_path)
    out = []
    all_cases = _read_jsonl(cases_path or CASES_PATH)
    if hypotheticals:
        all_cases = all_cases + load_hypotheticals()
    if first_pass is not None:
        all_cases = [c for c in all_cases if c["case_id"] in first_pass
                     and (not RELABEL["only"] or first_pass[c["case_id"]].get("confidence") in RELABEL["only"])]
        random.Random("relabel").shuffle(all_cases)  # not in the order they were first seen
    for case in all_cases:
        out.append({
            "case_id": case["case_id"],
            "date": case["date"],
            "synthetic": bool(case.get("synthetic")),
            "axis": case.get("axis"),
            "briefs": case["briefs"],
            "order": display_order(case),
            "coverage_notes": case.get("coverage_notes", ""),
            "preamble": case.get("preamble", ""),
            "production_pick_index": case.get("production_pick_index"),
            "production_answer": case.get("production_answer"),
            "label": labels.get(case["case_id"]),
        })
    return out


REQUIRED = ("case_id", "lead_index", "remembers_outcome", "confidence")


def save_label(payload, labels_path=None):
    missing = [k for k in REQUIRED if k not in payload]
    if missing:
        raise ValueError("missing fields: " + ", ".join(missing))
    payload = dict(payload)
    payload["labeled_at"] = datetime.now().isoformat(timespec="seconds")
    path = labels_path or LABELS_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return payload


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, body, content_type):
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/":
            with open(PAGE_PATH, "r", encoding="utf-8") as f:
                self._send(200, f.read(), "text/html")
        elif self.path == "/api/cases":
            self._send(200, json.dumps(cases_payload()), "application/json")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/label":
            self._send(404, "not found", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            saved = save_label(json.loads(self.rfile.read(length)), RELABELS_PATH if RELABEL["on"] else None)
            self._send(200, json.dumps(saved), "application/json")
        except (ValueError, json.JSONDecodeError) as e:
            self._send(400, json.dumps({"error": str(e)}), "application/json")

    def log_message(self, fmt, *args):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8917)
    ap.add_argument("--relabel", action="store_true", help="blind second pass over already-labeled mornings; saves to lead_pick_relabels.jsonl")
    ap.add_argument("--only-confidence", nargs="+", default=None, help="with --relabel: only mornings first labeled at these confidence levels")
    args = ap.parse_args()
    RELABEL["on"], RELABEL["only"] = args.relabel, args.only_confidence
    n = len(_read_jsonl(CASES_PATH))
    print("{} cases, {} labeled. http://127.0.0.1:{}".format(n, len(load_labels()), args.port))
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
