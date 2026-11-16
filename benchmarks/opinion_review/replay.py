"""Replay the opinion review loop on an aired Sunday piece. Usage: replay.py KEY"""
import json, os, sys, time
R = os.environ.get("REVIEW_REPLAY_DIR", os.path.dirname(os.path.abspath(__file__)))   # needs data/ from the Pi (README)
key = sys.argv[1]
os.environ["NEWSCASTER_LOG_DIR"] = f"{R}/logs/{key}"
sys.path.insert(0, "/Users/alexanderking/Desktop/random_stuff/newscaster3.5")
from newscaster import config, costs
config.init()
config.OPINION_REVIEW_ENABLED = True
out = f"{R}/out/{key}"; os.makedirs(out, exist_ok=True); os.chdir(out)
from newscaster import opinion
state = json.load(open(f"{R}/data/{key}_opinion_state.json"))
aired = open(f"{R}/data/{key}_segment_0.txt").read()
before = "\n\n".join(l[len("Opinion: "):] for l in aired.split("\n") if l.startswith("Opinion: "))
followups = list(state["followups"]); n0 = len(followups)
date_str = time.strftime("%B %-d, %Y", time.strptime(key, "%Y_%m_%d"))
t = time.time()
after, audit = opinion.reviewed(before, followups, state["claim"], date_str)
audit["seconds"] = round(time.time() - t)
audit["cost_usd"] = round(sum(v["usd"] for v in costs._ITEMS.values()), 2)
audit["cost_items"] = costs._ITEMS
audit["lookups_added"] = followups[n0:]
open("before.txt", "w").write(before); open("after.txt", "w").write(after)
json.dump(audit, open("audit.json", "w"), indent=1, ensure_ascii=False, default=str)
print(key, "rounds", len(audit.get("rounds", [])), "stopped:", audit.get("stopped"), "words", len(before.split()), "->",
      len(after.split()), f"${audit['cost_usd']}", f"{audit['seconds']}s")
