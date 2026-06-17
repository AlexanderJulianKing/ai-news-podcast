"""Build per-slot research records and index a day's research into the store."""
import glob
import json

from newscaster.logging import print_and_write
from newscaster.rag.embeddings import embed_texts
from newscaster.rag.store import Chunk, ResearchIndex


def build_research_record(date, slot, topic, arc_slug, articles, followups):
    """Assemble the sidecar dict. Stamps followup chunk_ids (articles already have them)."""
    stamped_followups = []
    for j, fu in enumerate(followups or []):
        item = dict(fu)
        item["chunk_id"] = f"{date}_seg{slot}_fu{j}"
        stamped_followups.append(item)
    return {
        "date": date,
        "slot": slot,
        "topic": topic,
        "arc_slug": arc_slug,
        "articles": list(articles or []),
        "followups": stamped_followups,
    }


def chunks_from_record(record):
    """Return a list of chunk-spec dicts (all Chunk fields except `vector`)."""
    date = record["date"]
    slot = record["slot"]
    arc_slug = record.get("arc_slug")
    specs = []
    for art in record.get("articles", []):
        specs.append({
            "chunk_id": art["chunk_id"], "date": date, "arc_slug": arc_slug, "slot": slot,
            "chunk_type": "article", "outlet": art.get("outlet"),
            "headline": art.get("original_headline"), "url": art.get("url"),
            "text": art.get("summary", ""),
        })
    for fu in record.get("followups", []):
        specs.append({
            "chunk_id": fu["chunk_id"], "date": date, "arc_slug": arc_slug, "slot": slot,
            "chunk_type": "followup", "outlet": None, "headline": None, "url": None,
            "text": f"Q: {fu.get('question', '')}\nA: {fu.get('answer', '')}",
        })
    return [s for s in specs if s["text"].strip()]


def index_day(date, store=None):
    """Embed and upsert every chunk in the day's `_research.json` sidecars. Idempotent."""
    paths = sorted(glob.glob(f"segment_summaries/{date}_segment*_research.json"))
    specs = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                record = json.load(f)
            specs.extend(chunks_from_record(record))
        except (IOError, json.JSONDecodeError, KeyError, TypeError) as e:
            print_and_write(f"index_day: skipping malformed sidecar {path}: {e}")
            continue
    if not specs:
        return 0
    store = store or ResearchIndex()
    # Skip chunks already indexed so reruns are cheap and an interrupted/failed prior
    # index self-heals without re-embedding what's already stored.
    already = store.existing_chunk_ids([s["chunk_id"] for s in specs])
    specs = [s for s in specs if s["chunk_id"] not in already]
    if not specs:
        return 0
    vectors = embed_texts([s["text"] for s in specs], task_type="RETRIEVAL_DOCUMENT")
    if len(vectors) != len(specs):
        raise RuntimeError(
            f"index_day: embed_texts returned {len(vectors)} vectors for {len(specs)} specs"
        )
    chunks = [Chunk(vector=vectors[i], **specs[i]) for i in range(len(specs))]
    return store.upsert(chunks)
