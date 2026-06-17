"""Query-side retrieval: embed a draft and pull similar prior-coverage chunks."""
import newscaster.config as _config
from newscaster.rag.embeddings import embed_texts
from newscaster.rag.store import ResearchIndex


def retrieve_prior_research(draft_text, exclude_date, store=None):
    """Return Hits for prior research most similar to `draft_text`.

    Truncates the query defensively; excludes `exclude_date` (today) so we never
    retrieve the current run's own chunks. Returns [] when the store is empty or
    no chunk clears RAG_MIN_SIM.
    """
    query = (draft_text or "").strip()
    if not query:
        return []
    query = query[:24000]  # ~6k tokens, safely under the 8192-token limit (no auto_truncate on the Gemini API)
    vecs = embed_texts([query], task_type="RETRIEVAL_QUERY")
    if not vecs:
        return []
    store = store or ResearchIndex()
    return store.search(vecs[0], exclude_date=exclude_date)
