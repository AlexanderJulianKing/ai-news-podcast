"""Re-export of the production source-hunter primitives.

The single source of truth is ``newscaster.source_hunter_primitives`` (a
self-contained module: stdlib + requests + BeautifulSoup only). This benchmark
module used to be a byte-for-byte copy of it; it now re-exports so the
primitives are maintained in exactly one place. The benchmark's runners and
tests import their names from here unchanged.
"""
from newscaster.source_hunter_primitives import *  # noqa: F401,F403
