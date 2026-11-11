"""The 1.0 path of :mod:`mnemotree.evaluation`, kept importable until 2.0.

Also re-exports the names the 1.0 module offered that now live elsewhere.
"""

from mnemotree.core.models import Message as Message
from mnemotree.core.models import Record as Record
from mnemotree.core.models import SearchOptions as SearchOptions
from mnemotree.core.models import digest as digest
from mnemotree.core.models import scope as scope
from mnemotree.engine import EvidenceEngine as Memory  # the engine's 1.0 name
from mnemotree.evaluation import AnswerSpan as AnswerSpan
from mnemotree.evaluation import CorpusRecord as CorpusRecord
from mnemotree.evaluation import QueryRecord as QueryRecord
from mnemotree.evaluation import ablate as ablate
from mnemotree.evaluation import ablate_context as ablate_context
from mnemotree.evaluation import ablate_retrieval as ablate_retrieval
from mnemotree.evaluation import evaluate as evaluate
from mnemotree.evaluation import file_hash as file_hash
from mnemotree.evaluation import ingest_corpus as ingest_corpus
from mnemotree.evaluation import installed_version as installed_version
from mnemotree.evaluation import read_jsonl as read_jsonl
from mnemotree.evaluation import retrieval_variants as retrieval_variants
from mnemotree.evaluation import score_queries as score_queries
