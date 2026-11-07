"""The 1.0 path of :mod:`mnemotree.ingest.atoms`, kept importable until 2.0.

Also re-exports the names the 1.0 module offered that now live elsewhere.
"""

from mnemotree.core.models import Atom as Atom
from mnemotree.core.models import AtomDraft as AtomDraft
from mnemotree.core.models import AtomHit as AtomHit
from mnemotree.core.models import Citation as Citation
from mnemotree.core.models import ExtractorSpec as ExtractorSpec
from mnemotree.core.models import Passage as Passage
from mnemotree.core.models import Record as Record
from mnemotree.core.models import SourceSpan as SourceSpan
from mnemotree.core.protocols import Extractor as Extractor
from mnemotree.ingest.atoms import *
from mnemotree.providers.extraction import EXTRACTION_PROMPT as EXTRACTION_PROMPT
from mnemotree.providers.extraction import ExtractionResponse as ExtractionResponse
from mnemotree.providers.extraction import OpenAIExtractor as OpenAIExtractor
