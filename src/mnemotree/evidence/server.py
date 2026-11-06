"""The 1.0 path of :mod:`mnemotree.service.app`, kept importable until 2.0.

Also re-exports the names the 1.0 module offered that now live elsewhere.
"""

from mnemotree.core.models import Message as Message
from mnemotree.core.models import scope as scope
from mnemotree.engine import EvidenceEngine
from mnemotree.service.app import *
from mnemotree.service.presets import AML_TOKEN_BUDGET as AML_TOKEN_BUDGET
from mnemotree.service.presets import aml_options as aml_options
from mnemotree.service.request_log import BOOT as BOOT
from mnemotree.service.request_log import MAX_LOGGED_PATH as MAX_LOGGED_PATH
from mnemotree.service.request_log import append_jsonl as append_jsonl
from mnemotree.service.request_log import auth_scheme as auth_scheme
from mnemotree.service.request_log import emit as emit
from mnemotree.service.request_log import failure as failure
from mnemotree.service.request_log import logger as logger
from mnemotree.service.request_log import returned_role_chars as returned_role_chars
from mnemotree.service.request_log import utc_now as utc_now
from mnemotree.service.schemas import MAX_TOP_K as MAX_TOP_K
from mnemotree.service.schemas import AddMessage as AddMessage
from mnemotree.service.schemas import AddRequest as AddRequest
from mnemotree.service.schemas import Request as Request
from mnemotree.service.schemas import SearchRequest as SearchRequest

# The engine's 1.0 name.
Memory = EvidenceEngine
