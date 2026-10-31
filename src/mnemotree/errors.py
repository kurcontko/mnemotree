"""Exceptions raised by mnemotree.

Every class also derives from ``ValueError``, so callers that already catch
``ValueError`` keep working; new callers can catch the narrower type.
"""

from __future__ import annotations


class MnemotreeError(Exception):
    """Base class of the errors mnemotree raises on purpose."""


class InvalidInput(MnemotreeError, ValueError):
    """A caller-supplied argument is unusable: a blank query, a bad namespace or ID."""


class ConfigurationError(MnemotreeError, ValueError):
    """The configuration or the opened database cannot serve the request.

    Raised for an unsupported schema version, a different embedding or extractor
    signature, a missing provider, or an invalid provider configuration. Retrying
    does not help; reopen with the right configuration or rebuild the database.
    """


class IngestConflict(MnemotreeError, ValueError):
    """A request or source ID was reused for a different occurrence."""
