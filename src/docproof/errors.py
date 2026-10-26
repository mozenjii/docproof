"""Errors that carry a stable machine-readable code from the field contract.

Callers branch on ``code``, never on message text, so messages can improve without breaking them.
"""

from __future__ import annotations


class ContractError(ValueError):
    """An input the field contract defines as invalid.

    Attributes:
        code: Stable error code, for example ``"unknown_currency"``.
    """

    def __init__(self, code: str, detail: str) -> None:
        """Create the error.

        Args:
            code: Stable error code from the contract.
            detail: Human-readable explanation including the offending value.
        """
        super().__init__(f"{code}: {detail}")
        self.code = code
