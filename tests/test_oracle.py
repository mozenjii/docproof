"""Differential test: the implementation against independently authored expected results.

The cases in ``tests/oracle/cases`` were written from the field contract alone, by someone who
never saw the implementation. This module only translates between their JSON encoding and the
Python API; it holds no rules of its own. When a case fails, the fix is whichever side misread
the contract, decided by cause. A case is never edited just to make it agree.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from docproof import normalize, parsing, routing, validators
from docproof.errors import ContractError
from docproof.schema import Check, Money

CASES = Path(__file__).parent / "oracle" / "cases"
MODULES = ("normalize", "parsing", "validators", "routing")

type Json = Any


def _money(raw: Json) -> Money | None:
    return None if raw is None else Money(Decimal(raw["amount"]), raw["currency"])


def _date(raw: Json) -> dt.date | None:
    return None if raw is None else dt.date.fromisoformat(raw)


def _encode_parse(result: parsing.ParseResult[Any], kind: str) -> Json:
    if isinstance(result, parsing.Parsed):
        value = result.value
        return {"kind": kind, "value": value.isoformat() if kind == "date" else value}
    if isinstance(result, parsing.Ambiguous):
        return {
            "kind": "ambiguous",
            "readings": [r.isoformat() if kind == "date" else r for r in result.readings],
        }
    return {"kind": "invalid"}


def _v8(quantity: Json) -> Check:
    if quantity is None:
        return validators.v8_positive(None, None)
    value = None if quantity["value"] is None else Decimal(quantity["value"])
    return validators.v8_positive(value, quantity["unit"])


def _route(a: Json) -> Json:
    decision = routing.route(
        required=a["required"],
        proposal=routing.Proposal(a["proposal"]),
        grounded=a["grounded"],
        checks=[
            routing.CheckOutcome(Check(c["result"]), c.get("advisory", False)) for c in a["checks"]
        ],
        confidence=a["confidence"],
        threshold=a["threshold"],
        ambiguous=a["ambiguous"],
        conflicting=a["conflicting"],
        in_vocabulary=a["in_vocabulary"],
    )
    return {"status": decision.status.value, "reasons": [r.value for r in decision.reasons]}


def _money_out(m: Money) -> Json:
    return {"amount": str(m.amount), "currency": m.currency}


ADAPTERS: dict[str, Callable[[Json], Json]] = {
    "normalize_text": lambda a: normalize.normalize_text(a["s"]),
    "normalize_identifier": lambda a: normalize.normalize_identifier(a["s"]),
    "normalize_money": lambda a: _money_out(normalize.normalize_money(a["amount"], a["currency"])),
    "parse_date": lambda a: _encode_parse(parsing.parse_date(a["text"], a["day_first"]), "date"),
    "parse_amount": lambda a: _encode_parse(
        parsing.parse_amount(a["text"], a["decimal_comma"]), "amount"
    ),
    "v1_line_sum": lambda a: (
        validators.v1_line_sum(
            [_money(x) for x in a["line_amounts"]], _money(a["subtotal"]), a["tol"]
        ).value
    ),
    "v2_line_arith": lambda a: (
        validators.v2_line_arith(
            None if a["quantity"] is None else validators.parse_quantity(a["quantity"]),
            _money(a["unit_price"]),
            _money(a["amount"]),
            a["tol"],
        ).value
    ),
    "v3_total_arith": lambda a: (
        validators.v3_total_arith(
            _money(a["subtotal"]), _money(a["tax_total"]), _money(a["total"]), a["tol"]
        ).value
    ),
    "v4_date_order": lambda a: (
        validators.v4_date_order(_date(a["earlier"]), _date(a["later"])).value
    ),
    "v5_currency": lambda a: (
        validators.v5_currency(a["document_currency"], [_money(x) for x in a["amounts"]]).value
    ),
    "v7_format": lambda a: validators.v7_format(a["kind"], a["value"]).value,
    "v8_positive": lambda a: _v8(a["quantity"]).value,
    "route": _route,
}


def _load() -> list[Json]:
    cases: list[Json] = []
    for module in MODULES:
        cases.extend(json.loads((CASES / f"{module}.json").read_text(encoding="utf-8")))
    return cases


ALL_CASES = _load()


def test_every_case_targets_a_known_function_and_ids_are_unique() -> None:
    assert {c["fn"] for c in ALL_CASES} <= set(ADAPTERS)
    ids = [c["id"] for c in ALL_CASES]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("case", ALL_CASES, ids=[c["id"] for c in ALL_CASES])
def test_oracle_case(case: Json) -> None:
    adapter = ADAPTERS[case["fn"]]
    if "expected_error" in case:
        with pytest.raises(ContractError) as info:
            adapter(case["args"])
        assert info.value.code == case["expected_error"], case["why"]
    else:
        assert adapter(case["args"]) == case["expected"], case["why"]
