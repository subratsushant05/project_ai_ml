"""Schema-based data validation gate.

A hand-rolled, pydantic-backed schema keeps the dependency surface small
while covering the checks that matter for tabular ML: column presence,
dtype kind, value ranges, null-fraction thresholds, and category domains.
The pipeline halts when validation fails.
"""

from __future__ import annotations

import hashlib
import json
import logging
from enum import StrEnum
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from mlops_pipeline.data import EMPLOYMENT_STATUSES, LOAN_TERMS

logger = logging.getLogger(__name__)


class ColumnKind(StrEnum):
    """Coarse dtype families used for schema checks."""

    NUMERIC = "numeric"
    INTEGER = "integer"
    CATEGORICAL = "categorical"


class ColumnSpec(BaseModel):
    """Validation contract for a single column."""

    name: str
    kind: ColumnKind
    min_value: float | None = None
    max_value: float | None = None
    max_null_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    allowed_values: list[str | int] | None = None


class CheckResult(BaseModel):
    """Outcome of one validation check."""

    column: str
    check: str
    passed: bool
    detail: str = ""


class ValidationReport(BaseModel):
    """Aggregated result of running a schema against a DataFrame."""

    passed: bool
    n_rows: int
    schema_hash: str
    checks: list[CheckResult]

    def failures(self) -> list[CheckResult]:
        """Return only the failed checks."""
        return [c for c in self.checks if not c.passed]

    def save(self, path: Path) -> None:
        """Write the report as pretty-printed JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))


class DataSchema(BaseModel):
    """A named collection of column specs, hashable for lineage tracking."""

    name: str
    columns: list[ColumnSpec]

    def schema_hash(self) -> str:
        """Stable SHA-256 over the schema definition (for MLflow lineage)."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def validate_frame(self, df: pd.DataFrame) -> ValidationReport:
        """Run all checks against ``df`` and collect a report.

        Args:
            df: The frame to validate.

        Returns:
            A :class:`ValidationReport`; ``passed`` is True only when every
            check succeeded.
        """
        checks: list[CheckResult] = []
        for spec in self.columns:
            if spec.name not in df.columns:
                checks.append(
                    CheckResult(
                        column=spec.name,
                        check="presence",
                        passed=False,
                        detail="column missing",
                    )
                )
                continue
            series = df[spec.name]
            checks.append(_check_dtype(spec, series))
            checks.append(_check_nulls(spec, series))
            if spec.min_value is not None or spec.max_value is not None:
                checks.append(_check_range(spec, series))
            if spec.allowed_values is not None:
                checks.append(_check_domain(spec, series))

        report = ValidationReport(
            passed=all(c.passed for c in checks),
            n_rows=len(df),
            schema_hash=self.schema_hash(),
            checks=checks,
        )
        for failure in report.failures():
            logger.warning(
                "Validation failure: %s/%s - %s",
                failure.column,
                failure.check,
                failure.detail,
            )
        return report


def _check_dtype(spec: ColumnSpec, series: pd.Series) -> CheckResult:
    """Verify the column's dtype family matches the spec."""
    if spec.kind is ColumnKind.INTEGER:
        ok = pd.api.types.is_integer_dtype(series)
    elif spec.kind is ColumnKind.NUMERIC:
        ok = pd.api.types.is_numeric_dtype(series)
    else:
        ok = (
            pd.api.types.is_string_dtype(series)
            or pd.api.types.is_object_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_integer_dtype(series)
        )
    return CheckResult(
        column=spec.name,
        check="dtype",
        passed=bool(ok),
        detail="" if ok else f"expected {spec.kind.value}, got {series.dtype}",
    )


def _check_nulls(spec: ColumnSpec, series: pd.Series) -> CheckResult:
    """Verify the null fraction does not exceed the allowed threshold."""
    null_frac = float(series.isna().mean())
    ok = null_frac <= spec.max_null_fraction
    return CheckResult(
        column=spec.name,
        check="null_fraction",
        passed=ok,
        detail="" if ok else f"{null_frac:.3f} > allowed {spec.max_null_fraction:.3f}",
    )


def _check_range(spec: ColumnSpec, series: pd.Series) -> CheckResult:
    """Verify non-null values fall inside ``[min_value, max_value]``."""
    values = series.dropna()
    low = spec.min_value if spec.min_value is not None else float("-inf")
    high = spec.max_value if spec.max_value is not None else float("inf")
    n_bad = int(((values < low) | (values > high)).sum())
    return CheckResult(
        column=spec.name,
        check="range",
        passed=n_bad == 0,
        detail="" if n_bad == 0 else f"{n_bad} values outside [{low}, {high}]",
    )


def _check_domain(spec: ColumnSpec, series: pd.Series) -> CheckResult:
    """Verify categorical values belong to the allowed domain."""
    allowed = set(spec.allowed_values or [])
    observed = set(series.dropna().unique().tolist())
    unexpected = observed - allowed
    return CheckResult(
        column=spec.name,
        check="category_domain",
        passed=not unexpected,
        detail="" if not unexpected else f"unexpected values: {sorted(map(str, unexpected))}",
    )


def loan_schema() -> DataSchema:
    """Build the canonical schema for the loan-default dataset."""
    return DataSchema(
        name="loan_default_v1",
        columns=[
            ColumnSpec(name="age", kind=ColumnKind.INTEGER, min_value=18, max_value=100),
            ColumnSpec(name="income", kind=ColumnKind.NUMERIC, min_value=0),
            ColumnSpec(name="loan_amount", kind=ColumnKind.NUMERIC, min_value=0),
            ColumnSpec(
                name="credit_score", kind=ColumnKind.INTEGER, min_value=300, max_value=850
            ),
            ColumnSpec(
                name="debt_to_income", kind=ColumnKind.NUMERIC, min_value=0.0, max_value=1.0
            ),
            ColumnSpec(
                name="num_prior_defaults", kind=ColumnKind.INTEGER, min_value=0, max_value=5
            ),
            ColumnSpec(
                name="employment_status",
                kind=ColumnKind.CATEGORICAL,
                allowed_values=list(EMPLOYMENT_STATUSES),
            ),
            ColumnSpec(
                name="loan_term_months",
                kind=ColumnKind.CATEGORICAL,
                allowed_values=list(LOAN_TERMS),
            ),
            ColumnSpec(name="default", kind=ColumnKind.INTEGER, min_value=0, max_value=1),
        ],
    )
