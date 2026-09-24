"""Field-level comparison of structured output against ground truth.

`Equals` compares structured output with whole-object `==`, scoring 0.0 or 1.0. An
extraction that gets nine of ten fields right is indistinguishable from one that
gets none right, and a reordered list counts as wrong. `StructuredOutputSimilarity`
compares field by field with type-aware comparators, order-independent list matching
and per-field thresholds, so the score reflects how wrong the output is and names
which field to fix.

Deterministic and offline: no LLM judge, no credentials, no per-call cost.

Requires the `stickler` extra:

```bash
pip install "strands-agents-evals[stickler]"
```
"""

import importlib
import logging
from typing import Any, Iterator, Mapping

from pydantic import BaseModel, ValidationError
from strands.agent.agent_result import AgentResult

from ...types.evaluation import NOT_APPLICABLE, EvaluationData, EvaluationOutput, InputT, OutputT
from ...types.evaluation_report import EvaluationReport
from ..evaluator import Evaluator

logger = logging.getLogger(__name__)


def _stickler() -> tuple[Any, Any]:
    """Import stickler on demand, with a directed error when the extra is missing.

    Imported lazily rather than at module scope: stickler costs ~450ms to import and
    `deterministic/__init__` puts this module on the `import strands_evals` path, so an
    eager import charges that to everyone with the extra installed.

    Returns:
        The `eval_for` and `aggregate_from_comparisons` callables.

    Raises:
        ImportError: If the `stickler` extra is not installed.
    """
    try:
        # optional extra: stickler-eval
        from stickler import aggregate_from_comparisons, eval_for
    except ImportError as exc:
        raise ImportError(
            "StructuredOutputSimilarity requires the 'stickler-eval' package. Install "
            'it with: pip install "strands-agents-evals[stickler]"'
        ) from exc
    return eval_for, aggregate_from_comparisons


def _resolve_model_cls(model_cls: type[BaseModel] | str) -> type[BaseModel]:
    """Accept a model class or the dotted path `to_dict` emits for it.

    The path is `module.QualName`, and a nested class makes the qualname itself dotted
    (`myapp.models.Outer.Inner`), so where the module ends cannot be found by splitting at
    the last dot. Each split is tried from the longest importable module prefix down.

    Args:
        model_cls: The Pydantic model class, or `module.QualName` naming it.

    Returns:
        The model class.

    Raises:
        TypeError: If the value is neither a Pydantic model class nor a dotted path
            resolving to one.
    """
    if isinstance(model_cls, str):
        model_cls = _import_dotted_path(model_cls)
    if not (isinstance(model_cls, type) and issubclass(model_cls, BaseModel)):
        raise TypeError(f"model_cls must be a Pydantic model class; got {model_cls!r}")
    return model_cls


def _import_dotted_path(path: str) -> Any:
    """Resolve `module.QualName`, where the qualname may itself be dotted.

    Args:
        path: The dotted path to resolve.

    Returns:
        The named object.

    Raises:
        ImportError: If a candidate module exists but fails to import for its own reasons,
            such as a missing dependency inside it. Reporting that as an unresolvable path
            would hide the real cause.
        TypeError: If no split of the path names an importable module plus a reachable
            attribute chain.
    """
    parts = path.split(".")
    for split in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:split])
        try:
            resolved: Any = importlib.import_module(module_name)
        except ImportError as exc:
            missing = getattr(exc, "name", None)
            if missing is not None and missing != module_name and not module_name.startswith(f"{missing}."):
                # The module itself is present; something it imports is not. That is the
                # caller's real problem, so it must not be reported as a bad path.
                raise
            continue
        for attribute in parts[split:]:
            resolved = getattr(resolved, attribute, None)
            if resolved is None:
                break
        else:
            return resolved
    raise TypeError(
        f"model_cls string must be a dotted path to an importable model class, "
        f"such as 'myapp.models.Invoice'; could not resolve {path!r}"
    )


def _passed(result: Any, match_threshold: float) -> bool:
    """Whether a case passes: the score cleared the bar AND the real values were found.

    `result.matched` alone is not enough. It is `overall_score >= match_threshold`, and
    `overall_score` credits a field absent on *both* sides with 1.0 at full weight -- a
    value the model correctly left blank is a value it got right. On a sparse extraction
    schema, which is the common case, those uninformative fields outvote the informative
    ones, so a prediction that found *nothing* can clear the threshold:

        10 optional fields, 2 populated in ground truth, prediction returns nothing
        -> overall_score 0.80, matched True, recall 0.00

    and it gets worse as the schema widens, so raising `match_threshold` cannot fix it.
    Recall counts only fields that had a value to find, so correct absence neither helps
    nor hurts it. This is what stickler recommends gating on for sparse schemas:
    https://awslabs.github.io/stickler/Getting-Started/thresholds-and-metrics/#sparse-objects

    Two genuinely empty objects are a match, not a failure, and that case needs detecting
    from the matrix rather than from a null recall: stickler reports recall as 0.0, not
    None, when the denominator is empty, so gating on recall alone would fail a document
    whose ground truth was legitimately blank.

    Counting what there was to find needs all three truth-populated categories, not just
    `tp + fn`. stickler classifies a *wrong* value as `fd` and a *missing* one as `fn`, so
    a prediction that got every populated field wrong reads `tp + fn == 0` while the
    ground truth was not blank at all:

        ground truth invoice_id/vendor_name populated, both predicted wrong
        -> {tp: 0, fn: 0, fd: 2, tn: 8}, score 0.80, recall 0.00

    Taking that for "nothing to find" short-circuited the gate and passed it. The count is
    `tp + fn + fd`.

    When there genuinely was nothing to find, an invented value (`fa`) is still a failure:
    nothing corroborates a prediction that fabricated fields for a blank document.

    Known limit: with something to find, this gates on recall alone, so a prediction that
    finds everything and *also* invents extra fields passes. That is deliberate. Ground
    truth is often annotated sparsely, and a precision gate would fail an agent for
    extracting a field the annotation merely omitted. Read `precision` from the row's
    metadata for a stricter check.
    """
    if not bool(result.matched):
        return False
    # `raw["confusion_matrix"]` is stickler's own result shape rather than a documented
    # accessor; pinned safe by the `<2.0.0` bound on the extra.
    overall = (result.raw or {}).get("confusion_matrix", {}).get("overall", {})
    findable = overall.get("tp", 0) + overall.get("fn", 0) + overall.get("fd", 0)
    if findable == 0:
        return overall.get("fa", 0) == 0
    return result.recall is not None and result.recall >= match_threshold


def _rows(report: EvaluationReport, evaluator: str | None) -> Iterator[tuple[Mapping[str, Any], EvaluationOutput]]:
    """Yield `(case, row)` pairs from a report, optionally for one evaluator only.

    Args:
        report: The report to read.
        evaluator: Evaluator name to filter on, matched against `cases[i]["evaluator"]`.
            None reads every row, which is what a single-evaluator report wants.

    Yields:
        One pair per `EvaluationOutput` in the report.
    """
    for index, rows in enumerate(report.detailed_results):
        case: Mapping[str, Any] = report.cases[index] if index < len(report.cases) else {}
        if evaluator is not None and case.get("evaluator") != evaluator:
            continue
        for row in rows:
            yield case, row


class StructuredOutputSimilarity(Evaluator[InputT, OutputT]):
    """Scores structured output against ground truth, field by field.

    Comparison configuration is inferred from the Pydantic model itself -- the same
    model the agent already passes as `structured_output_model` -- so no schema or
    annotation work is required. Every inferred decision is inspectable via `explain`.

    Returns one `EvaluationOutput` per case. The weighted score is on `score`, and the
    field-level detail rides on `metadata`, which keeps it in the report rather than on
    this instance: `metadata["field_scores"]` per document, plus `precision`, `recall`,
    `f1`, and the raw `comparison` that `metrics` aggregates. `metrics` and `per_case`
    are therefore functions of a report, and survive `report.model_dump_json()`.

    Args:
        model_cls: The Pydantic model the agent emits. Also accepts the dotted path
            `to_dict` writes, so a saved experiment reloads.
        match_threshold: Similarity at or above which an object counts as a match.
            Drives the per-element matching of `list[Model]` fields, and one half of
            `test_pass`. A case passes when the weighted score clears this AND recall
            does, because the score alone credits fields absent on both sides and so
            would pass a prediction that found nothing on a sparse schema. See `_passed`.

            One value bounds both gates, deliberately: the score gate catches wrong
            values, the recall gate catches missing ones, and the two loosen or tighten
            together. Raising it therefore also tightens the tolerated omission rate --
            at the 0.7 default an agent may leave up to 30% of the populated fields
            empty and still clear the recall gate. For independent policies (say,
            strict on values but tolerant of omissions), gate on `metadata["recall"]`
            from each row yourself rather than on `test_pass`.
        weight_hints: When True, weight fields by name-based importance heuristics
            (ids and amounts count for more). Off by default so weights stay uniform
            and metrics are not skewed by guessed business criticality.
        name: Optional evaluator name, forwarded to `Evaluator`. If two instances do
            share one `Experiment`, distinct names are what `metrics(evaluator=...)`
            filters on.

    Raises:
        ImportError: If the `stickler` extra is not installed.
        TypeError: If `model_cls` is not a Pydantic model class or a dotted path to one.
    """

    def __init__(
        self,
        model_cls: type[BaseModel] | str,
        *,
        match_threshold: float = 0.7,
        weight_hints: bool = False,
        name: str | None = None,
    ) -> None:
        """Initialize the evaluator. See the class docstring for argument detail."""
        super().__init__(name=name)
        eval_for, _ = _stickler()
        self._model_cls = _resolve_model_cls(model_cls)
        self.match_threshold = match_threshold
        self.weight_hints = weight_hints
        # Built here, not per case, so a model stickler cannot handle (a self-referencing
        # one, for instance) raises at construction instead of failing every case.
        self._spec = eval_for(
            self._model_cls,
            match_threshold=match_threshold,
            weight_hints=weight_hints,
        )

    @property
    def model_cls(self) -> type[BaseModel]:
        """The Pydantic model every case is compared as.

        Stored privately so the base `to_dict()` skips it; `to_dict()` below re-adds it
        as a dotted path, which is what survives a JSON round-trip.
        """
        return self._model_cls

    def to_dict(self) -> dict:
        """Convert the evaluator into a dictionary.

        Returns:
            dict: The evaluator's information, with `model_cls` as a dotted path so the
            result is JSON-serializable and `from_dict` can resolve it back. A model that
            no other process can import by that path is written anyway, with a warning,
            following how `OutputEvaluator.to_dict` handles tools it cannot serialize.
        """
        _dict = super().to_dict()
        path = f"{self._model_cls.__module__}.{self._model_cls.__qualname__}"
        if self._model_cls.__module__ == "__main__" or "<locals>" in self._model_cls.__qualname__:
            # A model defined in the running script or inside a function. The path is
            # written, because the rest of the experiment is still worth saving, but it
            # resolves to nothing -- or worse, to a different same-named class -- when
            # `from_file` runs elsewhere. Warning here beats failing at load time.
            logger.warning(
                "model_cls=<%s> | not importable by dotted path, so from_file() will not reload this "
                "evaluator, move the model to an importable module to make the experiment portable",
                path,
            )
        _dict["model_cls"] = path
        return _dict

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        """Compare one case's actual output against its expected output.

        Args:
            evaluation_case: The case to score.

        Returns:
            A single-element list carrying the weighted score and, on `metadata`, the
            per-field detail. A case with no `expected_output`, or one whose ground truth
            is a different model, returns a `NOT_APPLICABLE` row; output that will not
            validate returns a score-0 row naming why, rather than raising.
        """
        name = self._model_cls.__name__
        if evaluation_case.expected_output is None:
            return [
                EvaluationOutput(
                    score=0.0,
                    test_pass=True,
                    reason="expected_output is None, so there was no ground truth to compare against",
                    label=NOT_APPLICABLE,
                )
            ]

        dropped = self._foreign_fields(evaluation_case.expected_output)
        if dropped:
            # Ground truth from a different schema: this evaluator has nothing to judge.
            # Converting it would drop those fields, because Pydantic ignores unknown keys
            # by default, and on an all-optional target model both sides would come out
            # blank -- which `_passed` reads as a perfect match. That scored 1.0 and passed.
            other = type(evaluation_case.expected_output).__name__
            return [
                EvaluationOutput(
                    score=0.0,
                    test_pass=True,
                    reason=(
                        f"expected_output is {other}, whose fields {', '.join(dropped)} are not "
                        f"on {name}, so there was nothing for this evaluator to judge"
                    ),
                    label=NOT_APPLICABLE,
                )
            ]

        try:
            expected = self._coerce(evaluation_case.expected_output, "expected_output")
            actual = self._coerce(evaluation_case.actual_output, "actual_output")
        except (ValidationError, TypeError) as exc:
            # A prediction that will not validate is a real evaluation failure, not an
            # evaluator crash. Returning a row keeps the report and the rollup in step.
            logger.debug("case=<%s>, model=<%s> | output did not validate", evaluation_case.name, name)
            return [
                EvaluationOutput(
                    score=0.0,
                    test_pass=False,
                    reason=f"could not compare as {name}: {exc}",
                    label=name,
                    metadata={"error": "validation_failed"},
                )
            ]

        result = self._spec.evaluate(expected, actual)
        passed = _passed(result, self.match_threshold)

        logger.debug(
            "case=<%s>, model=<%s>, score=<%.4f>, matched=<%s> | scored structured output",
            evaluation_case.name,
            name,
            result.overall_score,
            result.matched,
        )

        return [
            EvaluationOutput(
                score=result.overall_score,
                # Not `result.matched` alone: that credits absent-on-both fields, so a
                # blank extraction passes on a sparse schema. See `_passed`.
                test_pass=passed,
                reason=self._weakest(result.field_scores),
                label=name,
                metadata={
                    "field_scores": dict(result.field_scores),
                    # Unlike the score, these ignore correct absence, so they stay
                    # meaningful on a sparse schema; `recall` is what `test_pass` gates on.
                    "precision": result.precision,
                    "recall": result.recall,
                    "f1": result.f1,
                    "matched": result.matched,
                    # `prediction_raw` is dropped: only the confidence accumulators read
                    # it, they need `field_comparisons` alongside it, and it is the
                    # bulkiest part of the result. field_metrics is identical without it.
                    "comparison": {k: v for k, v in result.raw.items() if k != "prediction_raw"},
                },
            )
        ]

    @staticmethod
    def metrics(report: EvaluationReport, *, evaluator: str | None = None) -> Any:
        """Per-field metrics across every case in a report.

        A pure function of the report, so it is index-aligned with the cases, includes
        the rows a failed case produced, and works on a report reloaded from JSON.

        Args:
            report: The report to aggregate.
            evaluator: Evaluator name to filter on, for a run carrying more than one
                evaluator. None aggregates every row that has a comparison, and raises if
                those rows came from more than one evaluator rather than merging them.

        Returns:
            stickler's `ProcessEvaluation`, whose `field_metrics` is keyed by dotted path
            (`line_items.sku`) with the five-category counts (tp/tn/fn/fa/fd) plus
            precision, recall, F1 and accuracy.

            A nested path's counts only cover documents whose parent pair scored at or
            above `match_threshold`: below that, threshold gating treats the pair as
            atomic and emits no field breakdown, so a nested field has a smaller
            denominator than the document count. Cases whose output never validated
            carry no comparison, so they are absent here while still appearing in the
            report as score-0 rows.

        Raises:
            ValueError: If the selected rows came from more than one evaluator. Merging
                two schemas unions their field paths, so `document_count` counts the whole
                suite and each schema's fields read as absent across the other's
                documents. Naming one evaluator is the only correct read.
        """
        _, aggregate_from_comparisons = _stickler()
        scored = [
            (case, row.metadata["comparison"])
            for case, row in _rows(report, evaluator)
            if row.metadata and "comparison" in row.metadata
        ]
        names = sorted({str(case.get("evaluator")) for case, _ in scored})
        if len(names) > 1:
            raise ValueError(
                f"report carries comparisons from {len(names)} evaluators ({', '.join(names)}); "
                f"pass evaluator=<name> to aggregate one schema at a time"
            )
        return aggregate_from_comparisons([comparison for _, comparison in scored])

    @staticmethod
    def per_case(report: EvaluationReport, *, evaluator: str | None = None) -> list[Mapping[str, Any]]:
        """Per-document field scores, in case order.

        A flat table, for printing or loading into a dataframe. Reads the same
        `metadata` the report already carries, so it runs no comparison of its own.

        Nested list children are absent from `field_scores` because no per-leaf score is
        emitted for them; use `metrics` for those, which reports their counts and
        precision/recall/F1.

        Args:
            report: The report to read.
            evaluator: Evaluator name to filter on. None reads every row that has field
                scores.

        Returns:
            One entry per scored case with `case`, `model`, `overall_score`, `test_pass`,
            `matched`, `precision`, `recall`, `f1` and `field_scores`. On a sparse schema
            read `recall` or `f1` rather than `overall_score`: the score credits fields
            absent on both sides, those three do not. Cases whose output never validated
            are omitted; they appear in the report as score-0 rows.
        """
        return [
            {
                "case": case.get("name"),
                "model": row.label,
                "overall_score": row.score,
                "test_pass": row.test_pass,
                # `test_pass` is the verdict the harness reported. `matched` is stickler's
                # score-only view; the two differ exactly when correct absence carried a
                # prediction that found nothing. See `_passed`.
                "matched": row.metadata["matched"],
                "precision": row.metadata["precision"],
                "recall": row.metadata["recall"],
                "f1": row.metadata["f1"],
                "field_scores": dict(row.metadata["field_scores"]),
            }
            for case, row in _rows(report, evaluator)
            if row.metadata and "field_scores" in row.metadata
        ]

    def explain(self) -> dict[str, dict[str, Any]]:
        """Per-field comparison config and why it was chosen.

        Configuration rather than results: derived from the model class alone, so it
        reads the same before and after a run. Keyed by dotted path, so nested decisions
        are auditable too.

        Returns:
            One entry per field path, carrying the comparator, threshold, weight and the
            provenance of each choice.
        """
        return self._spec.explain()

    def _foreign_fields(self, value: Any) -> list[str]:
        """Fields a model of another class carries that `model_cls` does not.

        Empty for anything that is not a Pydantic model, and for a model of another class
        whose fields all exist on `model_cls` -- the same class defined twice, as when a
        notebook cell is re-run, or a second class with the same fields. Those convert
        without loss. Class identity is the wrong test: re-running a cell creates a new
        class object, so `isinstance` rejects instances of the old one.
        """
        if not isinstance(value, BaseModel) or isinstance(value, self._model_cls):
            return []
        return sorted(set(type(value).model_fields) - set(self._model_cls.model_fields))

    def _coerce(self, value: Any, which: str) -> BaseModel:
        """Accept a model instance, an `AgentResult`, a dict, or a JSON string."""
        cls = self._model_cls
        if isinstance(value, AgentResult):
            # A task returning the agent's result directly is the natural way to write it.
            if value.structured_output is None:
                raise TypeError(
                    f"{which} is an AgentResult with no structured_output; call the agent with "
                    f"structured_output_model={cls.__name__}"
                )
            value = value.structured_output
        if isinstance(value, cls):
            return value
        if isinstance(value, BaseModel):
            dropped = self._foreign_fields(value)
            if dropped:
                raise TypeError(
                    f"{which} is {type(value).__name__}, whose fields {', '.join(dropped)} are not on {cls.__name__}"
                )
            return cls.model_validate(value.model_dump())
        if isinstance(value, dict):
            return cls.model_validate(value)
        if isinstance(value, str):
            return cls.model_validate_json(value)
        raise TypeError(
            f"{which} must be a {cls.__name__} instance, an AgentResult, a dict, or a JSON string; "
            f"got {type(value).__name__}"
        )

    @staticmethod
    def _weakest(field_scores: Mapping[str, float], limit: int = 4) -> str:
        """Name the weakest fields so a low case score is actionable."""
        imperfect = sorted(
            ((name, score) for name, score in field_scores.items() if score < 1.0),
            key=lambda pair: pair[1],
        )
        if not imperfect:
            return "all fields matched"
        listed = "; ".join(f"{name}={score:.2f}" for name, score in imperfect[:limit])
        if len(imperfect) > limit:
            listed += f"; (+{len(imperfect) - limit} more)"
        return f"weakest fields: {listed}"
