"""StructuredOutputSimilarity scores structured output field by field, via stickler.

The tests that matter here cover claims the harness itself does not enforce: that the
case score is stickler's weighted score rather than a mean, that the field-level detail
rides on the report and therefore survives serialization, that a case whose output will
not validate still produces a row, and that `test_pass` gates on recall.

Skipped as a module when the `stickler` extra is absent, via `pytestmark`, which keeps
the imports at the top of the file where the house style wants them.
"""

import asyncio
import tempfile
import threading
from importlib.util import find_spec
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from strands_evals import Case, Experiment
from strands_evals.evaluators import Equals, Evaluator, StructuredOutputSimilarity
from strands_evals.local_file_task_result_store import LocalFileTaskResultStore
from strands_evals.types.evaluation import NOT_APPLICABLE, EvaluationData
from strands_evals.types.evaluation_report import EvaluationReport

pytestmark = pytest.mark.skipif(find_spec("stickler") is None, reason="requires the stickler extra")


class LineItem(BaseModel):
    sku: str | None = None
    unit_price: float | None = None


class Invoice(BaseModel):
    invoice_id: str
    vendor_name: str
    total_amount: float | None = None
    line_items: list[LineItem] = []


class Receipt(BaseModel):
    merchant: str
    tax: float


def _invoice(iid="INV-1", vendor="Acme Corporation", total=100.0, sku="SKU-1", price=100.0):
    return Invoice(
        invoice_id=iid,
        vendor_name=vendor,
        total_amount=total,
        line_items=[LineItem(sku=sku, unit_price=price)],
    )


def _case(name, expected):
    return Case(name=name, input="", expected_output=expected, metadata={"name": name})


def _run(evaluator, pairs, **kwargs):
    """Run pairs of (expected, actual) through the real harness."""
    cases = [_case(name, exp) for name, (exp, _) in pairs.items()]
    actual = {name: act for name, (_, act) in pairs.items()}
    return asyncio.run(
        Experiment(cases=cases, evaluators=[evaluator]).run_evaluations_async(
            lambda c: actual[c.metadata["name"]], **kwargs
        )
    )


def _data(expected, actual):
    return EvaluationData(input="", invocation_input="", expected_output=expected, actual_output=actual)


def _reload(report):
    """Round-trip a report through JSON, as `strands-evals run --output` does."""
    return EvaluationReport.model_validate_json(report.model_dump_json())


class TestPerCaseOutput:
    """One output per case, carrying stickler's weighted score."""

    def test_one_output_per_case(self):
        evaluator = StructuredOutputSimilarity(Invoice)
        outputs = evaluator.evaluate(_data(_invoice(), _invoice()))

        assert len(outputs) == 1
        assert outputs[0].label == "Invoice"

    def test_score_is_sticklers_weighted_overall(self):
        """Not a mean of per-field scores; the weighted score stickler computed.

        An earlier draft emitted one output per field and installed a custom aggregator
        to recombine them, which required looking weights up by field name because
        EvaluationOutput carries none. Reporting overall_score directly removes that
        whole mechanism.
        """
        evaluator = StructuredOutputSimilarity(Invoice, weight_hints=True)
        gt, pred = _invoice(), _invoice(iid="INV-9", vendor="Acme Corp")

        expected = evaluator._spec.evaluate(gt, pred).overall_score
        outputs = evaluator.evaluate(_data(gt, pred))

        assert outputs[0].score == pytest.approx(expected)

    def test_reason_names_the_weakest_fields(self):
        evaluator = StructuredOutputSimilarity(Invoice)
        outputs = evaluator.evaluate(_data(_invoice(), _invoice(iid="INV-9")))

        assert "invoice_id" in (outputs[0].reason or "")

    def test_no_custom_aggregator_is_installed(self):
        """The framework default is correct for a single output, so leave it."""
        evaluator = StructuredOutputSimilarity(Invoice)

        assert evaluator.aggregator is Evaluator._default_aggregator


class TestDetailRidesOnTheReport:
    """Field detail lives on the row, so it stays with the report.

    Held on the evaluator instead, it was invisible to `strands-evals run --output`,
    blended across reuse of one instance, and ordered by completion rather than by case.
    """

    def test_row_metadata_carries_field_scores(self):
        evaluator = StructuredOutputSimilarity(Invoice)
        outputs = evaluator.evaluate(_data(_invoice(), _invoice(vendor="Acme Corp")))

        metadata = outputs[0].metadata
        assert set(metadata["field_scores"]) == set(Invoice.model_fields)
        for key in ("precision", "recall", "f1", "matched", "comparison"):
            assert key in metadata, f"metadata should carry {key}"

    def test_per_case_reads_the_report(self):
        report = _run(StructuredOutputSimilarity(Invoice), {"doc-a": (_invoice(), _invoice(vendor="Acme Corp"))})

        (entry,) = StructuredOutputSimilarity.per_case(report)
        assert entry["case"] == "doc-a"
        assert entry["model"] == "Invoice"
        assert set(entry["field_scores"]) == set(Invoice.model_fields)

    def test_per_case_is_in_case_order(self):
        report = _run(
            StructuredOutputSimilarity(Invoice),
            {f"doc-{i}": (_invoice(), _invoice()) for i in range(5)},
            max_workers=5,
        )

        assert [e["case"] for e in StructuredOutputSimilarity.per_case(report)] == [f"doc-{i}" for i in range(5)]

    def test_detail_survives_serialization(self):
        """The reason the detail moved. Held on the evaluator, none of this reloads."""
        report = _run(StructuredOutputSimilarity(Invoice), {"a": (_invoice(), _invoice(vendor="Acme Corp"))})
        reloaded = _reload(report)

        assert StructuredOutputSimilarity.per_case(reloaded) == StructuredOutputSimilarity.per_case(report)
        assert (
            StructuredOutputSimilarity.metrics(reloaded).field_metrics
            == StructuredOutputSimilarity.metrics(report).field_metrics
        )

    def test_reusing_one_instance_does_not_blend_runs(self):
        """Two runs of one instance used to share an accumulator and double the count."""
        evaluator = StructuredOutputSimilarity(Invoice)
        first = _run(evaluator, {"a": (_invoice(), _invoice())})
        second = _run(evaluator, {"b": (_invoice(), _invoice())})

        assert StructuredOutputSimilarity.metrics(first).document_count == 1
        assert StructuredOutputSimilarity.metrics(second).document_count == 1
        assert [e["case"] for e in StructuredOutputSimilarity.per_case(second)] == ["b"]


class TestCaseScore:
    def test_case_score_equals_stickler_overall_score(self):
        evaluator = StructuredOutputSimilarity(Invoice, weight_hints=True)
        gt, pred = _invoice(), _invoice(iid="INV-9", vendor="Acme Corp")

        expected = evaluator._spec.evaluate(gt, pred).overall_score
        report = _run(evaluator, {"a": (gt, pred)})

        assert report.scores[0] == pytest.approx(expected)

    def test_perfect_match_says_so(self):
        report = _run(StructuredOutputSimilarity(Invoice), {"a": (_invoice(), _invoice())})

        assert report.scores[0] == pytest.approx(1.0)
        assert report.test_passes[0] is True
        assert report.reasons[0] == "all fields matched"


class TestBeatsEquals:
    """The reason this integration exists."""

    def test_equals_cannot_rank_what_stickler_separates(self):
        pairs = {
            "near": (_invoice(), _invoice(vendor="Acme Corp")),
            "far": (_invoice(), _invoice(iid="X", vendor="Zeta", total=9.0, sku="Z")),
        }
        stickler_report = _run(StructuredOutputSimilarity(Invoice), pairs)
        equals_report = _run(Equals(), pairs)

        assert len(set(equals_report.scores)) == 1  # both wrong, indistinguishable
        assert stickler_report.scores[0] > stickler_report.scores[1]


class TestDatasetRollup:
    def test_metrics_includes_nested_paths(self):
        report = _run(StructuredOutputSimilarity(Invoice), {"a": (_invoice(), _invoice())})

        paths = StructuredOutputSimilarity.metrics(report).field_metrics
        assert "line_items" in paths
        assert "line_items.sku" in paths

    def test_metrics_counts_every_case(self):
        n = 5
        report = _run(StructuredOutputSimilarity(Invoice), {f"c{i}": (_invoice(), _invoice()) for i in range(n)})

        assert StructuredOutputSimilarity.metrics(report).document_count == n

    def test_five_categories_are_populated(self):
        """FN is a missed field, FA an invented one, FD a wrong one."""
        gt = _invoice(total=100.0)
        missing = _invoice(total=None)  # FN on total_amount
        wrong = _invoice(total=55.0)  # FD on total_amount

        report = _run(StructuredOutputSimilarity(Invoice), {"missing": (gt, missing), "wrong": (gt, wrong)})

        total = StructuredOutputSimilarity.metrics(report).field_metrics["total_amount"]
        assert total["fn"] == 1
        assert total["fd"] == 1

    def test_two_schemas_get_separate_rollups(self):
        """A merged rollup would union field paths and misreport denominators.

        With `model_cls` required, this is declared rather than inferred: two evaluators
        with distinct names, filtered apart by `evaluator=`.
        """
        cases = [_case("inv", _invoice()), _case("rec", Receipt(merchant="M", tax=1.0))]
        actual = {"inv": _invoice(), "rec": Receipt(merchant="M", tax=2.0)}
        report = Experiment(
            cases=cases,
            evaluators=[
                StructuredOutputSimilarity(Invoice, name="invoice"),
                StructuredOutputSimilarity(Receipt, name="receipt"),
            ],
        ).run_evaluations(lambda c: actual[c.metadata["name"]])

        invoice = StructuredOutputSimilarity.metrics(report, evaluator="invoice")
        receipt = StructuredOutputSimilarity.metrics(report, evaluator="receipt")

        assert set(receipt.field_metrics) == {"merchant", "tax"}
        assert "merchant" not in invoice.field_metrics


class TestModelClassIsRequired:
    """Inferring the class per case silently scored 0 on any case loaded from JSON.

    `Experiment.from_file()` datasets and a warm `LocalFileTaskResultStore` both hand
    back `expected_output` as a dict, which no longer carries its class.
    """

    def test_model_cls_is_required(self):
        with pytest.raises(TypeError):
            StructuredOutputSimilarity()

    def test_rejects_a_non_model(self):
        with pytest.raises(TypeError, match="Pydantic model class"):
            StructuredOutputSimilarity(dict)

    def test_dicts_score_the_same_as_instances(self):
        """A declared class coerces a dict, so nothing depends on the class surviving."""
        gt, pred = _invoice(), _invoice(vendor="Acme Corp")
        evaluator = StructuredOutputSimilarity(Invoice)

        from_models = evaluator.evaluate(_data(gt, pred))[0].score
        from_dicts = evaluator.evaluate(_data(gt.model_dump(), pred.model_dump()))[0].score

        assert from_dicts == pytest.approx(from_models)

    def test_a_warm_cache_scores_the_same_as_a_cold_one(self):
        """The reviewer's repro. Inferred mode scored run 1 at 1.0 and run 2 at 0.0:
        `LocalFileTaskResultStore` returns `actual_output` as a dict, which carried no
        class to infer, so every cached case failed."""
        gt, pred = _invoice(), _invoice(vendor="Acme Corp")

        with tempfile.TemporaryDirectory() as directory:
            store = LocalFileTaskResultStore(Path(directory))
            cases = [_case("a", gt)]

            def run():
                return asyncio.run(
                    Experiment(
                        cases=cases,
                        evaluators=[StructuredOutputSimilarity(Invoice)],
                    ).run_evaluations_async(lambda c: pred, evaluation_data_store=store)
                )

            cold, warm = run(), run()

        assert warm.scores == pytest.approx(cold.scores)
        assert "could not compare" not in (warm.reasons[0] or "")

    def test_an_unsupported_model_fails_at_construction(self):
        """A self-referencing model used to be accepted, then fail every case."""

        class Node(BaseModel):
            name: str
            child: "Node | None" = None

        Node.model_rebuild()

        with pytest.raises(Exception, match="recursive"):
            StructuredOutputSimilarity(Node)


class TestSerialization:
    """`Experiment.to_file()` used to raise on an experiment carrying this evaluator."""

    def test_to_dict_emits_a_dotted_path(self):
        evaluator = StructuredOutputSimilarity(Invoice)

        assert evaluator.to_dict() == {
            "evaluator_type": "StructuredOutputSimilarity",
            "model_cls": f"{Invoice.__module__}.Invoice",
        }

    def test_round_trips_through_file(self, tmp_path):
        path = tmp_path / "experiment.json"
        evaluator = StructuredOutputSimilarity(Invoice, match_threshold=0.8)
        Experiment(cases=[_case("a", _invoice())], evaluators=[evaluator]).to_file(path)

        # No custom_evaluators=: the registry resolves it.
        reloaded = Experiment.from_file(path)
        (restored,) = reloaded._evaluators

        assert isinstance(restored, StructuredOutputSimilarity)
        assert restored.model_cls is Invoice
        assert restored.match_threshold == 0.8

    def test_a_reloaded_experiment_scores_the_same(self, tmp_path):
        path = tmp_path / "experiment.json"
        Experiment(
            cases=[_case("a", _invoice())],
            evaluators=[StructuredOutputSimilarity(Invoice)],
        ).to_file(path)

        report = Experiment.from_file(path).run_evaluations(lambda c: _invoice(vendor="Acme Corp"))

        assert report.scores[0] == pytest.approx(
            StructuredOutputSimilarity(Invoice).evaluate(_data(_invoice(), _invoice(vendor="Acme Corp")))[0].score
        )


class TestCasesWithNothingToScore:
    """A case the evaluator cannot score returns a row, rather than raising."""

    def test_missing_expected_output_is_not_applicable(self):
        report = _run(StructuredOutputSimilarity(Invoice), {"a": (None, _invoice())})

        (row,) = report.detailed_results[0]
        assert row.label == NOT_APPLICABLE
        assert row.not_applicable is True
        assert report.test_passes[0] is True, "declining to judge passes, so the row is droppable"

    def test_an_unparseable_output_is_a_scored_failure(self):
        """Prose instead of structured output is an eval failure, not a crash."""
        report = _run(StructuredOutputSimilarity(Invoice), {"a": (_invoice(), "sorry, I could not read it")})

        (row,) = report.detailed_results[0]
        assert row.score == 0.0
        assert row.test_pass is False
        assert "could not compare as Invoice" in (row.reason or "")
        assert row.metadata == {"error": "validation_failed"}
        assert "Evaluator error" not in (row.reason or ""), "the harness never saw an exception"

    def test_a_failed_case_is_excluded_from_the_rollup_but_visible(self):
        """The counts can differ, but the report says which case was dropped."""
        report = _run(
            StructuredOutputSimilarity(Invoice),
            {"good": (_invoice(), _invoice()), "bad": (_invoice(), "prose")},
        )

        assert len(report.scores) == 2
        assert StructuredOutputSimilarity.metrics(report).document_count == 1
        assert [e["case"] for e in StructuredOutputSimilarity.per_case(report)] == ["good"]
        assert any(row.metadata == {"error": "validation_failed"} for rows in report.detailed_results for row in rows)

    def test_a_foreign_shape_does_not_score(self):
        """Strict mode must not coerce nonsense into a score."""
        report = _run(
            StructuredOutputSimilarity(Invoice),
            {"a": (Receipt(merchant="M", tax=1.0), Receipt(merchant="M", tax=1.0))},
        )

        assert report.scores[0] == 0.0
        assert report.detailed_results[0][0].metadata == {"error": "validation_failed"}


class TestConcurrency:
    def test_evaluate_runs_on_worker_threads(self):
        """`evaluate_async` is deliberately not overridden.

        The base class's is `await asyncio.to_thread(self.evaluate, ...)`. A synchronous
        override would run the comparison inline on the event loop, blocking every other
        case's task on a CPU-bound Hungarian match.
        """
        seen: set[str] = set()

        class Probe(StructuredOutputSimilarity):
            def evaluate(self, evaluation_case):
                seen.add(threading.current_thread().name)
                return super().evaluate(evaluation_case)

        _run(Probe(Invoice), {f"c{i}": (_invoice(), _invoice()) for i in range(40)}, max_workers=10)

        assert seen, "evaluate() never ran"
        assert seen != {"MainThread"}, (
            "evaluate() ran only on the main thread, so the harness's to_thread offload "
            "was bypassed -- is evaluate_async overridden?"
        )

    def test_no_results_are_lost_at_the_harness_default(self):
        n = 200
        report = _run(
            StructuredOutputSimilarity(Invoice),
            {f"c{i}": (_invoice(), _invoice()) for i in range(n)},
            max_workers=10,
        )

        assert len(report.scores) == n
        assert StructuredOutputSimilarity.metrics(report).document_count == n


class TestCoercion:
    @pytest.mark.parametrize(
        "actual",
        [
            {"invoice_id": "INV-1", "vendor_name": "Acme Corporation"},
            '{"invoice_id": "INV-1", "vendor_name": "Acme Corporation"}',
        ],
        ids=["dict", "json-string"],
    )
    def test_accepts_dicts_and_json_strings(self, actual):
        evaluator = StructuredOutputSimilarity(Invoice)
        outputs = evaluator.evaluate(_data(Invoice(invoice_id="INV-1", vendor_name="Acme Corporation"), actual))

        assert outputs[0].score == pytest.approx(1.0)

    def test_an_unusable_type_is_reported_not_raised(self):
        evaluator = StructuredOutputSimilarity(Invoice)
        outputs = evaluator.evaluate(_data(_invoice(), 42))

        assert outputs[0].score == 0.0
        assert "instance, dict, or JSON string" in (outputs[0].reason or "")

    def test_a_model_class_accepts_its_own_dotted_path(self):
        evaluator = StructuredOutputSimilarity(f"{Invoice.__module__}.Invoice")

        assert evaluator.model_cls is Invoice


class TestExplain:
    """Config derived from the model class, so it stays on the evaluator."""

    def test_covers_nested_paths(self):
        config = StructuredOutputSimilarity(Invoice).explain()

        assert "line_items.sku" in config
        assert config["invoice_id"]["comparator"] == "ExactComparator"

    def test_list_of_models_is_not_reported_as_a_string_comparator(self):
        """A List[StructuredModel] has no single comparator."""
        config = StructuredOutputSimilarity(Invoice).explain()

        assert config["line_items"]["comparator"] != "LevenshteinComparator"

    def test_reads_the_same_before_and_after_a_run(self):
        evaluator = StructuredOutputSimilarity(Invoice)
        before = evaluator.explain()
        _run(evaluator, {"a": (_invoice(), _invoice(vendor="Acme Corp"))})

        assert evaluator.explain() == before


class SparseInvoice(BaseModel):
    """The common extraction shape: two fields that matter, a long optional tail.

    `overall_score` credits a field absent on both sides with 1.0 at full weight, so the
    tail dominates the mean. These fields exist to make that measurable.
    """

    invoice_id: str
    vendor_name: str
    invoice_date: str | None = None
    total_amount: float | None = None
    tax_amount: float | None = None
    po_number: str | None = None
    payment_terms: str | None = None
    shipping_address: str | None = None
    billing_address: str | None = None
    notes: str | None = None


class TestPassRequiresFindingSomething:
    """`test_pass` gates on recall as well as the score.

    Every test here fails against a `test_pass = result.matched` implementation, and
    against stickler 0.7.0 where `matched` meant all-fields-AND. They are what makes the
    `stickler-eval>=1.0.0` floor in pyproject.toml mean something.
    """

    def test_a_prediction_that_found_nothing_does_not_pass(self):
        """The defect this gate exists for.

        Ground truth has 2 of 10 fields; the agent returns nothing. The 8 fields blank on
        both sides each score 1.0, so the mean is 0.80 and clears the 0.7 default while
        recall is 0.00. Passing here would tell a CI gate that an agent which extracted
        nothing is fine.
        """
        gt = SparseInvoice(invoice_id="INV-8842", vendor_name="Acme Corporation")
        blank = SparseInvoice(invoice_id="", vendor_name="")

        outputs = StructuredOutputSimilarity(SparseInvoice).evaluate(_data(gt, blank))

        assert outputs[0].score > 0.7, "the score really does clear the threshold"
        assert outputs[0].test_pass is False, "but nothing was found, so it must not pass"

        metadata = outputs[0].metadata
        assert metadata["recall"] == 0.0
        assert metadata["matched"] is True, "stickler's score-only verdict still says match"

    def test_widening_the_optional_tail_does_not_buy_a_pass(self):
        """Severity scales with schema width, so a higher threshold cannot fix it."""
        gt = SparseInvoice(invoice_id="INV-1", vendor_name="Acme")
        blank = SparseInvoice(invoice_id="", vendor_name="")

        for threshold in (0.7, 0.75, 0.8):
            outputs = StructuredOutputSimilarity(SparseInvoice, match_threshold=threshold).evaluate(_data(gt, blank))
            assert outputs[0].test_pass is False, f"blank passed at threshold {threshold}"

    def test_a_good_extraction_still_passes(self):
        """The gate must not cost a correct sparse extraction its pass."""
        gt = SparseInvoice(invoice_id="INV-8842", vendor_name="Acme Corporation")

        outputs = StructuredOutputSimilarity(SparseInvoice).evaluate(_data(gt, gt.model_copy()))

        assert outputs[0].test_pass is True
        assert outputs[0].metadata["recall"] == 1.0

    def test_nothing_to_find_is_a_pass_not_a_failure(self):
        """A document whose ground truth is legitimately blank.

        stickler reports recall as 0.0 rather than None when the denominator is empty, so
        a naive recall gate would fail this. The guard reads `tp + fn == 0` instead.
        """

        class AllOptional(BaseModel):
            a: str | None = None
            b: str | None = None

        outputs = StructuredOutputSimilarity(AllOptional).evaluate(_data(AllOptional(), AllOptional()))

        assert outputs[0].score == 1.0
        assert outputs[0].test_pass is True

    def test_per_case_exposes_the_metrics_the_verdict_rests_on(self):
        gt = SparseInvoice(invoice_id="INV-1", vendor_name="Acme")
        report = _run(StructuredOutputSimilarity(SparseInvoice), {"a": (gt, gt.model_copy())})

        (entry,) = StructuredOutputSimilarity.per_case(report)
        for key in ("test_pass", "matched", "precision", "recall", "f1"):
            assert key in entry, f"per_case() should expose {key}"


class TestWithoutTheExtra:
    def test_the_import_error_names_the_extra(self, monkeypatch):
        """Constructing without stickler installed must say how to fix it."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "stickler":
                raise ImportError("No module named 'stickler'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(ImportError, match=r"strands-agents-evals\[stickler\]"):
            StructuredOutputSimilarity(Invoice)


def test_validation_error_is_still_raised_by_pydantic_itself():
    """Sanity check on the type the evaluator catches."""
    with pytest.raises(ValidationError):
        Invoice.model_validate({"not": "an invoice"})
