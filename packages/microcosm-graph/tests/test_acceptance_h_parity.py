"""Charter group H: parity (migration acceptance).

These three properties are the only ones in the charter whose subject is not
the toy country. Each compares a graph node's output against a pinned artifact
produced by the lane that wraps the legacy kernel or migrates the country
spine, so each waits on a fixture this lane cannot manufacture: inventing one
would prove that the suite agrees with itself, which is exactly what parity
must not mean.

Every test names the fixture path it expects and reads it and nothing else, so
the producing lane can drop its artifact in and delete the marker. Until then
the fixture is absent, the test fails, and the ``xfail`` reason says whose
fixture it is waiting for.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

if "_toy" not in sys.modules:
    _SPEC = importlib.util.spec_from_file_location(
        "_toy", Path(__file__).with_name("_toy.py")
    )
    sys.modules["_toy"] = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(sys.modules["_toy"])
toy = sys.modules["_toy"]

#: Where the parity lanes drop their pinned fixtures.
PARITY = Path(__file__).parent / "fixtures" / "parity"

#: H1: one directory per wrapped legacy kernel, each holding ``graph.json``
#: (the node declaration), ``inputs.csv``, ``direct.csv`` (the direct call's
#: output at the pinned seed), and ``pins.json`` (seed, kernel ref, kernel
#: implementation hash, and the dependency versions the pin was taken under).
KERNEL_PARITY = PARITY / "kernels"

#: H2: ``uk_spine.json`` — the 27-stage FRS spine expressed as a graph — plus
#: ``sources/``, the data-only bundle both the graph and the legacy oracle
#: rebuild their transforms from. The root transform's weights differ at the
#: last bit between machines, so both sides recompute the root from the raw
#: tables in the test's own process and nothing is pinned.
UK_SPINE_PARITY = PARITY / "uk_spine"

#: H3: ``us_post_transfer.json`` — the derive/seed/simulate subgraph of the
#: stacked spine — plus ``expected.csv``, its pinned output.
US_POST_TRANSFER_PARITY = PARITY / "us_post_transfer"

#: The wrapped kernels H1 covers, in the order the charter names them.
#: The three wrapped kernels the kernel lane shipped: ``fit.qrf@1`` fits on
#: donors and draws on recipients in one node, so there is no separate draw.
WRAPPED_KERNELS = ("fit.qrf", "calibrate", "simulate")

#: What each wrapper honestly claims about its numbers. The forest stack does
#: not promise cross-platform bit stability, so ``fit.qrf@1`` says so; parity
#: in the locked environment is still asserted byte for byte below.
NUMERIC_CLAIMS = {
    "fit.qrf": "tolerance_bound",
    "calibrate": "bitwise",
    "simulate": "bitwise",
}


def _assert_same_bytes(actual, expected) -> None:
    assert actual.dtype == expected.dtype
    assert actual.to_numpy().tobytes() == expected.to_numpy().tobytes()
    assert np.array_equal(actual.isna().to_numpy(), expected.isna().to_numpy())


def _frame_differences(actual, expected) -> str:
    """Name the cells two frames disagree on; two identities alone say nothing."""
    import pandas as pd

    lines: list[str] = []
    for entity in sorted(set(actual.entities) | set(expected.entities)):
        if entity not in actual.entities or entity not in expected.entities:
            lines.append(f"{entity}: present in only one frame")
            continue
        left, right = actual.table(entity), expected.table(entity)
        if list(left.columns) != list(right.columns):
            symmetric = sorted(set(left.columns) ^ set(right.columns))
            lines.append(f"{entity}: column order or set differs ({symmetric})")
        if len(left) != len(right):
            lines.append(f"{entity}: {len(left)} vs {len(right)} rows")
            continue
        for column in left.columns:
            if column not in right.columns:
                continue
            x, y = left[column], right[column]
            if str(x.dtype) != str(y.dtype):
                lines.append(f"{entity}.{column}: dtype {x.dtype} vs {y.dtype}")
            if x.equals(y):
                continue
            if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
                xv = x.to_numpy(dtype="float64", na_value=np.nan)
                yv = y.to_numpy(dtype="float64", na_value=np.nan)
                unequal = ~np.isclose(xv, yv, rtol=0.0, atol=0.0, equal_nan=True)
                count = int(unequal.sum())
                largest = float(np.nanmax(np.abs(xv - yv)[unequal])) if count else 0.0
                lines.append(
                    f"{entity}.{column}: {count} of {len(x)} cells differ, "
                    f"max |difference| {largest:.3e}"
                )
            else:
                count = int((x.astype("string") != y.astype("string")).sum())
                lines.append(f"{entity}.{column}: {count} of {len(x)} cells differ")
    for entity in sorted(
        set(actual.weighted_entities) | set(expected.weighted_entities)
    ):
        if (
            entity not in actual.weighted_entities
            or entity not in expected.weighted_entities
        ):
            lines.append(f"{entity}: weighted in only one frame")
            continue
        xv = np.asarray(actual.weights_for(entity).values, dtype="float64")
        yv = np.asarray(expected.weights_for(entity).values, dtype="float64")
        if xv.shape != yv.shape:
            lines.append(f"{entity} weights: {xv.shape} vs {yv.shape}")
        elif xv.tobytes() != yv.tobytes():
            unequal = ~np.isclose(xv, yv, rtol=0.0, atol=0.0, equal_nan=True)
            lines.append(
                f"{entity} weights: {int(unequal.sum())} of {len(xv)} differ, "
                f"max |difference| {float(np.nanmax(np.abs(xv - yv))):.3e}"
            )
    return "\n".join(lines) or (
        "no table cell or weight differs; strata, mass log, or metadata differ"
    )


def _require(path: Path, produced_by: str) -> Path:
    """Fail with the fixture's path and its owner, never with a bare error."""
    assert path.exists(), (
        f"missing parity fixture {path}; it is produced by {produced_by}, not "
        "by the acceptance lane — inventing it here would make the test agree "
        "with itself instead of with the legacy kernel."
    )
    return path


def test_h1_kernel_parity(tmp_path: Path) -> None:
    """A wrapped legacy kernel is byte-identical to the direct call.

    Expects ``packages/microcosm-graph/tests/fixtures/parity/kernels/<name>/``
    for each of ``fit.qrf``, ``draw.qrf``, ``calibrate``, and ``simulate``,
    each holding ``graph.json``, ``inputs.csv``, ``direct.csv``, and
    ``pins.json``. The wrappers live in ``microcosm-fit``,
    ``microcosm-calibrate``, and the ``RulesEngine`` adapter; the lane that
    writes them produces these fixtures at the same pinned seed.
    """
    from microcosm.graph import ContentStore, compile_graph, graph_from_json, run_graph
    from tools.graph_parity_fixtures import parity_registry

    _require(KERNEL_PARITY, "the kernel-wrapper lane (#378 step 3)")
    registry = parity_registry()
    for name in WRAPPED_KERNELS:
        case = _require(KERNEL_PARITY / name, "the kernel-wrapper lane")
        pins = json.loads((case / "pins.json").read_text())
        assert set(pins) >= {"seed", "kernel", "implementation_hash", "dependencies"}
        kernel = registry.get(pins["kernel"])
        assert kernel.implementation_hash() == pins["implementation_hash"]
        assert set(pins["dependencies"]) == set(kernel.capabilities.dependencies)

        store = ContentStore(tmp_path / name)
        manifest = run_graph(
            compile_graph(graph_from_json((case / "graph.json").read_text())),
            sources={"fixture": case},
            store=store,
            kernels=registry,
            resume="forbid",
            decisions=(),
        )
        node = manifest.nodes[pins["node"]]
        assert node.receipt["capabilities"]["numeric"] == NUMERIC_CLAIMS[name]
        # A structural node re-keys every carried column as an artifact of its
        # own; the direct call produced only what direct.csv holds, so those
        # are the cells compared. A weight transition is compared through the
        # weight artifact under the ``<entity>.weights`` column.
        direct = _direct_table(case)
        compared = 0
        for cell, key in node.artifacts.items():
            label = f"{cell[0]}.{cell[1]}"
            if label in direct.columns:
                _assert_same_bytes(store.load_column(key), direct[label])
                compared += 1
        if node.weight_key is not None:
            entity = (
                graph_from_json((case / "graph.json").read_text())
                .node(pins["node"])
                .weights.entity
            )
            _assert_same_bytes(
                store.load_column(node.weight_key), direct[f"{entity}.weights"]
            )
            compared += 1
        assert compared, f"{name}: the fixture exposed nothing to compare"


def _direct_table(case: Path):
    import pandas as pd

    return pd.read_csv(case / "direct.csv", float_precision="round_trip")


@pytest.mark.requires_uk
def test_h2_uk_spine_parity(tmp_path: Path) -> None:
    """The UK spine as a graph reproduces ``uk_frame_content_identity``.

    The spine's stages read the UK engine's parameters, so this runs in the
    engine tier (``requires_uk``) and skips in the engine-free fast lane.

    Expects ``packages/microcosm-graph/tests/fixtures/parity/uk_spine/`` with
    ``uk_spine.json`` and ``sources/``; both sides run the 28 transforms from
    those sources in this process, root included, because the root weights
    differ at the last bit between machines. Stage order comes
    from declared ``consumes``: the assertion below is that the compiled
    topological order is derived, so the hand-maintained ``_STAGE_NAMES`` tuple
    in ``tools/build_uk_frs_spine.py`` — the 27 names intersected with a
    28-stage packaged manifest, kept in step by hand — can be deleted.
    """
    _require(UK_SPINE_PARITY, "the UK migration lane (charter H2, María reviews)")

    from microcosm.build.uk_runtime.content_identity import uk_frame_content_identity
    from microcosm.build.uk_runtime.graph import uk_registry, uk_spine_graph
    from microcosm.build.uk_runtime.graph_kernels import fixture_stage_plan_inputs
    from microcosm.graph import ContentStore, compile_graph, graph_from_json, run_graph
    from tools.graph_uk_spine_fixture import legacy_oracle_frame

    # The identity is a byte-exact fingerprint of every cell, and the FRS root
    # transform's weights differ at the last bit between machines, so both
    # sides derive the root from the same raw tables here, in this process:
    # the oracle through the legacy StagePlan, the graph through a CREATE
    # kernel bound to the same root transform class (its production path).
    oracle = legacy_oracle_frame(UK_SPINE_PARITY)
    expected = uk_frame_content_identity(oracle)
    _, implementations = fixture_stage_plan_inputs(UK_SPINE_PARITY / "sources")

    # The graph the UK lane ships is also pinned as JSON beside the fixture, so
    # a silent change to the declaration shows up as a fixture diff.
    graph = uk_spine_graph()
    assert graph_from_json((UK_SPINE_PARITY / "uk_spine.json").read_text()) == graph
    compiled = compile_graph(graph)
    assert len(compiled.order) >= 29, "a CREATE node plus the 28 spine stages"
    assert all(
        set(compiled.predecessors[node_id]) <= set(compiled.order[:index])
        for index, node_id in enumerate(compiled.order)
    )

    manifest = run_graph(
        compiled,
        sources={"frs": UK_SPINE_PARITY / "sources"},
        store=ContentStore(tmp_path / "store"),
        kernels=uk_registry(dict(implementations)),
        resume="forbid",
        decisions=(),
    )
    final_version = compiled.versions[compiled.order[-1]]
    final = manifest.population(final_version)
    actual = uk_frame_content_identity(final)
    if actual != expected:
        # Say which cells disagree, and whether the legacy path itself is
        # process-deterministic on this machine, before failing.
        differences = _frame_differences(final, oracle)
        repeated = uk_frame_content_identity(legacy_oracle_frame(UK_SPINE_PARITY))
        stability = (
            "legacy oracle reproduces its own identity in this process"
            if repeated == expected
            else f"legacy oracle is NOT process-deterministic here: {repeated}"
        )
        print(f"H2 graph {actual} != legacy {expected}\n{stability}\n{differences}")
        pytest.fail(f"graph != legacy oracle; {stability}\n{differences}")

    # Charter F12: stage order is derived from declared inputs, so the
    # hand-maintained tuple in the UK driver is gone.
    driver = (Path(__file__).parents[3] / "tools" / "build_uk_frs_spine.py").read_text()
    assert "_STAGE_NAMES" not in driver


@pytest.mark.requires_us
def test_h3_us_post_transfer_parity(tmp_path: Path) -> None:
    """The derive/seed/simulate subgraph reproduces its pinned fixture output.

    Expects
    ``packages/microcosm-graph/tests/fixtures/parity/us_post_transfer/`` with
    ``us_post_transfer.json`` (the graph, pinned), ``sources/`` (a synthetic
    stacked pool the CREATE node loads), and ``expected.csv`` (the cells the
    current ``prepare_stacked_tail_derivation`` → ``derive`` → ``seed`` →
    ``materialize`` functions produce on those sources, as ``entity.column``
    columns). This is the subgraph the stacked spine runs after the ACS
    transfer, so it is the first real US surface the executor owns. It needs
    the US engine, so it runs in the engine tier.
    """
    _require(US_POST_TRANSFER_PARITY, "the US migration lane (#378 step 3)")

    import pandas as pd

    from microcosm.build.us_runtime.graph import us_post_transfer_graph, us_registry
    from microcosm.graph import ContentStore, compile_graph, graph_from_json, run_graph

    # The graph the US lane ships is pinned as JSON beside the fixture, so a
    # silent change to the declaration shows up as a fixture diff.
    graph = us_post_transfer_graph()
    assert (
        graph_from_json((US_POST_TRANSFER_PARITY / "us_post_transfer.json").read_text())
        == graph
    )
    compiled = compile_graph(graph)
    manifest = run_graph(
        compiled,
        sources={"stacked": US_POST_TRANSFER_PARITY / "sources"},
        store=ContentStore(tmp_path / "store"),
        kernels=us_registry(),
        resume="forbid",
        decisions=(),
    )
    final = manifest.population(compiled.versions[compiled.order[-1]])
    expected = pd.read_csv(
        US_POST_TRANSFER_PARITY / "expected.csv", float_precision="round_trip"
    )
    assert len(expected.columns) >= 3, (
        "the fixture pins the derived, seeded, and simulated cells"
    )
    for column in expected.columns:
        entity, name = column.split(".", 1)
        actual = final.table(entity)[name]
        assert actual.dtype == expected[column].dtype, column
        assert actual.to_numpy().tobytes() == expected[column].to_numpy().tobytes(), (
            column
        )


def test_the_parity_fixtures_are_declared_but_not_faked() -> None:
    """Green from the first commit: no parity fixture is invented here.

    If a directory ever appears under ``fixtures/parity/`` in a commit that
    also touches this file, that is the acceptance lane manufacturing its own
    evidence. The suite says so out loud instead.
    """
    if not PARITY.exists():
        return
    for case in sorted(PARITY.iterdir()):
        assert case.is_dir()
        assert (case / "PRODUCED_BY.txt").exists(), (
            f"{case} carries no note saying which lane produced it"
        )
