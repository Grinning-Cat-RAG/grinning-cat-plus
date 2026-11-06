"""Stateful property test of the ODP ingestion lifecycle, with fault injection.

Invariants checked after every step, whatever the sequence of outcomes:
  * no temporary file or directory survives a parse (success, failure, timeout, early stop);
  * no LibreOffice process (nor its children) survives a parse;
  * a successful conversion reaches the inner parser as PPTX bytes with the original source;
  * a conversion failure surfaces as RuntimeError and never reaches the inner parser.
"""
from __future__ import annotations

import time

from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule, run_state_machine_as_test
from langchain_core.document_loaders import Blob

from grinning_cat_plus.parsers.odp_parser import OdpParser, PPTX_MIME
from tests.conftest import pid_alive
from tests.test_odp_parser import ODP_MIME, RecordingParser

FAILING_MODES = ["fail", "nooutput", "badbytes", "hang"]


def test_odp_lifecycle_state_machine(fake_soffice, isolated_tmp):
    class OdpLifecycle(RuleBasedStateMachine):
        def __init__(self):
            super().__init__()
            self.grandchildren: list[int] = []

        def _parse(self, mode, data, source, n_docs, fail_after, take):
            fake_soffice.mode(mode)
            if fake_soffice.pid_file.exists():
                fake_soffice.pid_file.unlink()
            inner = RecordingParser(n_docs=n_docs, fail_after=fail_after)
            gen = OdpParser(inner, timeout=0.3).lazy_parse(
                Blob.from_data(data=data, mime_type=ODP_MIME, path=source)
            )
            got, error = [], None
            try:
                for doc in gen:
                    got.append(doc)
                    if take is not None and len(got) >= take:
                        break
            except Exception as exc:  # noqa: BLE001 - classified below
                error = exc
            finally:
                gen.close()
            if fake_soffice.pid_file.exists():
                self.grandchildren.append(int(fake_soffice.pid_file.read_text()))
            return inner, got, error

        @rule(
            data=st.binary(max_size=64),
            source=st.one_of(st.none(), st.text(max_size=20)),
            n_docs=st.integers(0, 3),
            fail_after=st.one_of(st.none(), st.integers(0, 3)),
            take=st.one_of(st.none(), st.integers(1, 3)),
        )
        def parse_converts(self, data, source, n_docs, fail_after, take):
            inner, got, error = self._parse("ok", data, source, n_docs, fail_after, take)
            [received] = inner.blobs
            assert received.mimetype == PPTX_MIME
            assert received.as_bytes() == b"CONVERTED:" + data
            assert received.source == (source or None)
            # Model of the consumer: it stops after `take` documents, the inner parser raises at `fail_after`.
            expected_docs, expect_error = 0, False
            for i in range(n_docs):
                if fail_after == i:
                    expect_error = True
                    break
                expected_docs += 1
                if take is not None and expected_docs >= take:
                    break
            assert len(got) == expected_docs
            assert isinstance(error, ValueError) if expect_error else error is None

        @rule(mode=st.sampled_from(FAILING_MODES), data=st.binary(max_size=16))
        def parse_fails(self, mode, data):
            inner, got, error = self._parse(mode, data, "f.odp", 2, None, None)
            assert isinstance(error, RuntimeError)
            assert inner.blobs == [] and got == []

        @invariant()
        def no_temporary_leftovers(self):
            assert list(isolated_tmp.iterdir()) == []

        @invariant()
        def no_surviving_processes(self):
            deadline = time.monotonic() + 5
            while any(pid_alive(p) for p in self.grandchildren) and time.monotonic() < deadline:
                time.sleep(0.02)
            assert not any(pid_alive(p) for p in self.grandchildren)

    run_state_machine_as_test(
        OdpLifecycle,
        settings=settings(
            max_examples=40,
            stateful_step_count=6,
            deadline=None,
            suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        ),
    )
