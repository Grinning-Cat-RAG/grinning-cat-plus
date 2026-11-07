"""Stateful property test of the LibreOffice conversion lifecycle (ODP, PPT, DOC), with fault injection.

Invariants checked after every step, whatever the sequence of outcomes:
  * no temporary file or directory survives a parse (success, failure, timeout, early stop);
  * no LibreOffice process (nor its children) survives a parse;
  * a successful conversion reaches the inner parser as the target format with the original source;
  * a conversion failure surfaces as RuntimeError and never reaches the inner parser.
"""
from __future__ import annotations

import time

from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule, run_state_machine_as_test
from langchain_core.document_loaders import Blob

from grinning_cat_plus.parsers.conversion_parser import DOCX_MIME, PPTX_MIME, LibreOfficeConversionParser
from tests.conftest import pid_alive
from tests.test_conversion_parser import DOC_MIME, ODP_MIME, PPT_MIME, RecordingParser

FAILING_MODES = ["fail", "nooutput", "badbytes", "hang"]
ZIP_MAGIC = b"PK\x03\x04"


def _already_target(conversion, data: bytes) -> bool:
    """Model: DOC / PPT bytes that are a ZIP are already DOCX / PPTX: no LibreOffice involved."""
    return conversion[1] in {".doc", ".ppt"} and data.startswith(ZIP_MAGIC)
# (source mime, source extension, target extension, target mime)
CONVERSIONS = [
    (ODP_MIME, ".odp", "pptx", PPTX_MIME),
    (PPT_MIME, ".ppt", "pptx", PPTX_MIME),
    (DOC_MIME, ".doc", "docx", DOCX_MIME),
]


def test_conversion_lifecycle_state_machine(fake_soffice, isolated_tmp):
    class ConversionLifecycle(RuleBasedStateMachine):
        def __init__(self):
            super().__init__()
            self.grandchildren: list[int] = []

        def _parse(self, conversion, mode, data, source, n_docs, fail_after, take):
            mime, source_ext, target_ext, _ = conversion
            fake_soffice.mode(mode)
            if fake_soffice.pid_file.exists():
                fake_soffice.pid_file.unlink()
            inner = RecordingParser(n_docs=n_docs, fail_after=fail_after)
            # Short timeout only where a timeout is the point: a slow machine must not turn "ok" into a failure.
            timeout = 0.3 if mode == "hang" else 60
            parser = LibreOfficeConversionParser(inner, source_ext=source_ext, target_ext=target_ext, timeout=timeout)
            gen = parser.lazy_parse(Blob.from_data(data=data, mime_type=mime, path=source))
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
            if fake_soffice.pid_file.exists():  # written atomically by the fake: never empty
                self.grandchildren.append(int(fake_soffice.pid_file.read_text()))
            return inner, got, error

        @rule(
            conversion=st.sampled_from(CONVERSIONS),
            data=st.one_of(st.binary(max_size=64), st.binary(max_size=32).map(lambda b: ZIP_MAGIC + b)),
            source=st.one_of(st.none(), st.text(max_size=20)),
            n_docs=st.integers(0, 3),
            fail_after=st.one_of(st.none(), st.integers(0, 3)),
            take=st.one_of(st.none(), st.integers(1, 3)),
        )
        def parse_converts(self, conversion, data, source, n_docs, fail_after, take):
            inner, got, error = self._parse(conversion, "ok", data, source, n_docs, fail_after, take)
            [received] = inner.blobs
            assert received.mimetype == conversion[3]
            assert received.as_bytes() == (data if _already_target(conversion, data) else b"CONVERTED:" + data)
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

        @rule(
            conversion=st.sampled_from(CONVERSIONS),
            mode=st.sampled_from(FAILING_MODES),
            data=st.one_of(st.binary(max_size=16), st.binary(max_size=8).map(lambda b: ZIP_MAGIC + b)),
        )
        def parse_fails(self, conversion, mode, data):
            inner, got, error = self._parse(conversion, mode, data, "f" + conversion[1], 2, None, None)
            if _already_target(conversion, data):  # LibreOffice is never started: its faults cannot matter
                assert error is None and len(got) == 2 and inner.blobs[0].as_bytes() == data
                return
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
        ConversionLifecycle,
        settings=settings(
            max_examples=40,
            stateful_step_count=6,
            deadline=None,
            suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        ),
    )
