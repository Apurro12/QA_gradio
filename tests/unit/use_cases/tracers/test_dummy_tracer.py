from rag_system.use_cases.tracers.dummy_tracer import DummyTracer, DummySpan

class TestDummyTracer:
    def test_start_span(self):
        tracer = DummyTracer()
        span = tracer.start_span("test_span")
        assert span.name == "test_span"
        assert isinstance(span, DummySpan)

    def test_get_tracer(self):
        tracer = DummyTracer()
        returned_tracer = tracer.get_tracer("test_tracer")
        assert returned_tracer is tracer

    def test_start_as_current_span(self):
        tracer = DummyTracer()
        with tracer.start_as_current_span("test_current_span") as span:
            assert span.name == "test_current_span"
            assert isinstance(span, DummySpan)

    def test_dummy_span_end_does_nothing(self):
        """Test that DummySpan.end() does nothing and returns None."""
        span = DummySpan("test")
        result = span.end()
        assert result is None