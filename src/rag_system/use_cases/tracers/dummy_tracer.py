from contextlib import contextmanager


class DummyTracer:
    """A dummy tracer that does nothing. Used when tracing is disabled."""

    def start_span(self, name: str):
        """Start a dummy span."""
        return DummySpan(name)

    def get_tracer(self, name: str):
        """Get a dummy tracer."""
        return self

    @contextmanager
    def start_as_current_span(self, name: str):
        """Start a dummy span as current span (context manager)."""
        span = DummySpan(name)
        try:
            yield span
        finally:
            # Clean up if needed (dummy implementation does nothing)
            pass


class DummySpan:
    """A dummy span that does nothing."""

    def __init__(self, name: str):
        """Initialize a dummy span with a name."""
        self.name = name

    def set_attribute(self, key: str, value: str):
        """Set attribute on dummy span (does nothing)."""
        pass

    def end(self):
        """End dummy span (does nothing)."""
        pass
