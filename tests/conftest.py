import sys

# In Python 3.12+, nest_asyncio modifies asyncio internals in a way that breaks
# asyncio.current_task() used by AnyIO and Starlette/FastAPI TestClient.
# Ensure nest_asyncio.apply is a safe no-op on Python 3.12+ during test runs.
try:
    import nest_asyncio
    if sys.version_info >= (3, 12):
        nest_asyncio.apply = lambda *args, **kwargs: None
except ImportError:
    pass
