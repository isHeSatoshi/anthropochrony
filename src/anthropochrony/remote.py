"""Authorization failures are sticky for the duration of a remote operation."""

import threading
from contextlib import contextmanager

from huggingface_hub.utils import get_session


@contextmanager
def authorization_guard():
    client = get_session()
    denied = threading.Event()

    def request_guard(request):
        if denied.is_set():
            raise PermissionError("HF authorization failed; stop remote work and renew access")

    def response_guard(response):
        if response.status_code in {401, 403}:
            denied.set()
            request_guard(None)

    hooks = (("request", request_guard), ("response", response_guard))
    for name, hook in hooks:
        client.event_hooks.setdefault(name, []).append(hook)
    try:
        yield
        if denied.is_set():
            request_guard(None)
    finally:
        for name, hook in hooks:
            client.event_hooks[name].remove(hook)
