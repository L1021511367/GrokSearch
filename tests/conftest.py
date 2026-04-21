import os

os.environ.setdefault("GROK_API_URL", "https://api.example/v1")
os.environ.setdefault("GROK_API_KEY", "test-key")
os.environ.setdefault("GROK_RETRY_MAX_ATTEMPTS", "2")
os.environ.setdefault("GROK_RETRY_MULTIPLIER", "0")
os.environ.setdefault("GROK_RETRY_MAX_WAIT", "1")
