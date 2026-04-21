import httpx
import pytest
import respx

from grok_search.providers.grok import (
    GrokSearchProvider,
    _parse_responses_api,
)


def test_parse_responses_api_two_paths_dedup():
    resp = {
        "output": [
            {
                "type": "web_search_call",
                "action": {
                    "sources": [
                        {"url": "https://shared.example/a", "title": "Shared"},
                        {"url": "https://only-ws.example/b", "title": "WS"},
                    ]
                },
            },
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Answer body.",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url": "https://shared.example/a",
                                "title": "From Citation",
                                "snippet": "note",
                            },
                            {
                                "type": "url_citation",
                                "url": "https://only-cit.example/c",
                                "title": "CitOnly",
                            },
                        ],
                    }
                ],
            },
        ]
    }
    ans, srcs = _parse_responses_api(resp)
    assert ans == "Answer body."
    urls = [s["url"] for s in srcs]
    assert urls == [
        "https://shared.example/a",
        "https://only-ws.example/b",
        "https://only-cit.example/c",
    ]
    assert all(s["source_provider"] == "grok_live_search" for s in srcs)
    # first-seen-wins: shared URL keeps the web_search_call title "Shared", not "From Citation"
    assert srcs[0]["title"] == "Shared"


@pytest.mark.asyncio
@respx.mock
async def test_search_live_happy_path():
    endpoint = "https://api.example/v1/responses"
    respx.post(endpoint).mock(
        return_value=httpx.Response(
            200,
            json={
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Beijing is sunny.",
                                "annotations": [
                                    {
                                        "type": "url_citation",
                                        "url": "https://weather.com.cn/a",
                                        "title": "Weather",
                                    }
                                ],
                            }
                        ],
                    }
                ]
            },
        )
    )
    provider = GrokSearchProvider("https://api.example/v1", "test-key", "grok-4.20-expert")
    answer, sources = await provider.search_live("today weather in Beijing")
    assert answer == "Beijing is sunny."
    assert len(sources) == 1
    assert sources[0]["url"] == "https://weather.com.cn/a"
    assert sources[0]["source_provider"] == "grok_live_search"


@pytest.mark.asyncio
@respx.mock
async def test_search_live_retries_on_500_then_success():
    endpoint = "https://api.example/v1/responses"
    route = respx.post(endpoint)
    route.side_effect = [
        httpx.Response(500, json={"error": "oops"}),
        httpx.Response(
            200,
            json={
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Recovered.",
                                "annotations": [
                                    {
                                        "type": "url_citation",
                                        "url": "https://r.example/x",
                                    }
                                ],
                            }
                        ],
                    }
                ]
            },
        ),
    ]
    provider = GrokSearchProvider("https://api.example/v1", "test-key", "grok-4.20-expert")
    answer, sources = await provider.search_live("query")
    assert answer == "Recovered."
    assert sources[0]["url"] == "https://r.example/x"
    assert route.call_count >= 2


@pytest.mark.asyncio
@respx.mock
async def test_search_live_payload_includes_web_search_tool():
    endpoint = "https://api.example/v1/responses"
    captured = {}

    def _capture(request):
        import json as _json
        captured["body"] = _json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json={"output": []})

    respx.post(endpoint).mock(side_effect=_capture)
    provider = GrokSearchProvider("https://api.example/v1", "test-key", "grok-4.20-expert")
    await provider.search_live("q")
    body = captured["body"]
    assert body["model"] == "grok-4.20-expert"
    assert body["tools"] == [{"type": "web_search"}]
    assert "web_search_call.action.sources" in body["include"]


def test_live_search_eligible_models():
    from grok_search.config import config
    # Intentionally empty: Guda proxy returns HTTP 500 on /v1/responses.
    # All models are forced onto legacy /v1/chat/completions path with
    # search_parameters to trigger real web search.
    assert config.LIVE_SEARCH_ELIGIBLE_MODELS == frozenset()
