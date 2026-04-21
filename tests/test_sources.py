from grok_search.sources import (
    split_answer_and_sources,
    merge_sources,
    _MD_LINK_PATTERN,
)


def test_nested_bracket_citation_captured():
    text = "Beijing is sunny [[1]](https://weather.com.cn/a) today."
    ans, srcs = split_answer_and_sources(text)
    assert len(srcs) == 1
    assert srcs[0]["url"] == "https://weather.com.cn/a"
    assert srcs[0].get("title") == "[1]"


def test_plain_markdown_link_no_regression():
    text = "See [Weather China](https://weather.com.cn/b) for details."
    ans, srcs = split_answer_and_sources(text)
    assert len(srcs) == 1
    assert srcs[0]["url"] == "https://weather.com.cn/b"
    assert srcs[0].get("title") == "Weather China"


def test_url_dedup_first_title_wins():
    text = (
        "Check [First](https://example.com/page) and later "
        "[Second](https://example.com/page) too."
    )
    ans, srcs = split_answer_and_sources(text)
    assert len(srcs) == 1
    assert srcs[0]["url"] == "https://example.com/page"
    assert srcs[0].get("title") == "First"


def test_citation_card_block_parsed():
    text = (
        "Answer body here.\n\n"
        "```citation_card\n"
        "url: https://example.com/a\n"
        "title: Example A\n"
        "snippet: A short note\n"
        "```\n"
        "```citation_card\n"
        "url: https://example.com/b\n"
        "title: Example B\n"
        "```\n"
    )
    ans, srcs = split_answer_and_sources(text)
    assert "citation_card" not in ans
    assert ans.strip().startswith("Answer body")
    assert len(srcs) == 2
    urls = [s["url"] for s in srcs]
    assert urls == ["https://example.com/a", "https://example.com/b"]


def test_empty_input_returns_empty():
    ans, srcs = split_answer_and_sources("")
    assert ans == ""
    assert srcs == []


def test_malformed_citation_card_tolerated():
    text = (
        "Body.\n"
        "```citation_card\n"
        "no_url_here: whatever\n"
        "```\n"
        "And also [link](https://fallback.example/x)\n"
    )
    ans, srcs = split_answer_and_sources(text)
    urls = [s["url"] for s in srcs]
    assert "https://fallback.example/x" in urls


def test_inline_only_fallback_triggers():
    text = (
        "Scattered references throughout prose: first at "
        "[[1]](https://a.example/page) and second later "
        "[[2]](https://b.example/page). No headings, no blocks."
    )
    ans, srcs = split_answer_and_sources(text)
    urls = {s["url"] for s in srcs}
    assert urls == {"https://a.example/page", "https://b.example/page"}
    assert ans == text


def test_merge_sources_first_seen_wins():
    a = [{"url": "https://x.com", "title": "First"}]
    b = [{"url": "https://x.com", "title": "Second"}, {"url": "https://y.com"}]
    merged = merge_sources(a, b)
    urls = [m["url"] for m in merged]
    assert urls == ["https://x.com", "https://y.com"]
    assert merged[0].get("title") == "First"


def test_url_preserves_query_string():
    text = "See [link](https://example.com/q?a=1&b=2) here."
    ans, srcs = split_answer_and_sources(text)
    assert len(srcs) == 1
    assert srcs[0]["url"] == "https://example.com/q?a=1&b=2"


def test_regex_matches_nested_bracket_only():
    m = _MD_LINK_PATTERN.search("[[1]](https://a.example)")
    assert m is not None
    assert m.group(1) == "[1]"
    assert m.group(2) == "https://a.example"
