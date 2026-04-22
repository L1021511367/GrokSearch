import httpx
import json
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import List, Optional
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_random_exponential
from tenacity.wait import wait_base
from zoneinfo import ZoneInfo
from .base import BaseSearchProvider, SearchResult
from ..utils import search_prompt, fetch_prompt, url_describe_prompt, rank_sources_prompt
from ..logger import log_info
from ..config import config


def get_local_time_info() -> str:
    """获取本地时间信息，用于注入到搜索查询中"""
    try:
        # 尝试获取系统本地时区
        local_tz = datetime.now().astimezone().tzinfo
        local_now = datetime.now(local_tz)
    except Exception:
        # 降级使用 UTC
        local_now = datetime.now(timezone.utc)

    # 格式化时间信息
    weekdays_cn = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    weekday = weekdays_cn[local_now.weekday()]

    return (
        f"[Current Time Context]\n"
        f"- Date: {local_now.strftime('%Y-%m-%d')} ({weekday})\n"
        f"- Time: {local_now.strftime('%H:%M:%S')}\n"
        f"- Timezone: {local_now.tzname() or 'Local'}\n"
    )


def _needs_time_context(query: str) -> bool:
    """检查查询是否需要时间上下文"""
    # 中文时间相关关键词
    cn_keywords = [
        "当前", "现在", "今天", "明天", "昨天",
        "本周", "上周", "下周", "这周",
        "本月", "上月", "下月", "这个月",
        "今年", "去年", "明年",
        "最新", "最近", "近期", "刚刚", "刚才",
        "实时", "即时", "目前",
    ]
    # 英文时间相关关键词
    en_keywords = [
        "current", "now", "today", "tomorrow", "yesterday",
        "this week", "last week", "next week",
        "this month", "last month", "next month",
        "this year", "last year", "next year",
        "latest", "recent", "recently", "just now",
        "real-time", "realtime", "up-to-date",
    ]

    query_lower = query.lower()

    for keyword in cn_keywords:
        if keyword in query:
            return True

    for keyword in en_keywords:
        if keyword in query_lower:
            return True

    return False

RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def _parse_responses_api(resp_json: dict) -> tuple[str, list[dict]]:
    """解析 xAI /v1/responses 返回的结构化 JSON：抽出正文 + 结构化信源列表。

    支持的两条信源路径（按优先级合并，URL 去重）：
    1. output[].content[].annotations[] 中的 url_citation
    2. output[type=web_search_call].action.sources[] 中的 URL 列表
    """
    answer_parts: list[str] = []
    sources: list[dict] = []
    seen: set[str] = set()

    def _push(url: str, title: str = "", snippet: str = "") -> None:
        u = (url or "").strip()
        if not u or not u.startswith(("http://", "https://")):
            return
        if u in seen:
            return
        seen.add(u)
        item: dict = {"url": u, "source_provider": "grok_live_search"}
        t = (title or "").strip()
        if t:
            item["title"] = t
        s = (snippet or "").strip()
        if s:
            item["snippet"] = s
        sources.append(item)

    # 顶层便捷字段
    if isinstance(resp_json, dict):
        top_output_text = resp_json.get("output_text")
        if isinstance(top_output_text, str) and top_output_text.strip():
            answer_parts.append(top_output_text)

        output_list = resp_json.get("output") or []
        if not isinstance(output_list, list):
            output_list = []

        for item in output_list:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")

            if item_type == "message":
                content_list = item.get("content") or []
                for block in content_list or []:
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get("type")
                    if block_type in ("output_text", "text"):
                        text_val = block.get("text") or ""
                        if isinstance(text_val, str) and text_val.strip():
                            if not answer_parts or answer_parts[-1] != text_val:
                                answer_parts.append(text_val)
                    annotations = block.get("annotations") or []
                    for ann in annotations or []:
                        if not isinstance(ann, dict):
                            continue
                        if ann.get("type") == "url_citation":
                            _push(
                                ann.get("url") or "",
                                ann.get("title") or "",
                                ann.get("snippet") or ann.get("description") or "",
                            )
            elif item_type == "web_search_call":
                action = item.get("action") or {}
                action_sources = action.get("sources") or []
                for s in action_sources or []:
                    if isinstance(s, str):
                        _push(s)
                    elif isinstance(s, dict):
                        _push(
                            s.get("url") or "",
                            s.get("title") or "",
                            s.get("snippet") or s.get("description") or "",
                        )

    answer = "\n".join(p for p in answer_parts if p).strip()
    return answer, sources


def _is_retryable_exception(exc) -> bool:
    """检查异常是否可重试"""
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.ConnectError, httpx.RemoteProtocolError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS_CODES
    return False


class _WaitWithRetryAfter(wait_base):
    """等待策略：优先使用 Retry-After 头，否则使用指数退避"""

    def __init__(self, multiplier: float, max_wait: int):
        self._base_wait = wait_random_exponential(multiplier=multiplier, max=max_wait)
        self._protocol_error_base = 3.0

    def __call__(self, retry_state):
        if retry_state.outcome and retry_state.outcome.failed:
            exc = retry_state.outcome.exception()
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
                retry_after = self._parse_retry_after(exc.response)
                if retry_after is not None:
                    return retry_after
            if isinstance(exc, httpx.RemoteProtocolError):
                return self._base_wait(retry_state) + self._protocol_error_base
        return self._base_wait(retry_state)

    def _parse_retry_after(self, response: httpx.Response) -> Optional[float]:
        """解析 Retry-After 头（支持秒数或 HTTP 日期格式）"""
        header = response.headers.get("Retry-After")
        if not header:
            return None
        header = header.strip()

        if header.isdigit():
            return float(header)

        try:
            retry_dt = parsedate_to_datetime(header)
            if retry_dt.tzinfo is None:
                retry_dt = retry_dt.replace(tzinfo=timezone.utc)
            delay = (retry_dt - datetime.now(timezone.utc)).total_seconds()
            return max(0.0, delay)
        except (TypeError, ValueError):
            return None


class GrokSearchProvider(BaseSearchProvider):
    def __init__(self, api_url: str, api_key: str, model: str = "grok-4-fast"):
        super().__init__(api_url, api_key)
        self.model = model

    def get_provider_name(self) -> str:
        return "Grok"

    async def search(self, query: str, platform: str = "", min_results: int = 3, max_results: int = 10, ctx=None) -> List[SearchResult]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        platform_prompt = ""

        if platform:
            platform_prompt = "\n\nYou should search the web for the information you need, and focus on these platform: " + platform + "\n"

        time_context = get_local_time_info() + "\n"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": search_prompt,
                },
                {"role": "user", "content": time_context + query + platform_prompt},
            ],
            "stream": True,
            "search_parameters": {
                "mode": "on",
                "return_citations": True,
                "max_search_results": max(min_results, max_results),
            },
        }

        await log_info(ctx, f"platform_prompt: { query + platform_prompt}", config.debug_enabled)

        return await self._execute_stream_with_retry(headers, payload, ctx)

    async def fetch(self, url: str, ctx=None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": fetch_prompt,
                },
                {"role": "user", "content": url + "\n获取该网页内容并返回其结构化Markdown格式" },
            ],
            "stream": True,
        }
        return await self._execute_stream_with_retry(headers, payload, ctx)

    async def _parse_streaming_response(self, response, ctx=None) -> str:
        content = ""
        full_body_buffer = [] 
        
        async for line in response.aiter_lines():
            line = line.strip()
            if not line:
                continue
            
            full_body_buffer.append(line)

            # 兼容 "data: {...}" 和 "data:{...}" 两种 SSE 格式
            if line.startswith("data:"):
                if line in ("data: [DONE]", "data:[DONE]"):
                    continue
                try:
                    # 去掉 "data:" 前缀，并去除可能的空格
                    json_str = line[5:].lstrip()
                    data = json.loads(json_str)
                    choices = data.get("choices", [])
                    if choices and len(choices) > 0:
                        delta = choices[0].get("delta", {})
                        if "content" in delta:
                            content += delta["content"]
                except (json.JSONDecodeError, IndexError):
                    continue
                
        if not content and full_body_buffer:
            try:
                full_text = "".join(full_body_buffer)
                data = json.loads(full_text)
                if "choices" in data and len(data["choices"]) > 0:
                    message = data["choices"][0].get("message", {})
                    content = message.get("content", "")
            except json.JSONDecodeError:
                pass
        
        await log_info(ctx, f"content: {content}", config.debug_enabled)

        return content

    async def _execute_stream_with_retry(self, headers: dict, payload: dict, ctx=None) -> str:
        """执行带重试机制的流式 HTTP 请求"""
        await log_info(ctx, f"HTTP endpoint: POST {self.api_url}/chat/completions", True)
        timeout = httpx.Timeout(connect=6.0, read=120.0, write=10.0, pool=None)

        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(config.retry_max_attempts + 1),
                wait=_WaitWithRetryAfter(config.retry_multiplier, config.retry_max_wait),
                retry=retry_if_exception(_is_retryable_exception),
                reraise=True,
            ):
                with attempt:
                    async with client.stream(
                        "POST",
                        f"{self.api_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    ) as response:
                        response.raise_for_status()
                        return await self._parse_streaming_response(response, ctx)

    async def _execute_json_with_retry(self, endpoint: str, headers: dict, payload: dict, ctx=None) -> dict:
        """非流式 JSON POST + 重试，用于 /v1/responses"""
        await log_info(ctx, f"HTTP endpoint: POST {endpoint}", True)
        timeout = httpx.Timeout(connect=6.0, read=120.0, write=10.0, pool=None)

        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(config.retry_max_attempts + 1),
                wait=_WaitWithRetryAfter(config.retry_multiplier, config.retry_max_wait),
                retry=retry_if_exception(_is_retryable_exception),
                reraise=True,
            ):
                with attempt:
                    response = await client.post(endpoint, headers=headers, json=payload)
                    response.raise_for_status()
                    return response.json()

    async def search_live(
        self,
        query: str,
        platform: str = "",
        ctx=None,
    ) -> tuple[str, list[dict]]:
        """通过 xAI Responses API + web_search 工具执行真实联网搜索。

        返回 (answer, sources)。answer 为文本正文，sources 为结构化信源列表，
        每条含 {url, title, snippet, source_provider}。
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        user_text = query
        if platform:
            user_text = (
                f"{query}\n\nFocus on these platforms: {platform}\n"
            )
        if _needs_time_context(query):
            user_text = get_local_time_info() + "\n" + user_text

        payload = {
            "model": self.model,
            "input": user_text,
            "tools": [{"type": "web_search"}],
            "include": [
                "web_search_call.action.sources",
            ],
        }

        endpoint = f"{self.api_url}/responses"
        await log_info(ctx, f"search_live payload: model={self.model} query={query}", config.debug_enabled)

        resp_json = await self._execute_json_with_retry(endpoint, headers, payload, ctx)
        await log_info(ctx, f"search_live raw: {json.dumps(resp_json)[:2000]}", config.debug_enabled)

        return _parse_responses_api(resp_json)

    async def describe_url(self, url: str, ctx=None) -> dict:
        """让 Grok 阅读单个 URL 并返回 title + extracts"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": url_describe_prompt},
                {"role": "user", "content": url},
            ],
            "stream": True,
        }
        result = await self._execute_stream_with_retry(headers, payload, ctx)
        title, extracts = url, ""
        for line in result.strip().splitlines():
            if line.startswith("Title:"):
                title = line[6:].strip() or url
            elif line.startswith("Extracts:"):
                extracts = line[9:].strip()
        return {"title": title, "extracts": extracts, "url": url}

    async def rank_sources(self, query: str, sources_text: str, total: int, ctx=None) -> list[int]:
        """让 Grok 按查询相关度对信源排序，返回排序后的序号列表"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": rank_sources_prompt},
                {"role": "user", "content": f"Query: {query}\n\n{sources_text}"},
            ],
            "stream": True,
        }
        result = await self._execute_stream_with_retry(headers, payload, ctx)
        order: list[int] = []
        seen: set[int] = set()
        for token in result.strip().split():
            try:
                n = int(token)
                if 1 <= n <= total and n not in seen:
                    seen.add(n)
                    order.append(n)
            except ValueError:
                continue
        # 补齐遗漏的序号
        for i in range(1, total + 1):
            if i not in seen:
                order.append(i)
        return order
