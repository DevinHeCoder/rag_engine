"""P6 评估层：生成答案忠实度（faithfulness）评估。

衡量生成的答案是否忠实于检索到的上下文（不编造上下文之外的事实）。
使用 LLM 逐条判断：答案中的陈述能否由上下文支持。
"""
import re
from typing import List, Optional

from src.common.exceptions import EvaluationError
from src.llm_client.base import BaseLLMClient
from src.retriever.base import RetrieveResult

_NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+")


def evaluate_faithfulness(
    question: str,
    answer: str,
    contexts: List[RetrieveResult],
    llm_client: Optional[BaseLLMClient] = None,
    judge_prompt_fn=None,
    max_context_chars: int = 1200,
) -> float:
    """评估单条答案的忠实度，返回 0-1 分数。

    默认使用 LLM 判断；若未提供 llm_client，退化为简单规则：
    答案中引用的数字/专有名词是否在上下文中出现（粗略近似）。
    """
    if llm_client is not None:
        return _judge_with_llm(
            question, answer, contexts, llm_client, judge_prompt_fn, max_context_chars
        )
    return _judge_heuristic(question, answer, contexts)


def _build_context_text(contexts: List[RetrieveResult], max_chars: int) -> str:
    parts = []
    used = 0
    for i, c in enumerate(contexts, start=1):
        snippet = c.content[: max(0, max_chars - used)]
        if not snippet:
            break
        parts.append(f"[{i}] {snippet}")
        used += len(snippet)
    return "\n".join(parts)


def _judge_with_llm(
    question: str,
    answer: str,
    contexts: List[RetrieveResult],
    llm_client: BaseLLMClient,
    judge_prompt_fn,
    max_context_chars: int,
) -> float:
    context_text = _build_context_text(contexts, max_context_chars)
    if judge_prompt_fn is None:
        prompt = (
            "你是忠实度评估专家。请判断以下答案中的信息是否都能由给定上下文支持，"
            "有没有编造上下文之外的事实。只输出 0 到 1 之间的一个分数，"
            "完全忠于上下文为 1，完全编造为 0，不要输出其他文字。\n\n"
            f"上下文：\n{context_text}\n\n"
            f"问题：{question}\n\n"
            f"答案：{answer}\n\n"
            "忠实度分数："
        )
    else:
        prompt = judge_prompt_fn(question, answer, context_text)

    try:
        resp = llm_client.chat([{"role": "user", "content": prompt}])
    except Exception as e:
        raise EvaluationError(f"LLM 忠实度评估失败: {e}") from e

    match = _NUMBER_PATTERN.search(resp or "")
    if not match:
        return 0.0
    try:
        score = float(match.group())
    except ValueError:
        return 0.0
    return max(0.0, min(1.0, score))


def _judge_heuristic(question: str, answer: str, contexts: List[RetrieveResult]) -> float:
    """无 LLM 时的粗略忠实度近似：答案中的"信息词"出现在上下文中的比例。

    用连续的 2-4 字中文片段或英文单词做"信息词"，检查是否在上下文中出现。
    注意：这只是近似启发，适合快速自检，不能替代 LLM 判断。
    """
    context_text = " ".join(c.content for c in contexts)
    if not answer.strip() or not context_text.strip():
        return 0.0

    # 提取英文单词（>=2 字符）作为信息词
    words = re.findall(r"[A-Za-z]{2,}", answer.lower())
    # 提取中文 4-gram（中文信息主要在词组，用 4 字滑窗）
    cjk = re.sub(r"[^\u4e00-\u9fff]", "", answer)
    grams = [cjk[i : i + 4] for i in range(max(0, len(cjk) - 3))] if len(cjk) >= 4 else ([cjk] if cjk else [])

    info_tokens = [w for w in words if len(w) >= 3] + [g for g in grams if len(g) == 4]
    if not info_tokens:
        return 0.0

    ctx_lower = context_text.lower()
    hit = sum(1 for t in info_tokens if t in ctx_lower)
    return hit / len(info_tokens)
