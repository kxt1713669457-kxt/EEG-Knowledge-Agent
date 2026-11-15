"""回答交付前的结构完整性检查；不是通用事实正确性证明。"""
import math
import re


class IncompleteAnswerError(ValueError):
    def __init__(self, reasons, audit=None):
        super().__init__("模型未生成符合本次要求的完整回答，请重试或缩小问题。")
        self.reasons, self.audit = reasons, audit or {}


def _number(value):
    if value.isdigit():
        return int(value)
    digits = {c: n for n, c in enumerate("零一二三四五六七八九")}
    digits["两"] = 2
    total, current = 0, 0
    for char in value:
        if char in digits:
            current = digits[char]
        elif char in {"十", "百", "千"}:
            total += (current or 1) * {"十": 10, "百": 100, "千": 1000}[char]
            current = 0
    return total + current


def requested_length(question):
    found = re.search(r"([0-9一二两三四五六七八九十百千]+)\s*(字|词|words?)", question, re.I)
    if not found:
        return None
    target = _number(found[1])
    return (target, "words" if found[2].lower() in {"词", "word", "words"} else "characters") if target > 0 else None


def text_size(answer, unit="characters"):
    text = re.sub(r"\[[EA]\d+\]|[#*`|>_-]", "", str(answer or ""))
    return len(re.findall(r"\b[\w'-]+\b", text)) if unit == "words" else len(re.sub(r"\s", "", text))


def output_budget(question, base=1024):
    requested = requested_length(question)
    if not requested:
        return base
    target, unit = requested
    # 应用输出预留，不宣称等同远端tokenizer。长篇有明确上界，超限仍不截断交付。
    return max(base, min(4096, math.ceil(target * (1.8 if unit == "words" else 1.6)) + 512))


def answer_issues(question, answer, finish_reason=None):
    if finish_reason == "length":
        return ["output_limit"]
    if finish_reason not in {None, "stop"}:
        return ["invalid_finish_reason"]
    if not isinstance(answer, str) or not answer.strip():
        return ["empty_answer"]
    text = answer.strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    issues = []
    if lines and all(re.fullmatch(r"#{1,6}\s*.+", line) for line in lines):
        issues.append("heading_only")
    if re.search(r"(?:^|\n)\s*#{1,6}\s*$", text):
        issues.append("unfinished_heading")
    detailed = bool(re.search(r"总结|分析|解释|如何|怎么|比较|对比|为什么|原因|结果|说明|步骤", question))
    brief = bool(re.search(r"只(?:需|要|用)?(?:回答|返回|说|输出).{0,8}(?:数字|名称|名字|是或否|一个词|标题)|仅(?:返回|输出)标题", question))
    if detailed and not brief:
        if text_size(text) < 16 and not re.search(r"[。！？!?]", text):
            issues.append("label_instead_of_answer")
        elif (re.search(r"[\u4e00-\u9fff]", text) and not re.search(r"[。！？!?：:；;）)\]。]$", text)
              and len(lines) <= 3 and not re.match(r"(?:[-*+]\s|\d+[.)]\s|\||```)",lines[-1])):
            issues.append("unfinished_sentence")
    requested = requested_length(question)
    if len(re.findall(r"(?m)^\s*(?:`{3,}|~{3,})",text))%2:
        issues.append("unclosed_code_block")
    if requested and requested[0] >= 300 and text_size(text, requested[1]) < math.ceil(requested[0] * 0.8):
        issues.append("requested_length_not_covered")
    return list(dict.fromkeys(issues))
