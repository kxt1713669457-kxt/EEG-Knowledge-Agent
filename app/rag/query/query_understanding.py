"""查询理解的数据契约、历史上下文和保守的主题对齐。"""
import json
import re

from pydantic import BaseModel, Field, StrictBool, field_validator, model_validator

# 工程上的历史输入预算，单位为字符；不是已实测的最优模型token窗口。
HISTORY_MESSAGE_LIMIT = 10
HISTORY_CHAR_BUDGET = 6000
HISTORY_MESSAGE_CHAR_BUDGET = 1500


class QueryUnderstanding(BaseModel):
    item_names: list[str] = Field(default_factory=list)
    rewritten_query: str = Field(min_length=1)
    needs_clarification: StrictBool = False
    clarification_question: str = ""

    @field_validator("rewritten_query", "clarification_question")
    @classmethod
    def strip_text(cls, value):
        return value.strip()

    @field_validator("item_names")
    @classmethod
    def clean_entities(cls, values):
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))

    @model_validator(mode="after")
    def validate_meaning(self):
        if not self.rewritten_query:
            raise ValueError("改写问题不能为空")
        if self.needs_clarification and not self.clarification_question:
            raise ValueError("需要澄清时必须提供具体澄清问题")
        return self


def validate_understanding(result: dict, original_query: str) -> dict:
    if not isinstance(result, dict):
        raise ValueError("问题理解结果必须是JSON对象")
    return QueryUnderstanding.model_validate({"rewritten_query": original_query, **result}).model_dump()


def comparison_pronoun_clarification(query: str, messages: list[dict]) -> str:
    """已复现的边界：比较后的单数指代不能擅改成双对象问题或任选一个。"""
    singular = re.match(r"^\s*(?:它(?!们)|这个(?:模型|方法|数据集)|该(?:模型|方法|数据集))", query)
    if not singular:
        return ""
    previous = next((row for row in reversed(messages) if row.get("role") == "user"), None)
    if previous and re.search(r"比较|对比|\bvs\b|\bversus\b", previous.get("text") or "", re.IGNORECASE):
        return "上文在比较多个对象，您这里的‘它’具体指哪种方法、模型或数据集？"
    return ""


def format_history_messages(messages: list[dict]) -> str:
    """不按主题标签丢消息；保留角色及原问/改写，按最近消息预算裁剪。"""
    blocks = []
    for message in messages or []:
        if message.get("role") not in {"user", "assistant"}:
            continue
        text = message.get("text") or ""
        rewritten = message.get("rewritten_query") or ""
        body = str(text)
        if message["role"] == "user" and rewritten and rewritten != text:
            body += f"\n检索改写:{rewritten}"
        if not body.strip():
            continue
        if len(body) > HISTORY_MESSAGE_CHAR_BUDGET:
            body = body[:HISTORY_MESSAGE_CHAR_BUDGET] + "[本条历史已裁剪]"
        names = message.get("item_names") or []
        blocks.append(f"角色:{message['role']}\n内容:{body}\n历史主题标签:{json.dumps(names, ensure_ascii=False)}")
    kept, used = [], 0
    for block in reversed(blocks):
        if used + len(block) + 2 > HISTORY_CHAR_BUDGET:
            break
        kept.append(block)
        used += len(block) + 2
    if not kept:
        return "无对话记录。"
    prefix = "[历史上下文已裁剪，仅保留最近消息]\n" if len(kept) < len(blocks) else ""
    return prefix + "\n\n".join(reversed(kept))


def topic_key(name: str) -> str:
    normalized = re.sub(r"[\W_]+", "", name.casefold())
    aliases = {
        "d03csp":"csp", "mnecsp":"csp",
        "withinsession":"withinsessionevaluation", "crosssession":"crosssessionevaluation", "crosssubject":"crosssubjectevaluation",
        "moabbwithinsessionevaluationofficialdocumentation":"withinsessionevaluation",
        "moabbcrosssessionevaluationofficialdocumentation":"crosssessionevaluation",
        "d04crosssubject":"crosssubjectevaluation", "moabbevaluationscrosssubjectevaluation":"crosssubjectevaluation",
        "bcicompetitioniv2a": "bciiv2a", "bcicompetitionivdataset2a": "bciiv2a",
        "bciciv2a": "bciiv2a", "bciiv2a": "bciiv2a", "2a": "bciiv2a",
        "bcicompetitioniv2b": "bciiv2b", "bcicompetitionivdataset2b": "bciiv2b",
        "bciciv2b": "bciiv2b", "bciiv2b": "bciiv2b", "2b": "bciiv2b",
        "bcicompetitionivdataset2bgrazb": "bciiv2b",
        "bcicompetitionivdataset2agraza": "bciiv2a",
        "moabbevaluationswithinsessionevaluation": "withinsessionevaluation",
        "moabbevaluationscrosssessionevaluation": "crosssessionevaluation",
    }
    return aliases.get(normalized, normalized)


def topic_filter(item_names: list[str]) -> str | None:
    return f"item_name in {json.dumps(item_names, ensure_ascii=False)}" if item_names else None


def topic_variants(name: str) -> list[str]:
    """显式登记的写法；未知别名不靠相似度猜成另一个主体。"""
    groups = {
        "csp":["CSP","MNE CSP","D03_csp"],
        "crosssubjectevaluation":["CrossSubjectEvaluation","CrossSubject","moabb.evaluations.CrossSubjectEvaluation","D04_cross_subject"],
        "bciiv2a": ["BCI Competition IV 2a", "BCI Competition IV Dataset 2a", "BCI Competition IV data set 2a Graz A", "BCI IV 2a", "BCICIV_2a", "2a"],
        "bciiv2b": ["BCI Competition IV 2b", "BCI Competition IV Dataset 2b", "BCI Competition IV data set 2b Graz B", "BCI IV 2b", "BCICIV_2b", "2b"],
        "withinsessionevaluation": ["WithinSessionEvaluation", "moabb.evaluations.WithinSessionEvaluation","MOABB WithinSessionEvaluation official documentation"],
        "crosssessionevaluation": ["CrossSessionEvaluation", "moabb.evaluations.CrossSessionEvaluation","MOABB CrossSessionEvaluation official documentation"],
    }
    return list(dict.fromkeys([name, *groups.get(topic_key(name), [])]))


def preserve_named_document_scope(query,entities):
    """明确命名的示例高于通用API词；比较对象不因库标签写法而丢失。"""
    if re.search(r'(?<![A-Za-z0-9_])MNE\s+CSP(?![A-Za-z0-9_]).*示例',query,re.I):return ['MNE CSP']
    names=[n for n in ['WithinSession','CrossSession','CrossSubject'] if re.search(r'(?<![A-Za-z0-9_])'+n+r'(?:Evaluation)?(?![A-Za-z0-9_])',query,re.I)]
    if names:
        return list(dict.fromkeys(names+[n for n in entities if n.casefold() not in {'moabb'} and topic_key(n) not in {'withinsessionevaluation','crosssessionevaluation','crosssubjectevaluation'}]))
    datasets=re.findall(r'(?<![A-Za-z0-9])2[ab](?![A-Za-z0-9])',query,re.I)
    if len(set(x.lower()for x in datasets))==2 and re.search(r'对比|比较|compare',query,re.I):
        return ['BCI IV 2a','BCI IV 2b']
    return entities


def validate_retrieval_input(state: dict) -> tuple[list[str], str]:
    query = state.get("rewritten_query")
    names = state.get("item_names") or []
    if not isinstance(query, str) or not query.strip():
        raise ValueError("检索问题不能为空")
    if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
        raise ValueError("主题过滤必须是字符串列表")
    return names, query
