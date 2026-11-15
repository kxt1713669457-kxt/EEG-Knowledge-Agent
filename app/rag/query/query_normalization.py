"""为英文EEG语料补充检索同义词，不改写事实或预填答案数值。"""
import re


def expand_eeg_retrieval_terms(query: str) -> str:
    compact = re.sub(r"\s+", "", query.casefold())
    expansions = []
    for names, terms in (
        (("采样率", "采样频率"), "sampling rate sampling frequency"),
        (("eeg通道",), "EEG channels EEG recordings"),
        (("反馈",), "feedback period trial timing cue"),
        (("开始时间", "起始时间", "起点"), "onset start"),
        (("结束时间", "终止时间", "终点"), "offset end"),
    ):
        if any(name in compact for name in names) and terms.casefold() not in query.casefold():
            expansions.append(terms)
    return query if not expansions else query + "\nRetrieval terms: " + "; ".join(expansions)
