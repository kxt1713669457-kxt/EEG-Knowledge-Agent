"""区分查官方入口元信息与要求原文支持实验事实。"""
import re


def official_page_lookup(query):
    return (bool(re.search(r'官方.*(?:页面|网址|链接|入口)|官网',query))
            and not re.search(r'原文|原件|采样|通道|电极|实验|准确率|训练|测试|时间窗|算法细节',query))


def requires_original(query,source_requirement='auto'):
    if source_requirement!='auto':return source_requirement=='original'
    return bool(re.search(r'原文|原件',query) or re.search(r'官方|官网',query) and not official_page_lookup(query))
