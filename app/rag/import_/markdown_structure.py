"""保护Markdown结构；普通段落递归切分，图/代码/HTML表格独立处理。"""
import re


PROTECTED_BLOCK = re.compile(
    r'(?P<table><table\b[^>]*>.*?</table\s*>)'
    r'|(?P<code>^(?P<fence>`{3,}|~{3,})[^\n]*\n.*?^(?P=fence)[ \t]*$)'
    r'|(?P<image>^!\[(?:\\.|[^\]\\\n])*\]\([^\n]+\)[ \t]*'
    r'(?:\n\s*\n> 图像说明（模型生成）：[^\n]*)?)',
    re.MULTILINE | re.DOTALL | re.IGNORECASE,
)


def structure_blocks(text: str):
    cursor = 0
    for match in PROTECTED_BLOCK.finditer(text):
        if text[cursor:match.start()].strip():
            yield "prose", text[cursor:match.start()].strip()
        kind = next(key for key in ("table", "code", "image") if match[key] is not None)
        yield kind, match.group(0).strip()
        cursor = match.end()
    if text[cursor:].strip():
        yield "prose", text[cursor:].strip()


def table_parts(table: str, budget: int) -> list[str]:
    """长的简单表按行拆并重复表头；跨行单元格保持整体，避免错配。"""
    if len(table) <= budget or re.search(r'\browspan\s*=', table, re.I):
        return [table]
    rows = re.findall(r'<tr\b[^>]*>.*?</tr\s*>', table, re.S | re.I)
    opening = re.match(r'<table\b[^>]*>', table, re.I)
    if not rows or opening is None:
        return [table]
    # 非行结构（如caption、嵌套table）不在这里擅自丢弃。
    rest = re.sub(r'<tr\b[^>]*>.*?</tr\s*>', '', table, flags=re.S | re.I)
    rest = re.sub(r'</?(?:table|thead|tbody|tfoot)\b[^>]*>', '', rest, flags=re.I)
    if rest.strip():
        return [table]
    header_count = 0
    for row in rows:
        if re.search(r'<th\b', row, re.I):
            header_count += 1
        else:
            break
    header_count = header_count or 1
    header = ''.join(rows[:header_count])
    prefix, suffix = opening.group(0) + header, '</table>'
    parts, group = [], []
    for row in rows[header_count:]:
        if group and len(prefix + ''.join(group) + row + suffix) > budget:
            parts.append(prefix + ''.join(group) + suffix)
            group = []
        group.append(row)
    if group:
        parts.append(prefix + ''.join(group) + suffix)
    return parts or [table]
