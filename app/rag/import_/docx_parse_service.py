"""读取DOCX正文OOXML，保留标题、表格与图片的顺序，生成Markdown。"""
import html
import json
import posixpath
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.parse_result_service import record_parse_result

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
    "v": "urn:schemas-microsoft-com:vml",
}


def _tag(namespace, name):
    return f"{{{NS[namespace]}}}{name}"


class _DocxReader:
    def __init__(self, package, output):
        self.package, self.output = package, output
        self.images, self.equations, self.warnings = {}, [], []
        self.table_count = 0
        self.relationships, self.heading_styles = {}, {}
        if "word/_rels/document.xml.rels" in package.namelist():
            root = ET.fromstring(package.read("word/_rels/document.xml.rels"))
            self.relationships = {item.get("Id"): item.attrib for item in root}
        if "word/styles.xml" in package.namelist():
            for style in ET.fromstring(package.read("word/styles.xml")):
                outline, name = style.find("w:pPr/w:outlineLvl", NS), style.find("w:name", NS)
                level = self.heading_level(name.get(_tag("w", "val"), "") if name is not None else "")
                if outline is not None:
                    number = int(outline.get(_tag("w", "val"), "9"))
                    level = min(6, number + 1) if 0 <= number <= 8 else 0
                if level:
                    self.heading_styles[style.get(_tag("w", "styleId"))] = level

    def warn(self, message):
        if message not in self.warnings:
            self.warnings.append(message)

    @staticmethod
    def heading_level(name):
        match = re.fullmatch(r"(?:heading|标题)\s*([1-6])", name, re.IGNORECASE)
        return int(match.group(1)) if match else (1 if name.lower() == "title" else 0)

    def image(self, relationship_id):
        if relationship_id in self.images:
            return self.images[relationship_id]
        relation = self.relationships.get(relationship_id, {})
        target = relation.get("Target", "")
        if not target or relation.get("TargetMode") == "External":
            self.warn("外部或缺失的图片引用未下载，需查看原件")
            return "[图片见原件]"
        member = posixpath.normpath(target.lstrip("/") if target.startswith("/") else "word/" + target)
        if not member.startswith("word/media/") or member not in self.package.namelist():
            raise ValueError("DOCX图片引用无法在包内定位")
        suffix = Path(member).suffix.lower()
        filename = f"image_{len(self.images) + 1:04d}{suffix}"
        directory = self.output / "images"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / filename).write_bytes(self.package.read(member))
        if suffix not in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}:
            self.warn("存在图片增强器不支持的格式，原始图片已保留")
        reference = f"![DOCX图片](images/{filename})"
        self.images[relationship_id] = reference
        return reference

    def inline(self, element):
        if element.tag == _tag("w", "del"):
            self.warn("存在修订，输出包含插入并排除删除的视图")
            return ""
        if element.tag == _tag("w", "t"):
            return element.text or ""
        if element.tag in {_tag("w", "br"), _tag("w", "cr")}:
            return "\n"
        if element.tag == _tag("w", "tab"):
            return "\t"
        if element.tag == _tag("a", "blip"):
            return self.image(element.get(_tag("r", "embed")))
        if element.tag == _tag("v", "imagedata"):
            return self.image(element.get(_tag("r", "id")))
        if element.tag == _tag("m", "oMath"):
            number = len(self.equations) + 1
            self.equations.append({"id": number, "omml": ET.tostring(element, encoding="unicode")})
            self.warn("OMML公式保存在equations.json，未转换为LaTeX")
            return f"[公式{number}见原件及equations.json]"
        if element.tag == _tag("w", "hyperlink"):
            text = "".join(self.inline(child) for child in element)
            target = self.relationships.get(element.get(_tag("r", "id")), {}).get("Target", "")
            if target.startswith(("https://", "http://")):
                return f"[{text}]({target.replace(' ', '%20').replace(')', '%29')})"
            return text
        return "".join(self.inline(child) for child in element)

    def paragraph(self, paragraph):
        text = self.inline(paragraph).strip()
        style, outline = paragraph.find("w:pPr/w:pStyle", NS), paragraph.find("w:pPr/w:outlineLvl", NS)
        name = style.get(_tag("w", "val"), "") if style is not None else ""
        level = self.heading_styles.get(name, self.heading_level(name))
        if outline is not None:
            number = int(outline.get(_tag("w", "val"), "9"))
            level = min(6, number + 1) if 0 <= number <= 8 else 0
        if text and level:
            return "#" * level + " " + text
        if text and paragraph.find("w:pPr/w:numPr", NS) is not None:
            self.warn("列表顺序保留，自动编号统一表示为列表标记")
            return "- " + text
        return text

    def blocks(self, parent):
        for block in parent:
            if block.tag == _tag("w", "p"):
                yield self.paragraph(block)
            elif block.tag == _tag("w", "tbl"):
                yield self.table(block)
            elif block.tag in {_tag("w", "sdt"), _tag("w", "sdtContent"), _tag("w", "ins")}:
                yield from self.blocks(block)
            elif block.tag == _tag("w", "altChunk"):
                self.warn("嵌入的altChunk对象未展开，需查看原件")

    def table(self, table):
        self.table_count += 1
        rows, vertical = [], {}
        for row in table.findall("w:tr", NS):
            cells, column = [], 0
            for cell in row.findall("w:tc", NS):
                span_node = cell.find("w:tcPr/w:gridSpan", NS)
                span = int(span_node.get(_tag("w", "val"), "1")) if span_node is not None else 1
                merge = cell.find("w:tcPr/w:vMerge", NS)
                continuation = merge is not None and merge.get(_tag("w", "val")) != "restart"
                if continuation and column in vertical and vertical[column]["colspan"] == span:
                    vertical[column]["rowspan"] += 1
                else:
                    if continuation:
                        self.warn("存在未关联起始单元格的纵向合并，请核对原件")
                    content = self.cell_content(cell)
                    record = {"text": content, "colspan": span, "rowspan": 1}
                    cells.append(record)
                    vertical.pop(column, None)
                    if merge is not None and not continuation:
                        vertical[column] = record
                column += span
            rows.append(cells)
        rendered = []
        for row in rows:
            cells = []
            for cell in row:
                attributes = "".join(f' {name}="{cell[name]}"' for name in ("colspan", "rowspan") if cell[name] > 1)
                cells.append(f"<td{attributes}>{cell['text']}</td>")
            rendered.append("<tr>" + "".join(cells) + "</tr>")
        return "<table>\n" + "\n".join(rendered) + "\n</table>"

    def cell_content(self, cell):
        parts = []
        for child in cell:
            if child.tag == _tag("w", "p"):
                parts.append(html.escape(self.paragraph(child)))
            elif child.tag == _tag("w", "tbl"):
                parts.append(self.table(child))
            elif child.tag in {_tag("w", "sdt"), _tag("w", "sdtContent")}:
                parts.append(self.cell_content(child))
        return "<br/>".join(part for part in parts if part)


def parse_docx_to_markdown(state: ImportGraphState) -> ImportGraphState:
    source = Path(state["docx_path"])
    output = Path(state.get("local_dir") or source.parent) / "docx_markdown"
    output.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(source) as package:
            reader = _DocxReader(package, output)
            root = ET.fromstring(package.read("word/document.xml"))
            body = root.find("w:body", NS)
            if body is None:
                raise ValueError("DOCX缺少正文")
            content = "\n\n".join(block for block in reader.blocks(body) if block).strip()
            if not content:
                raise ValueError("DOCX正文为空")
            for member, label in (("word/footnotes.xml", "脚注"), ("word/endnotes.xml", "尾注")):
                if member in package.namelist():
                    reader.warn(f"{label}未加入正文，需结合原件核对")
            markdown = output / f"{source.stem}.md"
            markdown.write_text(content + "\n", encoding="utf-8")
            if reader.equations:
                (output / "equations.json").write_text(json.dumps(reader.equations, ensure_ascii=False, indent=2), encoding="utf-8")
    except (zipfile.BadZipFile, KeyError, ET.ParseError) as error:
        raise ValueError("DOCX损坏或缺少必需的OOXML内容") from error
    state["md_path"] = str(markdown)
    return record_parse_result(state, "docx_ooxml_v1", table_count=reader.table_count,
                               image_count=len(reader.images), equation_count=len(reader.equations),
                               warnings=reader.warnings)
