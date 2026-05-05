from __future__ import annotations

import re
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


INPUT_TXT = Path("outputs/article_text_source_ru.txt")
OUTPUT_DOCX = Path("outputs/didactic_collapse_article_ru.docx")


def _run_xml(text: str, kind: str) -> str:
    text_xml = escape(text)
    space_attr = ' xml:space="preserve"' if text[:1].isspace() or text[-1:].isspace() else ""
    if kind == "title":
        rpr = "<w:rPr><w:b/><w:sz w:val=\"34\"/></w:rPr>"
    elif kind == "h1":
        rpr = "<w:rPr><w:b/><w:sz w:val=\"30\"/></w:rPr>"
    elif kind == "h2":
        rpr = "<w:rPr><w:b/><w:sz w:val=\"27\"/></w:rPr>"
    else:
        rpr = "<w:rPr><w:sz w:val=\"24\"/></w:rPr>"
    return f"<w:r>{rpr}<w:t{space_attr}>{text_xml}</w:t></w:r>"


def _paragraph_xml(text: str, kind: str) -> str:
    if text == "":
        return "<w:p/>"
    if kind == "title":
        ppr = (
            "<w:pPr>"
            "<w:jc w:val=\"center\"/>"
            "<w:spacing w:before=\"240\" w:after=\"240\"/>"
            "</w:pPr>"
        )
    elif kind == "h1":
        ppr = "<w:pPr><w:spacing w:before=\"240\" w:after=\"120\"/></w:pPr>"
    elif kind == "h2":
        ppr = "<w:pPr><w:spacing w:before=\"180\" w:after=\"90\"/></w:pPr>"
    else:
        ppr = "<w:pPr><w:spacing w:after=\"80\"/></w:pPr>"
    return f"<w:p>{ppr}{_run_xml(text, kind)}</w:p>"


def _kind_for_line(line: str, idx: int) -> str:
    s = line.strip()
    if idx == 0:
        return "title"
    if s == "":
        return "body"
    if s == "Аннотация":
        return "h1"
    if re.match(r"^\d+\.\d+\s", s):
        return "h2"
    if re.match(r"^\d+\.\s", s):
        return "h1"
    return "body"


def build_docx(input_txt: Path, output_docx: Path) -> Path:
    lines = input_txt.read_text(encoding="utf-8").splitlines()
    paragraphs = []
    for i, line in enumerate(lines):
        kind = _kind_for_line(line, i)
        paragraphs.append(_paragraph_xml(line, kind))

    body = "".join(paragraphs)
    sect = (
        "<w:sectPr>"
        "<w:pgSz w:w=\"11906\" w:h=\"16838\"/>"
        "<w:pgMar w:top=\"1440\" w:right=\"1440\" w:bottom=\"1440\" w:left=\"1440\" "
        "w:header=\"708\" w:footer=\"708\" w:gutter=\"0\"/>"
        "</w:sectPr>"
    )
    document_xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<w:document xmlns:wpc=\"http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas\" "
        "xmlns:mc=\"http://schemas.openxmlformats.org/markup-compatibility/2006\" "
        "xmlns:o=\"urn:schemas-microsoft-com:office:office\" "
        "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" "
        "xmlns:m=\"http://schemas.openxmlformats.org/officeDocument/2006/math\" "
        "xmlns:v=\"urn:schemas-microsoft-com:vml\" "
        "xmlns:wp14=\"http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing\" "
        "xmlns:wp=\"http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing\" "
        "xmlns:w10=\"urn:schemas-microsoft-com:office:word\" "
        "xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\" "
        "xmlns:w14=\"http://schemas.microsoft.com/office/word/2010/wordml\" "
        "xmlns:wpg=\"http://schemas.microsoft.com/office/word/2010/wordprocessingGroup\" "
        "xmlns:wpi=\"http://schemas.microsoft.com/office/word/2010/wordprocessingInk\" "
        "xmlns:wne=\"http://schemas.microsoft.com/office/word/2006/wordml\" "
        "xmlns:wps=\"http://schemas.microsoft.com/office/word/2010/wordprocessingShape\" "
        "mc:Ignorable=\"w14 wp14\">"
        f"<w:body>{body}{sect}</w:body>"
        "</w:document>"
    )

    content_types_xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
        "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
        "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
        "<Override PartName=\"/word/document.xml\" "
        "ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/>"
        "</Types>"
    )

    rels_xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
        "<Relationship Id=\"rId1\" "
        "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" "
        "Target=\"word/document.xml\"/>"
        "</Relationships>"
    )

    output_docx.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_docx, "w", compression=ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("word/document.xml", document_xml)
    return output_docx


if __name__ == "__main__":
    out = build_docx(INPUT_TXT, OUTPUT_DOCX)
    print(out)
