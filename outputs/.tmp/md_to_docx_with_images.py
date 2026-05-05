from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
import sys
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image


INPUT_MD = Path("outputs/article_manuscript_prefinal/manuscript_ru_prefinal.md")
OUTPUT_DOCX = Path("outputs/article_manuscript_prefinal/manuscript_ru_prefinal.docx")


@dataclass
class TableBlock:
    headers: list[str]
    rows: list[list[str]]


@dataclass
class ImageBlock:
    alt_text: str
    src: str


@dataclass
class ImagePart:
    rel_id: str
    part_name: str
    target: str
    bytes_data: bytes
    cx: int
    cy: int
    name: str


def _run_xml(text: str, kind: str, bold: bool = False) -> str:
    text_xml = escape(text)
    space_attr = ' xml:space="preserve"' if (text[:1].isspace() or text[-1:].isspace()) else ""
    if kind == "title":
        size = 34
        default_bold = True
    elif kind == "h1":
        size = 30
        default_bold = True
    elif kind == "h2":
        size = 27
        default_bold = True
    elif kind == "h3":
        size = 25
        default_bold = True
    elif kind == "caption":
        size = 23
        default_bold = True
    else:
        size = 24
        default_bold = False

    use_bold = bold or default_bold
    b_xml = "<w:b/>" if use_bold else ""
    rpr = f'<w:rPr>{b_xml}<w:sz w:val="{size}"/></w:rPr>'
    return f"<w:r>{rpr}<w:t{space_attr}>{text_xml}</w:t></w:r>"


def _paragraph_xml(text: str, kind: str = "body") -> str:
    if text == "":
        return "<w:p/>"
    if kind == "title":
        ppr = '<w:pPr><w:jc w:val="center"/><w:spacing w:before="240" w:after="240"/></w:pPr>'
    elif kind == "h1":
        ppr = '<w:pPr><w:spacing w:before="220" w:after="120"/></w:pPr>'
    elif kind == "h2":
        ppr = '<w:pPr><w:spacing w:before="180" w:after="100"/></w:pPr>'
    elif kind == "h3":
        ppr = '<w:pPr><w:spacing w:before="140" w:after="80"/></w:pPr>'
    elif kind == "caption":
        ppr = '<w:pPr><w:spacing w:before="100" w:after="80"/></w:pPr>'
    elif kind == "bullet":
        ppr = '<w:pPr><w:ind w:left="720" w:hanging="360"/><w:spacing w:after="70"/></w:pPr>'
        text = f"• {text}"
    else:
        ppr = '<w:pPr><w:spacing w:after="80"/></w:pPr>'
    return f"<w:p>{ppr}{_run_xml(text, kind)}</w:p>"


def _table_cell_xml(text: str, width: int, is_header: bool = False) -> str:
    shading = '<w:shd w:val="clear" w:color="auto" w:fill="F2F2F2"/>' if is_header else ""
    tcpr = (
        "<w:tcPr>"
        f'<w:tcW w:w="{width}" w:type="dxa"/>'
        '<w:vAlign w:val="center"/>'
        f"{shading}"
        "</w:tcPr>"
    )
    run = _run_xml(text.strip(), "body", bold=is_header)
    p = '<w:p><w:pPr><w:spacing w:before="40" w:after="40"/></w:pPr>' + run + "</w:p>"
    return f"<w:tc>{tcpr}{p}</w:tc>"


def _table_xml(tbl: TableBlock) -> str:
    total_width = 9026
    num_cols = max(1, len(tbl.headers))
    col_width = int(total_width / num_cols)

    tbl_pr = (
        "<w:tblPr>"
        '<w:tblStyle w:val="TableGrid"/>'
        '<w:tblW w:w="9026" w:type="dxa"/>'
        "<w:tblBorders>"
        '<w:top w:val="single" w:sz="8" w:space="0" w:color="808080"/>'
        '<w:left w:val="single" w:sz="8" w:space="0" w:color="808080"/>'
        '<w:bottom w:val="single" w:sz="8" w:space="0" w:color="808080"/>'
        '<w:right w:val="single" w:sz="8" w:space="0" w:color="808080"/>'
        '<w:insideH w:val="single" w:sz="6" w:space="0" w:color="B0B0B0"/>'
        '<w:insideV w:val="single" w:sz="6" w:space="0" w:color="B0B0B0"/>'
        "</w:tblBorders>"
        "<w:tblCellMar>"
        '<w:top w:w="80" w:type="dxa"/>'
        '<w:left w:w="120" w:type="dxa"/>'
        '<w:bottom w:w="80" w:type="dxa"/>'
        '<w:right w:w="120" w:type="dxa"/>'
        "</w:tblCellMar>"
        "</w:tblPr>"
    )

    grid = "<w:tblGrid>" + "".join(f'<w:gridCol w:w="{col_width}"/>' for _ in range(num_cols)) + "</w:tblGrid>"

    header_cells = "".join(_table_cell_xml(c, col_width, is_header=True) for c in tbl.headers)
    header_row = "<w:tr>" + header_cells + "</w:tr>"

    row_xml = []
    for row in tbl.rows:
        padded = row + [""] * max(0, num_cols - len(row))
        cells = "".join(_table_cell_xml(c, col_width, is_header=False) for c in padded[:num_cols])
        row_xml.append("<w:tr>" + cells + "</w:tr>")

    return "<w:tbl>" + tbl_pr + grid + header_row + "".join(row_xml) + "</w:tbl>"


def _is_table_sep(line: str) -> bool:
    s = line.strip()
    if "|" not in s:
        return False
    no_pipes = s.replace("|", "").replace(":", "").replace("-", "").replace(" ", "")
    return no_pipes == "" and "-" in s


def _split_md_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def parse_markdown(md: str):
    lines = md.splitlines()
    i = 0
    blocks = []
    image_pat = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$")
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped == "":
            blocks.append(("blank", ""))
            i += 1
            continue

        # image
        m = image_pat.match(stripped)
        if m:
            blocks.append(("image", ImageBlock(alt_text=m.group(1).strip(), src=m.group(2).strip())))
            i += 1
            continue

        # table block
        if "|" in line and i + 1 < len(lines) and _is_table_sep(lines[i + 1]):
            headers = _split_md_row(line)
            i += 2
            rows: list[list[str]] = []
            while i < len(lines):
                rline = lines[i]
                if rline.strip() == "" or "|" not in rline:
                    break
                rows.append(_split_md_row(rline))
                i += 1
            blocks.append(("table", TableBlock(headers=headers, rows=rows)))
            continue

        if stripped.startswith("# "):
            blocks.append(("title", stripped[2:].strip()))
            i += 1
            continue
        if stripped.startswith("## "):
            blocks.append(("h1", stripped[3:].strip()))
            i += 1
            continue
        if stripped.startswith("### "):
            blocks.append(("h2", stripped[4:].strip()))
            i += 1
            continue
        if stripped.startswith("#### "):
            blocks.append(("h3", stripped[5:].strip()))
            i += 1
            continue

        if stripped.startswith("- "):
            blocks.append(("bullet", stripped[2:].strip()))
            i += 1
            continue

        if stripped.startswith("**") and stripped.endswith("**") and len(stripped) > 4:
            stripped = stripped[2:-2].strip()

        if re.match(r"^Таблица\s+\d+\.", stripped):
            blocks.append(("caption", stripped))
            i += 1
            continue

        text = line.replace("`", "").replace("**", "")
        blocks.append(("body", text.strip()))
        i += 1
    return blocks


def _image_para_xml(part: ImagePart, docpr_id: int) -> str:
    drawing = (
        "<w:r><w:drawing>"
        '<wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{part.cx}" cy="{part.cy}"/>'
        '<wp:effectExtent l="0" t="0" r="0" b="0"/>'
        f'<wp:docPr id="{docpr_id}" name="{escape(part.name)}"/>'
        '<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>'
        "<a:graphic>"
        '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        "<pic:pic>"
        f'<pic:nvPicPr><pic:cNvPr id="0" name="{escape(part.name)}"/><pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{part.rel_id}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        "<pic:spPr>"
        f'<a:xfrm><a:off x="0" y="0"/><a:ext cx="{part.cx}" cy="{part.cy}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        "</pic:spPr>"
        "</pic:pic>"
        "</a:graphicData>"
        "</a:graphic>"
        "</wp:inline>"
        "</w:drawing></w:r>"
    )
    ppr = '<w:pPr><w:jc w:val="center"/><w:spacing w:before="80" w:after="80"/></w:pPr>'
    return f"<w:p>{ppr}{drawing}</w:p>"


def _prepare_image_part(abs_path: Path, image_index: int) -> ImagePart:
    ext = abs_path.suffix.lower()
    if ext not in {".png", ".jpg", ".jpeg"}:
        raise ValueError(f"Unsupported image extension: {ext}")
    with Image.open(abs_path) as im:
        w_px, h_px = im.size
    # Assume 96 DPI fallback, clamp to page text width
    max_width_emu = int(6.2 * 914400)  # ~6.2in within A4 text block
    width_emu = int((w_px / 96.0) * 914400)
    height_emu = int((h_px / 96.0) * 914400)
    if width_emu > max_width_emu:
        scale = max_width_emu / width_emu
        width_emu = int(width_emu * scale)
        height_emu = int(height_emu * scale)

    rel_id = f"rIdImg{image_index}"
    part_name = f"image{image_index}{ext}"
    target = f"media/{part_name}"
    bytes_data = abs_path.read_bytes()
    return ImagePart(
        rel_id=rel_id,
        part_name=part_name,
        target=target,
        bytes_data=bytes_data,
        cx=width_emu,
        cy=height_emu,
        name=abs_path.name,
    )


def build_docx_from_markdown(input_md: Path, output_docx: Path) -> Path:
    md = input_md.read_text(encoding="utf-8")
    blocks = parse_markdown(md)

    image_map: dict[Path, ImagePart] = {}
    image_counter = 1
    docpr_counter = 1
    parts: list[str] = []

    for kind, payload in blocks:
        if kind == "table":
            parts.append(_table_xml(payload))
            parts.append(_paragraph_xml("", "body"))
        elif kind == "blank":
            parts.append(_paragraph_xml("", "body"))
        elif kind == "image":
            img: ImageBlock = payload
            src = img.src.strip()
            # strip optional quotes around path
            src = src[1:-1] if (src.startswith('"') and src.endswith('"')) else src
            abs_path = (input_md.parent / src).resolve()
            if not abs_path.exists():
                # keep a visible marker instead of failing silently
                parts.append(_paragraph_xml(f"[IMAGE NOT FOUND: {src}]", "body"))
                continue
            if abs_path not in image_map:
                image_map[abs_path] = _prepare_image_part(abs_path, image_counter)
                image_counter += 1
            parts.append(_image_para_xml(image_map[abs_path], docpr_counter))
            docpr_counter += 1
        else:
            parts.append(_paragraph_xml(payload, kind))

    sect = (
        "<w:sectPr>"
        '<w:footerReference w:type="default" r:id="rIdFooter1"/>'
        '<w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" '
        'w:header="708" w:footer="708" w:gutter="0"/>'
        "</w:sectPr>"
    )

    body = "".join(parts)
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
        'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
        'xmlns:o="urn:schemas-microsoft-com:office:office" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
        'xmlns:v="urn:schemas-microsoft-com:vml" '
        'xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
        'xmlns:w10="urn:schemas-microsoft-com:office:word" '
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
        'xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup" '
        'xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk" '
        'xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml" '
        'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture" '
        'mc:Ignorable="w14 wp14">'
        f"<w:body>{body}{sect}</w:body>"
        "</w:document>"
    )

    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Default Extension="png" ContentType="image/png"/>',
        '<Default Extension="jpg" ContentType="image/jpeg"/>',
        '<Default Extension="jpeg" ContentType="image/jpeg"/>',
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>',
        '<Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>',
        "</Types>",
    ]
    content_types_xml = "".join(content_types)

    root_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/>'
        "</Relationships>"
    )

    doc_rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
        '<Relationship Id="rIdFooter1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/>',
    ]
    for part in image_map.values():
        doc_rels.append(
            f'<Relationship Id="{part.rel_id}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
            f'Target="{part.target}"/>'
        )
    doc_rels.append("</Relationships>")
    doc_rels_xml = "".join(doc_rels)

    footer_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:ftr xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
        'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
        'xmlns:o="urn:schemas-microsoft-com:office:office" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
        'xmlns:v="urn:schemas-microsoft-com:vml" '
        'xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
        'xmlns:w10="urn:schemas-microsoft-com:office:word" '
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
        'xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup" '
        'xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk" '
        'xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml" '
        'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
        'mc:Ignorable="w14 wp14">'
        '<w:p>'
        '<w:pPr><w:jc w:val="center"/></w:pPr>'
        '<w:r><w:rPr><w:sz w:val="20"/></w:rPr><w:t>Страница </w:t></w:r>'
        '<w:fldSimple w:instr=" PAGE "><w:r><w:rPr><w:sz w:val="20"/></w:rPr><w:t>1</w:t></w:r></w:fldSimple>'
        '</w:p>'
        '</w:ftr>'
    )

    output_docx.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_docx, "w", compression=ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", root_rels_xml)
        zf.writestr("word/document.xml", document_xml)
        zf.writestr("word/_rels/document.xml.rels", doc_rels_xml)
        zf.writestr("word/footer1.xml", footer_xml)
        for part in image_map.values():
            zf.writestr(f"word/media/{part.part_name}", part.bytes_data)
    return output_docx


if __name__ == "__main__":
    in_md = Path(sys.argv[1]) if len(sys.argv) > 1 else INPUT_MD
    out_docx = Path(sys.argv[2]) if len(sys.argv) > 2 else OUTPUT_DOCX
    out = build_docx_from_markdown(in_md, out_docx)
    print(out)
