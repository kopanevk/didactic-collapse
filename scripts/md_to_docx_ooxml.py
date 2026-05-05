from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


INPUT_MD = Path('outputs/article_manuscript_clean/manuscript_ru_clean.md')
OUTPUT_DOCX = Path('outputs/article_manuscript_clean/manuscript_ru_clean.docx')


@dataclass
class TableBlock:
    headers: list[str]
    rows: list[list[str]]


def _run_xml(text: str, kind: str, bold: bool = False) -> str:
    text_xml = escape(text)
    space_attr = ' xml:space="preserve"' if (text[:1].isspace() or text[-1:].isspace()) else ''
    if kind == 'title':
        size = 34
        default_bold = True
    elif kind == 'h1':
        size = 30
        default_bold = True
    elif kind == 'h2':
        size = 27
        default_bold = True
    elif kind == 'h3':
        size = 25
        default_bold = True
    elif kind == 'caption':
        size = 23
        default_bold = True
    else:
        size = 24
        default_bold = False

    use_bold = bold or default_bold
    b_xml = '<w:b/>' if use_bold else ''
    rpr = f'<w:rPr>{b_xml}<w:sz w:val="{size}"/></w:rPr>'
    return f'<w:r>{rpr}<w:t{space_attr}>{text_xml}</w:t></w:r>'


def _paragraph_xml(text: str, kind: str = 'body') -> str:
    if text == '':
        return '<w:p/>'
    if kind == 'title':
        ppr = '<w:pPr><w:jc w:val="center"/><w:spacing w:before="240" w:after="240"/></w:pPr>'
    elif kind == 'h1':
        ppr = '<w:pPr><w:spacing w:before="220" w:after="120"/></w:pPr>'
    elif kind == 'h2':
        ppr = '<w:pPr><w:spacing w:before="180" w:after="100"/></w:pPr>'
    elif kind == 'h3':
        ppr = '<w:pPr><w:spacing w:before="140" w:after="80"/></w:pPr>'
    elif kind == 'caption':
        ppr = '<w:pPr><w:spacing w:before="100" w:after="80"/></w:pPr>'
    elif kind == 'bullet':
        ppr = '<w:pPr><w:ind w:left="720" w:hanging="360"/><w:spacing w:after="70"/></w:pPr>'
        text = f'• {text}'
    else:
        ppr = '<w:pPr><w:spacing w:after="80"/></w:pPr>'
    return f'<w:p>{ppr}{_run_xml(text, kind)}</w:p>'


def _table_cell_xml(text: str, width: int, is_header: bool = False) -> str:
    shading = '<w:shd w:val="clear" w:color="auto" w:fill="F2F2F2"/>' if is_header else ''
    tcpr = (
        '<w:tcPr>'
        f'<w:tcW w:w="{width}" w:type="dxa"/>'
        '<w:vAlign w:val="center"/>'
        f'{shading}'
        '</w:tcPr>'
    )
    run = _run_xml(text.strip(), 'body', bold=is_header)
    p = '<w:p><w:pPr><w:spacing w:before="40" w:after="40"/></w:pPr>' + run + '</w:p>'
    return f'<w:tc>{tcpr}{p}</w:tc>'


def _table_xml(tbl: TableBlock) -> str:
    # approximate text width: 9026 dxa on A4 with 1in margins
    total_width = 9026
    num_cols = max(1, len(tbl.headers))
    col_width = int(total_width / num_cols)

    tbl_pr = (
        '<w:tblPr>'
        '<w:tblStyle w:val="TableGrid"/>'
        '<w:tblW w:w="9026" w:type="dxa"/>'
        '<w:tblBorders>'
        '<w:top w:val="single" w:sz="8" w:space="0" w:color="808080"/>'
        '<w:left w:val="single" w:sz="8" w:space="0" w:color="808080"/>'
        '<w:bottom w:val="single" w:sz="8" w:space="0" w:color="808080"/>'
        '<w:right w:val="single" w:sz="8" w:space="0" w:color="808080"/>'
        '<w:insideH w:val="single" w:sz="6" w:space="0" w:color="B0B0B0"/>'
        '<w:insideV w:val="single" w:sz="6" w:space="0" w:color="B0B0B0"/>'
        '</w:tblBorders>'
        '<w:tblCellMar>'
        '<w:top w:w="80" w:type="dxa"/>'
        '<w:left w:w="120" w:type="dxa"/>'
        '<w:bottom w:w="80" w:type="dxa"/>'
        '<w:right w:w="120" w:type="dxa"/>'
        '</w:tblCellMar>'
        '</w:tblPr>'
    )

    grid = '<w:tblGrid>' + ''.join(f'<w:gridCol w:w="{col_width}"/>' for _ in range(num_cols)) + '</w:tblGrid>'

    header_cells = ''.join(_table_cell_xml(c, col_width, is_header=True) for c in tbl.headers)
    header_row = '<w:tr>' + header_cells + '</w:tr>'

    row_xml = []
    for row in tbl.rows:
        padded = row + [''] * max(0, num_cols - len(row))
        cells = ''.join(_table_cell_xml(c, col_width, is_header=False) for c in padded[:num_cols])
        row_xml.append('<w:tr>' + cells + '</w:tr>')

    return '<w:tbl>' + tbl_pr + grid + header_row + ''.join(row_xml) + '</w:tbl>'


def _is_table_sep(line: str) -> bool:
    s = line.strip()
    if '|' not in s:
        return False
    no_pipes = s.replace('|', '').replace(':', '').replace('-', '').replace(' ', '')
    return no_pipes == '' and '-' in s


def _split_md_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith('|'):
        s = s[1:]
    if s.endswith('|'):
        s = s[:-1]
    return [c.strip() for c in s.split('|')]


def parse_markdown(md: str):
    lines = md.splitlines()
    i = 0
    blocks = []
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped == '':
            blocks.append(('blank', ''))
            i += 1
            continue

        # table block
        if '|' in line and i + 1 < len(lines) and _is_table_sep(lines[i + 1]):
            headers = _split_md_row(line)
            i += 2
            rows: list[list[str]] = []
            while i < len(lines):
                rline = lines[i]
                if rline.strip() == '' or '|' not in rline:
                    break
                rows.append(_split_md_row(rline))
                i += 1
            blocks.append(('table', TableBlock(headers=headers, rows=rows)))
            continue

        # headings
        if stripped.startswith('# '):
            blocks.append(('title', stripped[2:].strip()))
            i += 1
            continue
        if stripped.startswith('## '):
            blocks.append(('h1', stripped[3:].strip()))
            i += 1
            continue
        if stripped.startswith('### '):
            blocks.append(('h2', stripped[4:].strip()))
            i += 1
            continue
        if stripped.startswith('#### '):
            blocks.append(('h3', stripped[5:].strip()))
            i += 1
            continue

        # bullet
        if stripped.startswith('- '):
            blocks.append(('bullet', stripped[2:].strip()))
            i += 1
            continue

        # strip markdown bold wrappers for full line
        if stripped.startswith('**') and stripped.endswith('**') and len(stripped) > 4:
            stripped = stripped[2:-2].strip()

        # caption-like lines
        if re.match(r'^Таблица\s+\d+\.', stripped):
            blocks.append(('caption', stripped))
            i += 1
            continue

        # paragraph, de-mark inline code and bold markers
        text = line.replace('`', '')
        text = text.replace('**', '')
        blocks.append(('body', text.strip()))
        i += 1

    return blocks


def build_docx_from_markdown(input_md: Path, output_docx: Path) -> Path:
    md = input_md.read_text(encoding='utf-8')
    blocks = parse_markdown(md)

    parts = []
    for kind, payload in blocks:
        if kind == 'table':
            parts.append(_table_xml(payload))
            parts.append(_paragraph_xml('', 'body'))
        elif kind == 'blank':
            parts.append(_paragraph_xml('', 'body'))
        else:
            parts.append(_paragraph_xml(payload, kind))

    sect = (
        '<w:sectPr>'
        '<w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" '
        'w:header="708" w:footer="708" w:gutter="0"/>'
        '</w:sectPr>'
    )

    body = ''.join(parts)
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
        'mc:Ignorable="w14 wp14">'
        f'<w:body>{body}{sect}</w:body>'
        '</w:document>'
    )

    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '</Types>'
    )

    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/>'
        '</Relationships>'
    )

    output_docx.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_docx, 'w', compression=ZIP_DEFLATED) as zf:
        zf.writestr('[Content_Types].xml', content_types_xml)
        zf.writestr('_rels/.rels', rels_xml)
        zf.writestr('word/document.xml', document_xml)

    return output_docx


if __name__ == '__main__':
    out = build_docx_from_markdown(INPUT_MD, OUTPUT_DOCX)
    print(out)
