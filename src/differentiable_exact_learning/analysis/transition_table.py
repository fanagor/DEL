"""Parchment transition matrices with typographically distinct state numbers."""

import base64
import csv
from pathlib import Path

BODY = "#FBF7EC"
ALTERNATE = "#F6EFDF"
HEADER = "#E9DDBE"
INITIAL = "#E4EDDA"
FINAL = "#EFDDDA"
INK = "#3F382F"
RULE = "#CFC2A5"
VERTICAL_RULE = "#AD9B78"
MUTED = "#877D6A"


def transition_matrix(machine, expert):
    """Return actual state labels and exact transition triples; None means unknown."""
    return [
        dict(
            state=q,
            role="initial / halt"
            if q == machine.initial_state and q in machine.final_states
            else "initial"
            if q == machine.initial_state
            else "halt"
            if q in machine.final_states
            else "",
            cells=[
                None
                if (q, x) not in machine.transitions
                else (
                    machine.transitions[q, x][0],
                    expert.vocab_rev[machine.transitions[q, x][1]] or "λ",
                    expert.motion_vocab_rev[machine.transitions[q, x][2]],
                )
                for x in machine.alphabet
            ],
        )
        for q in machine.states
    ]


def _fonts():
    import matplotlib
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    directory = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    fonts = {
        "DELState": directory / "DejaVuSansMono.ttf",
        "DELSymbol": directory / "DejaVuSerif.ttf",
        "DELLabel": directory / "DejaVuSans.ttf",
    }
    for name, path in fonts.items():
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(path)))
    return fonts


def _drawing(rows, headers, dimensions, page, total):
    from reportlab.graphics.shapes import Drawing, Line, Rect, String
    from reportlab.lib.colors import HexColor

    state_width, cell_width, row_height, fields = dimensions
    margin, header_height, footer = 7, 24, 33
    width = 2 * margin + state_width + len(headers) * cell_width
    height = 2 * margin + header_height + len(rows) * row_height + footer
    drawing = Drawing(width, height)
    left, right = margin, width - margin
    top = height - margin
    bottom = top - header_height - len(rows) * row_height

    def text(x, y, value, font="DELLabel", size=9, color=INK, anchor="middle"):
        drawing.add(
            String(
                x,
                y,
                str(value),
                fontName=font,
                fontSize=size,
                fillColor=HexColor(color),
                textAnchor=anchor,
            )
        )

    drawing.add(
        Rect(
            left,
            top - header_height,
            right - left,
            header_height,
            fillColor=HexColor(HEADER),
            strokeColor=None,
        )
    )
    text(left + state_width / 2, top - 15, "State", size=9)
    for column, label in enumerate(headers):
        text(
            left + state_width + (column + 0.5) * cell_width,
            top - 16,
            label,
            font="DELSymbol",
            size=11,
        )
    for index, row in enumerate(rows):
        y = top - header_height - (index + 1) * row_height
        fill = (
            INITIAL
            if "initial" in row["role"]
            else FINAL
            if row["role"]
            else (BODY if index % 2 == 0 else ALTERNATE)
        )
        drawing.add(
            Rect(left, y, right - left, row_height, fillColor=HexColor(fill), strokeColor=None)
        )
        baseline = y + row_height / 2 - 3.3
        if row["role"]:
            text(left + 16, baseline, row["state"], font="DELState", size=9.5)
            text(left + 29, baseline + 0.3, row["role"], size=5.6, color=MUTED, anchor="start")
        else:
            text(left + 16, baseline, row["state"], font="DELState", size=9.5)
        for column, cell in enumerate(row["cells"]):
            center = left + state_width + (column + 0.5) * cell_width
            if cell is None:
                text(center, baseline, "-", size=9, color=MUTED)
                continue
            target, symbol, motion = cell
            # Align each field separately, allowing three-digit state IDs.
            tw, sw, mw = fields
            start = center - (tw + sw + mw + 15) / 2
            text(start + tw, baseline, target, font="DELState", size=9, anchor="end")
            text(start + tw + 3.75, baseline, "|", size=8, color=RULE)
            text(start + tw + 7.5 + sw / 2, baseline, symbol, font="DELSymbol", size=9.5)
            text(start + tw + sw + 11.25, baseline, "|", size=8, color=RULE)
            text(start + tw + sw + 15 + mw / 2, baseline, motion, size=8.5)
        drawing.add(Line(left, y, right, y, strokeColor=HexColor(RULE), strokeWidth=0.3))
    for y in (top, top - header_height, bottom):
        drawing.add(Line(left, y, right, y, strokeColor=HexColor("#9C8B69"), strokeWidth=0.65))
    drawing.add(
        Line(
            left + state_width,
            bottom,
            left + state_width,
            top,
            strokeColor=HexColor(VERTICAL_RULE),
            strokeWidth=0.5,
        )
    )
    for column in range(1, len(headers)):
        x = left + state_width + column * cell_width
        drawing.add(Line(x, bottom, x, top, strokeColor=HexColor(VERTICAL_RULE), strokeWidth=0.45))
    text(
        left,
        bottom - 13,
        "Entry: next state | written symbol | motion",
        size=6.8,
        color=MUTED,
        anchor="start",
    )
    text(
        left,
        bottom - 24,
        "λ = blank     - = unobserved     S = stop",
        font="DELSymbol",
        size=7,
        color=MUTED,
        anchor="start",
    )
    if total > 1:
        text(right, bottom - 24, f"{page} / {total}", size=6.8, color=MUTED, anchor="end")
    return drawing


def _latex(rows, headers):
    def escape(value):
        if value == "λ":
            return r"$\lambda$"
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        return "".join(replacements.get(c, c) for c in str(value))

    lines = [
        "% Requires \\usepackage{booktabs,array} and \\usepackage[table]{xcolor}.",
        "% State numbers and transitions match the exported FST exactly.",
        r"\begingroup",
        r"\definecolor{DELBody}{HTML}{FBF7EC}",
        r"\definecolor{DELAlternate}{HTML}{F6EFDF}",
        r"\definecolor{DELHeader}{HTML}{E9DDBE}",
        r"\definecolor{DELInitial}{HTML}{E4EDDA}",
        r"\definecolor{DELFinal}{HTML}{EFDDDA}",
        r"\definecolor{DELRule}{HTML}{CFC2A5}",
        r"\definecolor{DELVerticalRule}{HTML}{AD9B78}",
        r"\setlength{\tabcolsep}{6pt}\renewcommand{\arraystretch}{1.22}\small",
        r"\begin{tabular}{l"
        + r"!{\color{DELVerticalRule}\vrule width 0.45pt}c" * len(headers)
        + "}",
        r"\toprule\rowcolor{DELHeader}",
        "State & " + " & ".join(escape(h) for h in headers) + r" \\",
        r"\midrule",
    ]
    for index, row in enumerate(rows):
        color = (
            "DELInitial"
            if "initial" in row["role"]
            else "DELFinal"
            if row["role"]
            else "DELBody"
            if index % 2 == 0
            else "DELAlternate"
        )
        label = r"\texttt{" + str(row["state"]) + "}"
        if row["role"]:
            label += r"\, {\scriptsize\textsf{" + escape(row["role"]) + "}}"
        cells = []
        for cell in row["cells"]:
            if cell is None:
                cells.append(r"\textcolor{gray}{-}")
            else:
                target, symbol, motion = cell
                separator = r"\,\textcolor{DELRule}{\textbar}\,"
                cells.append(
                    r"\texttt{"
                    + str(target)
                    + "}"
                    + separator
                    + escape(symbol)
                    + separator
                    + r"\textsf{"
                    + escape(motion)
                    + "}"
                )
        lines.append(r"\rowcolor{" + color + "}" + label + " & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\endgroup", ""])
    return "\n".join(lines)


def render_transition_table(machine, expert, prefix, *, rows_per_page=32):
    """Export a state-by-read-symbol table as embedded-font PDF, SVG, LaTeX and CSV.

    State numbers are not changed here. Use extraction.number_states before
    exporting the machine to obtain initial-first, halt-last consecutive IDs.
    PDF/SVG row panels repeat the column headers without reducing text size.
    """
    from reportlab.graphics import renderPDF, renderSVG
    from reportlab.pdfbase.pdfmetrics import stringWidth
    from reportlab.pdfgen import canvas

    if isinstance(rows_per_page, bool) or not isinstance(rows_per_page, int) or rows_per_page < 1:
        raise ValueError("rows_per_page must be a positive integer")
    fonts = _fonts()
    rows = transition_matrix(machine, expert)
    headers = [expert.vocab_rev[x] or "λ" for x in machine.alphabet]
    parts = [cell for row in rows for cell in row["cells"] if cell is not None]
    fields = (
        max([stringWidth(str(q), "DELState", 9) for q in machine.states]),
        max([stringWidth(cell[1], "DELSymbol", 9.5) for cell in parts] or [7]),
        max([stringWidth(cell[2], "DELLabel", 8.5) for cell in parts] or [7]),
    )
    cell_width = max(
        48, sum(fields) + 27, max([stringWidth(h, "DELSymbol", 11) + 12 for h in headers] or [0])
    )
    state_width = 76 if any(row["role"] == "initial / halt" for row in rows) else 56
    dimensions = state_width, cell_width, 19, fields
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "pdf": prefix.with_suffix(".pdf"),
        "tex": prefix.with_suffix(".tex"),
        "csv": prefix.with_suffix(".csv"),
        "svg": [],
    }
    paths["tex"].write_text(_latex(rows, headers))
    with paths["csv"].open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["state", "role", *headers])
        for row in rows:
            writer.writerow(
                [
                    row["state"],
                    row["role"],
                    *["" if cell is None else " | ".join(map(str, cell)) for cell in row["cells"]],
                ]
            )
    panels = [rows[i : i + rows_per_page] for i in range(0, len(rows), rows_per_page)]
    pdf = canvas.Canvas(str(paths["pdf"]))
    pdf.setTitle("Finite-state controller transition table")
    pdf.setAuthor("DEL")
    font_css = "".join(
        "@font-face{font-family:'"
        + name
        + "';src:url(data:font/ttf;base64,"
        + base64.b64encode(path.read_bytes()).decode()
        + ") format('truetype');}"
        for name, path in fonts.items()
    )
    for page, panel in enumerate(panels, 1):
        drawing = _drawing(panel, headers, dimensions, page, len(panels))
        pdf.setPageSize((drawing.width, drawing.height))
        renderPDF.draw(drawing, pdf, 0, 0)
        pdf.showPage()
        svg_path = (
            prefix.with_suffix(".svg")
            if len(panels) == 1
            else prefix.with_name(f"{prefix.name}_{page:02d}").with_suffix(".svg")
        )
        svg = renderSVG.drawToString(drawing)
        if isinstance(svg, bytes):
            svg = svg.decode()
        start = svg.index(">", svg.index("<svg")) + 1
        svg = (
            svg[:start]
            + '<defs><style type="text/css"><![CDATA['
            + font_css
            + "]]></style></defs>"
            + svg[start:]
        )
        svg_path.write_text(svg)
        paths["svg"].append(svg_path)
    pdf.save()
    return paths
