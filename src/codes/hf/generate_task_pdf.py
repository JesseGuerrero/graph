"""Generate a PDF showing all Mind2Web-2 task descriptions from dev and test sets."""

import csv
import textwrap
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER

# -- colours --
HEADER_BG = HexColor("#2C3E50")
HEADER_FG = HexColor("#FFFFFF")
DOMAIN_BG = HexColor("#ECF0F1")
ROW_ALT    = HexColor("#F9F9F9")
ACCENT     = HexColor("#2980B9")

def load_csv(path):
    tasks = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tasks.append(row)
    return tasks

def build_pdf(output_path):
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=18*mm, bottomMargin=18*mm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title2", parent=styles["Title"],
        fontSize=22, leading=26, spaceAfter=4,
        textColor=HEADER_BG,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle2", parent=styles["Normal"],
        fontSize=11, leading=14, spaceAfter=14,
        textColor=HexColor("#7F8C8D"), alignment=TA_CENTER,
    )
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading1"],
        fontSize=16, leading=20, spaceBefore=16, spaceAfter=8,
        textColor=ACCENT,
    )
    domain_style = ParagraphStyle(
        "Domain", parent=styles["Heading2"],
        fontSize=13, leading=16, spaceBefore=12, spaceAfter=4,
        textColor=HEADER_BG,
    )
    task_id_style = ParagraphStyle(
        "TaskID", parent=styles["Normal"],
        fontSize=9, leading=11, textColor=ACCENT,
        fontName="Helvetica-Bold",
    )
    task_desc_style = ParagraphStyle(
        "TaskDesc", parent=styles["Normal"],
        fontSize=9, leading=13, textColor=HexColor("#2C3E50"),
    )
    subdomain_style = ParagraphStyle(
        "Subdomain", parent=styles["Normal"],
        fontSize=8, leading=10, textColor=HexColor("#7F8C8D"),
        fontName="Helvetica-Oblique",
    )

    elements = []

    # Title page content
    elements.append(Spacer(1, 40*mm))
    elements.append(Paragraph("Mind2Web-2", title_style))
    elements.append(Paragraph("Task Descriptions", ParagraphStyle(
        "Title3", parent=title_style, fontSize=18, leading=22, spaceAfter=8,
    )))
    elements.append(Spacer(1, 6*mm))
    elements.append(Paragraph(
        "A comprehensive listing of all deep research task descriptions<br/>"
        "from the Mind2Web-2 benchmark dataset.",
        subtitle_style,
    ))

    # Load data
    dev_tasks = load_csv("hf/Mind2Web-2/dev_set.csv")
    test_tasks = load_csv("hf/Mind2Web-2/test_set.csv")

    elements.append(Spacer(1, 10*mm))
    stats_data = [
        ["", "Tasks", "Domains"],
        ["Dev Set", str(len(dev_tasks)),
         str(len({t["domain"] for t in dev_tasks}))],
        ["Test Set", str(len(test_tasks)),
         str(len({t["domain"] for t in test_tasks}))],
        ["Total", str(len(dev_tasks) + len(test_tasks)),
         str(len({t["domain"] for t in dev_tasks + test_tasks}))],
    ]
    stats_table = Table(stats_data, colWidths=[50*mm, 30*mm, 30*mm])
    stats_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), HEADER_FG),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("BACKGROUND", (0, -1), (-1, -1), DOMAIN_BG),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#BDC3C7")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2), [HexColor("#FFFFFF"), ROW_ALT]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(stats_table)
    elements.append(PageBreak())

    def add_task_section(tasks, section_title, section_subtitle):
        elements.append(Paragraph(section_title, section_style))
        elements.append(Paragraph(section_subtitle, subtitle_style))
        elements.append(Spacer(1, 4*mm))

        # Group by domain
        by_domain = {}
        for t in tasks:
            by_domain.setdefault(t["domain"], []).append(t)

        for domain in sorted(by_domain.keys()):
            domain_tasks = sorted(by_domain[domain], key=lambda t: t["task_id"])
            elements.append(Paragraph(
                f"{domain} ({len(domain_tasks)} tasks)", domain_style
            ))

            for i, t in enumerate(domain_tasks):
                desc = t["task_description"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                # Convert newlines to <br/>
                desc = desc.replace("\n", "<br/>")

                task_block = KeepTogether([
                    Spacer(1, 2*mm),
                    Paragraph(
                        f'<font color="{ACCENT.hexval()}">{i+1}.</font> '
                        f'<b>{t["task_id"]}</b>'
                        f'  <font color="#7F8C8D" size="7">({t["subdomain"]})</font>',
                        task_id_style,
                    ),
                    Spacer(1, 1*mm),
                    Paragraph(desc, task_desc_style),
                    Spacer(1, 3*mm),
                ])
                elements.append(task_block)

    # Dev set
    add_task_section(
        dev_tasks,
        f"Development Set ({len(dev_tasks)} tasks)",
        "Tasks used for development and validation",
    )
    elements.append(PageBreak())

    # Test set
    add_task_section(
        test_tasks,
        f"Test Set ({len(test_tasks)} tasks)",
        "Full evaluation benchmark from HuggingFace",
    )

    doc.build(elements)
    print(f"PDF generated: {output_path}")
    print(f"  Dev tasks:  {len(dev_tasks)}")
    print(f"  Test tasks: {len(test_tasks)}")

if __name__ == "__main__":
    build_pdf("hf/Mind2Web-2_Task_Descriptions.pdf")
