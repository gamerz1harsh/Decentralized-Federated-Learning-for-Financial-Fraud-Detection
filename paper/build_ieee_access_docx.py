"""Build the manuscript in the supplied IEEE Access Word template."""
from __future__ import annotations

import argparse
from copy import deepcopy
from io import BytesIO
from pathlib import Path
import re
import tempfile

import pypandoc
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from docx.table import Table
from docx.text.paragraph import Paragraph


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "results" / "benchmark_pilot_5seed_5round_20261106" / "figures"
AUTHORS = "Harsh Pathak, Siddhant Rathi, Aryan Kumar, and Shaurya Singh Chauhan"
CITATION_ORDER = (
    "saito2015",
    "mcmahan2017",
    "yin2018",
    "blanchard2017",
    "li2020",
    "karimireddy2020",
    "wang2020",
    "ghorbani2019",
    "creditcarddataset",
    "miadul_dataset",
)


def build_latex_source(source_path: Path) -> tuple[str, str, str]:
    source_tex = source_path.read_text(encoding="utf-8")
    abstract_match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}", source_tex, re.S)
    keywords_match = re.search(
        r"\\begin\{keywords\}(.*?)\\end\{keywords\}", source_tex, re.S)
    title_match = re.search(r"\\title\{(.*?)\}", source_tex, re.S)
    if not all((abstract_match, keywords_match, title_match)):
        raise RuntimeError("Paper title, abstract, or index terms could not be read")

    abstract = " ".join(abstract_match.group(1).split())
    word_count = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", abstract)
    if not 150 <= len(word_count) <= 250:
        raise RuntimeError(
            f"IEEE abstract must be 150-250 words; found {len(word_count)}")

    latex = source_tex[source_tex.index(r"\begin{document}"):]
    latex = latex.replace(r"\caption{\resultcaption{", r"\caption{")
    latex = latex.replace(
        r"\caption{\architecturecaption{", r"\caption{")
    latex = re.sub(r"}}(?=\n\\label\{fig:)", "}", latex)
    latex = re.sub(
        r"\\history\{.*?\}|\\doi\{.*?\}|\\titlepgskip=.*?(?:\n|$)",
        "",
        latex,
        flags=re.S,
    )
    latex = re.sub(
        r"^\\(?:address|corresp)\b.*(?:\n|$)",
        "",
        latex,
        flags=re.M,
    )
    latex = re.sub(
        r"\\author\{.*?(?=\\maketitle)",
        rf"\\author{{{AUTHORS}}}",
        latex,
        count=1,
        flags=re.S,
    )
    latex = re.sub(r"\\uppercase\{([^{}]+)\}", r"\1", latex)
    latex = re.sub(r"\\authorrefmark\{\d+\}", "", latex)

    citation_numbers = {
        key: index for index, key in enumerate(CITATION_ORDER, start=1)
    }

    def numeric_citation(match: re.Match[str]) -> str:
        keys = [key.strip() for key in match.group(1).split(",")]
        return "[" + ", ".join(str(citation_numbers[key]) for key in keys) + "]"

    latex = re.sub(r"\\cite\{([^}]+)\}", numeric_citation, latex)
    latex = latex.replace(
        r"\begin{thebibliography}{00}", r"\section*{References}")
    latex = latex.replace(r"\end{thebibliography}", "")
    latex = re.sub(
        r"\\bibitem\{([^}]+)\}",
        lambda match: r"\par [" + str(citation_numbers[match.group(1)]) + "] ",
        latex,
    )
    latex = latex.replace(
        r"\begin{keywords}", r"\section*{Index Terms}")
    latex = latex.replace(r"\end{keywords}", "")

    figure_names = re.findall(r"\\benchmarkfigure\{([^}]+)\}", latex)
    if len(figure_names) < 7 or "architecture.pdf" not in figure_names:
        raise RuntimeError(
            "Expected the architecture figure and at least six result figures; "
            f"found {len(figure_names)}")

    def local_figure(match: re.Match[str]) -> str:
        name = Path(match.group(1)).with_suffix(".png").name
        image_path = FIGURES / name
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        return (
            r"\includegraphics[width=3.1in]{"
            + image_path.relative_to(ROOT).as_posix()
            + "}"
        )

    latex = re.sub(
        r"\\benchmarkfigure\{([^}]+)\}\{[^}]+\}",
        local_figure,
        latex,
    )
    return latex, " ".join(title_match.group(1).split()), abstract


def replace_front_matter(template: Document, title: str, abstract: str) -> None:
    front = template.paragraphs[:12]
    if (
        len(front) != 12
        or front[11]._p.pPr is None
        or not front[11]._p.pPr.xpath("./w:sectPr")
    ):
        raise RuntimeError(
            "The supplied IEEE template front-page section break was not found")

    front[2].text = title
    front[2].style = template.styles["Paper Title"]
    front[3].text = AUTHORS
    front[3].style = template.styles["AU"]
    front[4].text = (
        "Student, VIT-AP University, Amaravati, Andhra Pradesh, India")
    front[4].style = template.styles["PI_No Space"]
    front[5].text = ""
    front[6].text = ""
    front[7].text = (
        "Corresponding author: Harsh Pathak "
        "(pathakharsh2004@gmail.com).")
    front[7].style = template.styles["PI"]
    front[8].text = "This research received no external funding."
    front[8].style = template.styles["footnote text"]
    front[9].text = "ABSTRACT: " + abstract
    front[9].style = template.styles["Abstract"]
    front[10].text = (
        "INDEX TERMS: client contribution, federated learning, fraud "
        "detection, heterogeneous features, imbalanced classification, "
        "robust aggregation")
    front[10].style = template.styles["IT"]

    for paragraph in (front[0], front[1]):
        paragraph._element.getparent().remove(paragraph._element)

    body = template.element.body
    reached_break = False
    for child in list(body):
        if child is front[11]._p:
            reached_break = True
            continue
        if reached_break and child.tag != qn("w:sectPr"):
            body.remove(child)


def copy_relationships(element, source_part, target_part, relationship_map) -> None:
    for node in element.iter():
        for attribute in (qn("r:id"), qn("r:embed"), qn("r:link")):
            old_id = node.get(attribute)
            if not old_id:
                continue
            if old_id not in relationship_map:
                relationship = source_part.rels[old_id]
                if relationship.reltype.endswith("/image"):
                    new_id, _ = target_part.get_or_add_image(
                        BytesIO(relationship.target_part.blob))
                elif relationship.is_external:
                    new_id = target_part.relate_to(
                        relationship.target_ref,
                        relationship.reltype,
                        is_external=True,
                    )
                else:
                    raise RuntimeError(
                        "Unsupported internal relationship in manuscript: "
                        + relationship.reltype)
                relationship_map[old_id] = new_id
            node.set(attribute, relationship_map[old_id])


def section_break(properties, break_type=None):
    paragraph = OxmlElement("w:p")
    paragraph_properties = OxmlElement("w:pPr")
    section_properties = deepcopy(properties)
    if break_type is not None:
        section_type = section_properties.find(qn("w:type"))
        if section_type is None:
            section_type = OxmlElement("w:type")
            section_properties.insert(0, section_type)
        section_type.set(qn("w:val"), break_type)
    paragraph_properties.append(section_properties)
    paragraph.append(paragraph_properties)
    return paragraph


def set_page_number_footer(footer) -> None:
    paragraph = footer.paragraphs[0]
    paragraph.clear()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    for field_type, text in (
        ("begin", None),
        (None, " PAGE "),
        ("separate", None),
        (None, "1"),
        ("end", None),
    ):
        if field_type is None:
            element = OxmlElement("w:instrText" if text == " PAGE " else "w:t")
            element.text = text
            if element.tag == qn("w:instrText"):
                element.set(qn("xml:space"), "preserve")
        else:
            element = OxmlElement("w:fldChar")
            element.set(qn("w:fldCharType"), field_type)
        run._r.append(element)


def style_content(element, document: Document) -> None:
    if element.tag == qn("w:p"):
        paragraph = Paragraph(element, document)
        style_name = paragraph.style.name if paragraph.style else "Normal"
        if style_name == "Heading 1":
            paragraph.style = document.styles["H1"]
            paragraph.text = paragraph.text.upper()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif style_name == "Heading 2":
            paragraph.style = document.styles["H2_First"]
        elif style_name == "Caption":
            previous = element.getprevious()
            follows_image = (
                previous is not None
                and bool(previous.xpath(".//w:drawing"))
            )
            paragraph.style = document.styles[
                "Fig Caption" if follows_image else "Table Caption"]
        elif style_name in ("Normal", "Body Text"):
            paragraph.style = document.styles["PARA"]

        if element.xpath(".//m:oMathPara"):
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.keep_together = True
            paragraph.paragraph_format.space_before = Pt(2)
            paragraph.paragraph_format.space_after = Pt(2)

        if paragraph.text.strip().upper() == "REFERENCES":
            paragraph.paragraph_format.page_break_before = True

        if re.match(r"^\[\d+\]", paragraph.text):
            paragraph.style = document.styles["Normal"]
            paragraph.paragraph_format.left_indent = Inches(0.18)
            paragraph.paragraph_format.first_line_indent = Inches(-0.18)
            paragraph.paragraph_format.space_after = Pt(2)
            for run in paragraph.runs:
                run.font.name = "Times New Roman"
                run.font.size = Pt(8)

        if element.xpath(".//w:drawing"):
            paragraph.style = document.styles["Normal"]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.line_spacing = 1.0
            paragraph.paragraph_format.space_before = Pt(0)
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.keep_with_next = True
            for extent in element.xpath(".//wp:extent"):
                new_width = int(Inches(3.1))
                old_width = max(int(extent.get("cx", "1")), 1)
                old_height = int(extent.get("cy", "1"))
                extent.set("cx", str(new_width))
                extent.set("cy", str(max(int(old_height * new_width / old_width), 1)))

    elif element.tag == qn("w:tbl"):
        table = Table(element, document)
        table.autofit = True
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    paragraph.style = document.styles["PARA"]
                    paragraph.paragraph_format.space_after = Pt(1)
                    for run in paragraph.runs:
                        run.font.size = Pt(8)


def add_article_body(template: Document, converted: Document) -> None:
    introduction = None
    for child in converted.element.body:
        if child.tag != qn("w:p"):
            continue
        if Paragraph(child, converted).text.strip().upper() == "INTRODUCTION":
            introduction = child
            break
    if introduction is None:
        raise RuntimeError("Could not locate Introduction in the converted paper")

    body = template.element.body
    final_section = body.find(qn("w:sectPr"))
    if final_section is None:
        raise RuntimeError("IEEE template section properties are missing")
    two_columns = deepcopy(final_section)
    body.remove(final_section)
    columns = two_columns.find(qn("w:cols"))
    if columns is None:
        columns = OxmlElement("w:cols")
        two_columns.append(columns)
    columns.set(qn("w:num"), "2")
    one_column = deepcopy(two_columns)
    one_column.find(qn("w:cols")).set(qn("w:num"), "1")

    children = list(converted.element.body)
    start = children.index(introduction)
    relationship_map = {}
    source_part = converted.part
    target_part = template.part
    in_figure = False

    for child in children[start:]:
        if child.tag == qn("w:sectPr"):
            continue
        copied = deepcopy(child)
        copy_relationships(
            copied, source_part, target_part, relationship_map)
        is_figure = (
            copied.tag == qn("w:p")
            and bool(copied.xpath(".//w:drawing")))
        is_table = copied.tag == qn("w:tbl")
        is_figure_caption = (
            in_figure
            and child.tag == qn("w:p")
            and bool(Paragraph(child, converted).text.strip())
        )

        if is_table:
            body.append(section_break(two_columns))
        style_content(copied, template)
        if is_figure_caption:
            Paragraph(copied, template).style = template.styles["Fig Caption"]
            Paragraph(copied, template).paragraph_format.keep_together = True
        body.append(copied)

        if is_figure_caption:
            in_figure = False
        elif is_table:
            body.append(section_break(one_column))
        elif is_figure:
            in_figure = True

    body.append(two_columns)

    updated_footers = set()
    for section in template.sections:
        footer = section.footer
        footer_part = footer.part
        if footer_part.partname not in updated_footers:
            set_page_number_footer(footer)
            updated_footers.add(footer_part.partname)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template", type=Path,
        default=Path.home() / "Downloads" / "Access-Template-2024.docx")
    parser.add_argument(
        "--source", type=Path,
        default=ROOT / "paper" / "ieee_access_final_draft_backup.tex")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "paper" / "ieee_access_final.docx")
    args = parser.parse_args()
    if not args.template.is_file():
        raise FileNotFoundError(args.template)
    if not args.source.is_file():
        raise FileNotFoundError(args.source)

    latex, title, abstract = build_latex_source(args.source)
    with tempfile.TemporaryDirectory(prefix="ieee_access_build_") as folder:
        converted_path = Path(folder) / "converted.docx"
        pypandoc.convert_text(
            latex,
            "docx",
            format="latex",
            outputfile=str(converted_path),
            extra_args=[
                "--reference-doc=" + str(args.template),
                "--resource-path=" + str(ROOT),
            ],
        )
        template = Document(args.template)
        converted = Document(converted_path)
        replace_front_matter(template, title, abstract)
        add_article_body(template, converted)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        template.save(args.output)

    result = Document(args.output)
    expected = (
        "Harsh Pathak",
        "Siddhant Rathi",
        "Aryan Kumar",
        "Shaurya Singh Chauhan",
        "Student, VIT-AP University",
        "VIT-AP University",
        "pathakharsh2004@gmail.com",
        "no external funding",
        "References",
    )
    content = "\n".join(paragraph.text for paragraph in result.paragraphs)
    missing = [item for item in expected if item.casefold() not in content.casefold()]
    if missing:
        raise RuntimeError("Final Word paper is missing: " + ", ".join(missing))
    expected_figures = len(re.findall(
        r"\\benchmarkfigure\{([^}]+)\}", args.source.read_text(
            encoding="utf-8")))
    if len(result.inline_shapes) != expected_figures:
        raise RuntimeError(
            f"Expected {expected_figures} figures in the Word paper; found "
            f"{len(result.inline_shapes)}")
    print(f"Built {args.output}")
    print(
        f"Size {args.output.stat().st_size} bytes; "
        f"{len(result.inline_shapes)} figures; {len(result.tables)} tables; "
        f"{len(result.sections)} Word sections.")


if __name__ == "__main__":
    main()
