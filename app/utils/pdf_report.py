from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pathlib import Path

# 환경 준비
try:
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    _CID_AVAILABLE = True
except ImportError:
    _CID_AVAILABLE = False

try:
    import reportlab
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


def export_frame_labels_pdf(records: list[dict], pdf_path: Path,
                            cv_text_map: dict[str, str],
                            pred_text_map: dict[str, str],
                            STATIC_DIR: Path):
    """ReportLab로 표 형태 PDF 생성 (A4 가로, 좁은 여백, 한글 폰트 임베드, 자동 줄바꿈)."""
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("reportlab not available")

    # 페이지/여백/폰트
    use_font = _register_korean_font(STATIC_DIR)
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=landscape(A4),
        leftMargin=14, rightMargin=14, topMargin=14, bottomMargin=14,
        title="프레임 라벨 요약"
    )
    styles = getSampleStyleSheet()
    
    # 기본 스타일에 폰트 적용
    for k in ["Title", "Heading1", "Heading2", "Heading3", "BodyText"]:
        if k in styles.byName:
            styles[k].fontName = use_font

    # 표 셀에서 더 작은 글씨/줄바꿈을 위한 스타일
    cell_style = ParagraphStyle(
        "Cell",
        parent=styles["BodyText"],
        fontName=use_font,
        fontSize=8,
        leading=10,
    )
    small_head = ParagraphStyle(
        "SmallHead",
        parent=styles["Heading2"],
        fontName=use_font,
        fontSize=11,
        leading=13,
    )

    story = []
    story.append(Paragraph("프레임 라벨 요약", styles["Title"]))
    story.append(Spacer(1, 6))

    # === Legend (CV / Pred) ===
    story.append(Paragraph("코드 설명 (CV 규칙)", small_head))
    cv_rows = [[Paragraph("코드", cell_style), Paragraph("설명", cell_style)]]  # ← 변경
    for k, v in sorted(cv_text_map.items()):
        cv_rows.append([Paragraph(k, cell_style), Paragraph(v, cell_style)])

    tbl_cv = Table(cv_rows, colWidths=[doc.width * 0.12, doc.width * 0.88])
    tbl_cv.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("FONTNAME", (0,0), (-1,-1), use_font),  # ← 추가: 혹시 문자열이 남아도 폰트 강제
    ]))
    story.append(tbl_cv)
    story.append(Spacer(1, 6))

    story.append(Paragraph("코드 설명 (검출 결과)", small_head))
    pred_rows = [[Paragraph("코드", cell_style), Paragraph("설명", cell_style)]]  # ← 변경
    for k, v in sorted(pred_text_map.items()):
        pred_rows.append([Paragraph(k, cell_style), Paragraph(v, cell_style)])

    tbl_pred = Table(pred_rows, colWidths=[doc.width * 0.12, doc.width * 0.88])
    tbl_pred.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("FONTNAME", (0,0), (-1,-1), use_font),  # ← 추가
    ]))

    story.append(tbl_pred)
    story.append(Spacer(1, 10))

    # === Records ===
    if not records:
        story.append(Paragraph("활성 라벨이 없습니다(모든 프레임이 0).", styles["BodyText"]))
        doc.build(story)
        return

    header = ["프레임", "구간(s)", "활성 라벨", "수치 기반 결과", "예측 기반 결과"]
    table_rows = [[Paragraph(h, cell_style) for h in header]]

    for r in records:
        active_labels = [k for k, v in r["labels"].items() if v == 1]
        table_rows.append([
            Paragraph(str(r["frame"]), cell_style),
            Paragraph(f"{r['start']}–{r['end']}", cell_style),
            Paragraph(", ".join(active_labels) if active_labels else "-", cell_style),
            Paragraph(" / ".join(r["수치_기반_결과"]) if r["수치_기반_결과"] else "-", cell_style),
            Paragraph(" / ".join(r["예측_기반_결과"]) if r["예측_기반_결과"] else "-", cell_style),
        ])

    # 넓은 표: 가로폭을 꽉 쓰도록 가변 너비 설정 (A4 가로)
    avail = doc.width
    col_widths = [avail * 0.08, avail * 0.12, avail * 0.20, avail * 0.30, avail * 0.30]

    tbl = Table(table_rows, repeatRows=1, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 3),
        ("RIGHTPADDING", (0,0), (-1,-1), 3),
        ("TOPPADDING", (0,0), (-1,-1), 2),
        ("BOTTOMPADDING", (0,0), (-1,-1), 2),
        ("FONTSIZE", (0,0), (-1,-1), 8),
    ]))

    story.append(tbl)
    doc.build(story)


def _register_korean_font(STATIC_DIR: Path) -> str:
    """
    한글 지원 폰트를 등록하고 그 폰트명을 반환.
    우선순위:
      1) STATIC_DIR/fonts 안의 TTF/OTF (NanumGothic, NotoSansKR, Pretendard 등)
      2) ReportLab CID 폰트(HYGoThic-Medium) - 환경에 따라 미지원일 수 있음
      3) 실패 시 Helvetica (한글은 깨질 수 있음)
    """
    if not REPORTLAB_AVAILABLE:
        return "Helvetica"

    candidates = [
        STATIC_DIR / "fonts" / "PretendardGOV-ExtraBold.ttf",
        STATIC_DIR / "fonts" / "PretendardGOV-Bold.ttf",
        STATIC_DIR / "fonts" / "PretendardGOV-Regular.ttf",
        STATIC_DIR / "fonts" / "PretendardGOV-Medium.ttf",
        STATIC_DIR / "fonts" / "PretendardGOV-Light.ttf"
    ]
    for path in candidates:
        try:
            if path.exists():
                pdfmetrics.registerFont(TTFont("KoreanUI", str(path)))
                return "KoreanUI"
        except Exception:
            pass

    # CID 폰트(환경에 따라 동작). 실패해도 전파하지 않고 넘어간다.
    try:
        if _CID_AVAILABLE:
            pdfmetrics.registerFont(UnicodeCIDFont("HYGoThic-Medium"))
            return "HYGoThic-Medium"
    except Exception:
        pass

    # 최후 fallback
    return "Helvetica"