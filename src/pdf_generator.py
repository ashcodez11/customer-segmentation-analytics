import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

class PDFReportGenerator:
    def build_pdf_report_bytes(self, df_filtered, insights_list):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        story = []

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'LumiereTitle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=20,
            textColor=colors.HexColor('#2E2A28'),
            spaceAfter=6
        )
        subtitle_style = ParagraphStyle(
            'LumiereSubTitle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            textColor=colors.HexColor('#8C837D'),
            spaceAfter=15
        )
        section_style = ParagraphStyle(
            'LumiereSection',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=12,
            textColor=colors.HexColor('#C9A86A'),
            spaceBefore=10,
            spaceAfter=8
        )
        body_style = ParagraphStyle(
            'LumiereBody',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9,
            textColor=colors.HexColor('#2E2A28'),
            spaceAfter=6
        )

        # Header
        story.append(Paragraph("LUMIÈRE AI ANALYTICS EXECUTIVE REPORT", title_style))
        story.append(Paragraph("Customer Intelligence, Persona Segmentation & Performance Insights", subtitle_style))
        story.append(Spacer(1, 10))

        # Executive Metrics Table
        story.append(Paragraph("EXECUTIVE KPI SUMMARY", section_style))
        kpi_data = [
            ["Metric", "Value"],
            ["Active Customer Base", f"{len(df_filtered):,}"],
            ["Gross Revenue", f"${df_filtered['Total_Spending'].sum():,.2f}"],
            ["Avg Customer Spend", f"${df_filtered['Total_Spending'].mean():,.2f}"],
            ["Avg Order Value", f"${df_filtered['Average_Order_Value'].mean():,.2f}"],
            ["Repeat Purchase Rate", f"{((df_filtered['Purchase_Frequency'] > 1).mean() * 100):.1f}%"]
        ]
        t = Table(kpi_data, colWidths=[250, 250])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (1,0), colors.HexColor('#F7F1EC')),
            ('TEXTCOLOR', (0,0), (1,0), colors.HexColor('#2E2A28')),
            ('FONTNAME', (0,0), (1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E8CFCF')),
        ]))
        story.append(t)
        story.append(Spacer(1, 15))

        # Insights
        story.append(Paragraph("AUTOMATED BUSINESS INSIGHTS", section_style))
        for ins in insights_list:
            story.append(Paragraph(f"• {ins}", body_style))

        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
