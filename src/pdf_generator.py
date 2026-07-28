import os
import sys
from pathlib import Path
from fpdf import FPDF
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

class CleanPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(30, 136, 229)
        self.cell(0, 10, 'Cosmetics Customer Intelligence & Analytics Report', border=False, ln=True, align='C')
        self.set_draw_color(200, 200, 200)
        self.line(10, 22, 200, 22)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}} | Executive Intelligence Report', align='C')

class PDFReportGenerator:
    """
    Automated PDF Report Builder for Customer Analytics.
    """
    def __init__(self, data_path=config.FINAL_SEGMENTED_PATH):
        self.data_path = Path(data_path)
        self.output_pdf = config.REPORTS_DIR / "Customer_Intelligence_Executive_Report.pdf"

    def build_pdf_report(self, insights_list):
        print("\n================ GENERATING EXECUTIVE PDF REPORT ================")
        if not self.data_path.exists():
            from src.clustering import CosmeticsClusteringEngine
            df, _ = CosmeticsClusteringEngine().run_all_clustering_algorithms()
        else:
            df = pd.read_csv(self.data_path)

        pdf = CleanPDF()
        pdf.alias_nb_pages()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Title / Executive Summary Header
        pdf.set_font('Helvetica', 'B', 16)
        pdf.set_text_color(33, 33, 33)
        pdf.cell(0, 10, 'Executive Analytics Summary', ln=True)
        pdf.ln(2)

        # Core Metrics Table
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_fill_color(240, 244, 248)
        
        tot_cust = len(df)
        tot_rev = df['Total_Spending'].sum()
        avg_spend = df['Total_Spending'].mean()
        top_cat = df['Preferred_Category'].mode()[0]

        pdf.cell(45, 8, 'Total Customers', 1, 0, 'C', fill=True)
        pdf.cell(45, 8, 'Total Revenue', 1, 0, 'C', fill=True)
        pdf.cell(50, 8, 'Avg Customer Spend', 1, 0, 'C', fill=True)
        pdf.cell(50, 8, 'Top Category', 1, 1, 'C', fill=True)

        pdf.set_font('Helvetica', '', 10)
        pdf.cell(45, 8, f"{tot_cust:,}", 1, 0, 'C')
        pdf.cell(45, 8, f"${tot_rev:,.2f}", 1, 0, 'C')
        pdf.cell(50, 8, f"${avg_spend:,.2f}", 1, 0, 'C')
        pdf.cell(50, 8, str(top_cat), 1, 1, 'C')

        pdf.ln(8)

        # Section: Automated Business Insights
        pdf.set_font('Helvetica', 'B', 13)
        pdf.set_text_color(30, 136, 229)
        pdf.cell(0, 8, 'Key Strategic Insights', ln=True)
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(50, 50, 50)

        for ins in insights_list:
            # Clean markup tags for standard PDF output
            clean_ins = ins.replace('**', '').replace('💡', '').replace('✨', '').replace('💄', '').replace('🎯', '').replace('⚠️', '').replace('💎', '')
            pdf.multi_cell(0, 6, f"- {clean_ins.strip()}")
            pdf.ln(1)

        pdf.ln(5)

        # Section: Customer Persona Summary Table
        pdf.set_font('Helvetica', 'B', 13)
        pdf.set_text_color(30, 136, 229)
        pdf.cell(0, 8, 'Customer Persona Segment Breakdown', ln=True)
        
        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_fill_color(230, 238, 248)
        
        pdf.cell(60, 7, 'Persona Name', 1, 0, 'L', fill=True)
        pdf.cell(30, 7, 'Count', 1, 0, 'C', fill=True)
        pdf.cell(35, 7, 'Avg Spend', 1, 0, 'R', fill=True)
        pdf.cell(35, 7, 'Avg Frequency', 1, 0, 'R', fill=True)
        pdf.cell(30, 7, 'Recency (Days)', 1, 1, 'R', fill=True)

        pdf.set_font('Helvetica', '', 9)
        persona_stats = df.groupby('Customer_Persona').agg(
            Count=('Customer_ID', 'count'),
            Avg_Spend=('Total_Spending', 'mean'),
            Avg_Freq=('Purchase_Frequency', 'mean'),
            Avg_Recency=('Days_Since_Last_Purchase', 'mean')
        ).reset_index()

        for _, row in persona_stats.iterrows():
            clean_name = row['Customer_Persona'].encode('latin-1', 'ignore').decode('latin-1')
            pdf.cell(60, 7, clean_name[:32], 1, 0, 'L')
            pdf.cell(30, 7, f"{int(row['Count']):,}", 1, 0, 'C')
            pdf.cell(35, 7, f"${row['Avg_Spend']:,.2f}", 1, 0, 'R')
            pdf.cell(35, 7, f"{row['Avg_Freq']:.1f}", 1, 0, 'R')
            pdf.cell(30, 7, f"{row['Avg_Recency']:.1f}", 1, 1, 'R')

        # Export PDF
        pdf.output(self.output_pdf)
        print(f"✅ PDF Executive Report generated successfully: '{self.output_pdf}'")
        print("================ PDF GENERATION COMPLETE ================\n")
        return self.output_pdf

if __name__ == "__main__":
    from src.insights import AutomatedInsightGenerator
    insights = AutomatedInsightGenerator().generate_all_insights()
    generator = PDFReportGenerator()
    generator.build_pdf_report(insights)
