import pandas as pd
from datetime import datetime, timedelta
from jinja2 import Environment, FileSystemLoader
import pdfkit

class ReportGenerator:
    def __init__(self, template_dir="app/reporting/templates"):
        self.env = Environment(loader=FileSystemLoader(template_dir))
    
    def generate_sar(self, transactions, customer_info, output_path):
        # Suspicious Activity Report
        template = self.env.get_template("sar_template.html")
        
        context = {
            "transactions": transactions,
            "customer": customer_info,
            "report_date": datetime.now().strftime("%Y-%m-%d"),
            "threshold": 0.7  # Configurable risk threshold
        }
        
        html = template.render(context)
        pdfkit.from_string(html, output_path)
    
    def generate_ctr(self, transactions, output_path):
        # Currency Transaction Report (for large transactions)
        large_txns = [t for t in transactions if t['amount'] > 10000]  # Example threshold
        
        template = self.env.get_template("ctr_template.html")
        html = template.render({
            "transactions": large_txns,
            "report_date": datetime.now().strftime("%Y-%m-%d")
        })
        
        pdfkit.from_string(html, output_path)
    
    def generate_daily_summary(self, stats, output_path):
        # Daily compliance summary
        template = self.env.get_template("daily_summary.html")
        html = template.render({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "total_transactions": stats['total'],
            "flagged_transactions": stats['flagged'],
            "sar_filed": stats['sar_count'],
            "ctr_filed": stats['ctr_count']
        })
        
        pdfkit.from_string(html, output_path)