"""Optional report export; does not perform or select numerical experiments."""
from pathlib import Path
import re
import markdown
from weasyprint import HTML

def main():
 root=Path(__file__).resolve().parents[1];source=root/'FINAL_RESEARCH_REPORT_KO.md';body=markdown.markdown(source.read_text(),extensions=['tables','fenced_code'])
 base='https://github.com/cywon-commits/Physics-Based-Geometric-Image-Analysis/blob/codex/final-pbgia-20261117/final_pbgia_study/'
 body=re.sub(r'href="(?!https?://|#)([^"]+)"',lambda m:'href="'+base+m.group(1)+'"',body)
 css='''@page{size:A4;margin:18mm 16mm 17mm;@bottom-center{content:counter(page);font-size:8pt;color:#666}}body{font-family:"Noto Sans CJK KR",sans-serif;font-size:9.1pt;line-height:1.65;color:#172231;word-break:normal}h1{font-size:20pt;line-height:1.4;color:#123650}h2{font-size:13pt;margin-top:18pt;break-after:avoid;color:#123650}p{orphans:3;widows:3}table{border-collapse:collapse;width:100%;font-size:7.5pt;line-height:1.45;margin:9pt 0}th,td{border:0.5pt solid #cad3dc;padding:4pt;overflow-wrap:anywhere}th{background:#e8eef4}tr{break-inside:avoid}img{display:block;width:100%;max-height:125mm;object-fit:contain;margin:8pt auto}pre{font-size:7.6pt;white-space:pre-wrap;overflow-wrap:anywhere;background:#edf2f6;padding:7pt}code{font-family:"Noto Sans Mono CJK KR",monospace;font-size:8pt}a{color:#155c8a;text-decoration:none}strong{font-weight:700}'''
 HTML(string='<html lang="ko"><meta charset="utf-8"><style>'+css+'</style><body>'+body+'</body></html>',base_url=str(root)).write_pdf(root/'output/FINAL_RESEARCH_REPORT_KO.pdf')
 print(root/'output/FINAL_RESEARCH_REPORT_KO.pdf')

if __name__=='__main__':main()
