from playwright.sync_api import sync_playwright
import os
# import base64
# import re
#
# from markdown_it import MarkdownIt
#
# from pykatex import render
# import markdown
# from pygments import highlight
# from pygments.lexers import get_lexer_by_name
# from pygments.formatters import HtmlFormatter
# from markdown.extensions.codehilite import CodeHiliteExtension


# INPUT_HTML = r'E:\virtualxiaoman.github.io\article\深度学习.html'
INPUT_HTML = r'D:\HP\Desktop\小满の大学笔记\数据科学导论\深度学习.HTML'
OUTPUT_HTML = 'output.html'
OUTPUT_PDF = 'output.pdf'
# INPUT_MD = r'D:\HP\Desktop\小满の大学笔记\数据科学导论\深度学习.md'

# def md_to_html(md_file, output_html):
#     """将 Markdown 文件转换为 HTML"""
#     with open(md_file, 'r', encoding='utf-8') as f:
#         md_content = f.read()
#
#     # 使用 markdown-it-py 转换为 HTML
#     md = MarkdownIt()
#     html_content = md.render(md_content)
#
#     # 保存 HTML 文件（可选，用于检查生成的 HTML）
#     with open(output_html, 'w', encoding='utf-8') as f:
#         f.write(html_content)
#
#     return html_content


# def md2katex(md: str) -> str:
#     # 转换 $$公式$$ 为占位符
#     md_with_placeholders = re.sub(r'\$\$(.*?)\$\$',
#                                   lambda m: f'{{{{katex_block:{base64.b64encode(m.group(1).encode()).decode()}}}}}', md)
#     # 转换 $公式$ 为占位符
#     md_with_placeholders = re.sub(r'\$(.*?)\$',
#                                   lambda m: f'{{{{katex_inline:{base64.b64encode(m.group(1).encode()).decode()}}}}}',
#                                   md_with_placeholders)
#     return md_with_placeholders
#
#
# def md2html(md: str) -> str:
#     # 使用 markdown 转换为 HTML
#     html = markdown.markdown(md, extensions=[CodeHiliteExtension(linenums=False, css_class="highlight")])
#     return html
#
#
# def highlight_code(html: str) -> str:
#     # 手动高亮代码块
#     def replacer(match):
#         lexer = get_lexer_by_name(match.group(1), stripall=True)
#         formatter = HtmlFormatter()
#         return highlight(match.group(2), lexer, formatter)
#
#     html = re.sub(r'<pre><code class="language-(.*?)">(.*?)</code></pre>', replacer, html, flags=re.S)
#     return html
#
#
# def katex2html(html: str) -> str:
#     # 替换占位符为 KaTeX 渲染的公式
#     html = re.sub(r'{{katex_block:(.*?)}}', lambda m: render(base64.b64decode(m.group(1)).decode(), display_mode=True), html)
#     html = re.sub(r'{{katex_inline:(.*?)}}', lambda m: render(base64.b64decode(m.group(1)).decode(), display_mode=False), html)
#     return html
#
#
# def process_md_to_html(md_content: str) -> str:
#     # 第一步：处理数学公式
#     md_with_placeholders = md2katex(md_content)
#
#     # 第二步：转换为 HTML 并处理代码高亮
#     html = md2html(md_with_placeholders)
#     html = highlight_code(html)
#
#     # 第三步：替换占位符为 KaTeX 渲染的公式
#     html_with_katex = katex2html(html)
#
#     return html_with_katex


def html_to_pdf(html_file, output_pdf):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        # 加载 HTML 文件
        page.goto(f'file://{os.path.abspath(html_file)}')
        # 设置边距并保存为 PDF
        page.pdf(path=output_pdf, format='A4', margin={
            'top': '20mm',
            'bottom': '20mm',
            'left': '15mm',
            'right': '15mm'
        })
        browser.close()


if __name__ == '__main__':
    # # # 第一步：将 Markdown 转换为 HTML
    # # html_content = md_to_html(INPUT_MD, OUTPUT_HTML)
    # with open(INPUT_MD, 'r', encoding='utf-8') as f:
    #     md_content = f.read()
    # html = process_md_to_html(md_content)
    # with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
    #     f.write(html)
    # 第二步：将生成的 HTML 转换为 PDF
    html_to_pdf(INPUT_HTML, OUTPUT_PDF)




# from weasyprint import HTML
# HTML(INPUT_HTML).write_pdf(OUTPUT_PDF)

# import asyncio
# from pyppeteer import launch
# import os
#
#
# async def html_to_pdf(html_file, output_pdf):
#     # 确保使用绝对路径
#     html_file_path = f'file://{os.path.abspath(html_file)}'
#
#     browser = await launch()
#     page = await browser.newPage()
#
#     # 加载本地 HTML 文件
#     await page.goto(html_file_path, {'waitUntil': 'networkidle2'})
#
#     # 保存为 PDF
#     await page.pdf({'path': output_pdf, 'format': 'A4'})
#
#     await browser.close()


# 使用异步任务
# asyncio.get_event_loop().run_until_complete(
#     html_to_pdf(INPUT_HTML, OUTPUT_PDF)
# )
