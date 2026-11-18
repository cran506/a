"""从本次数据生成20页正文、自动目录的PDF与可编辑Word。"""
from pathlib import Path
import json,html
from io import BytesIO
from reportlab.platypus import (BaseDocTemplate,Frame,PageTemplate,Paragraph,Spacer,PageBreak,Image,Table,TableStyle,Preformatted)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER,TA_LEFT
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from docx import Document
from docx.shared import Pt,Cm,Inches
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.section import WD_SECTION_START
from docx.enum.text import WD_ALIGN_PARAGRAPH
from report_content import make_pages,REFERENCES


TITLE='变负载液压伺服系统的多阶控制算法设计与对比'
FONT='STSong-Light'
pdfmetrics.registerFont(UnicodeCIDFont(FONT))
STYLES={
    'p':ParagraphStyle('body',fontName=FONT,fontSize=10.5,leading=17,spaceAfter=8,wordWrap='CJK',firstLineIndent=21),
    'h1':ParagraphStyle('course_h1',fontName=FONT,fontSize=17,leading=23,spaceAfter=9,keepWithNext=True,wordWrap='CJK'),
    'h2':ParagraphStyle('course_h2',fontName=FONT,fontSize=13,leading=19,spaceAfter=10,keepWithNext=True,wordWrap='CJK'),
    'cap':ParagraphStyle('caption',fontName=FONT,fontSize=9,leading=13,spaceAfter=8,alignment=TA_CENTER,wordWrap='CJK'),
    'cell':ParagraphStyle('cell',fontName=FONT,fontSize=9,leading=13,wordWrap='CJK'),
    'title':ParagraphStyle('title',fontName=FONT,fontSize=23,leading=34,alignment=TA_CENTER,wordWrap='CJK',spaceAfter=15),
    'center':ParagraphStyle('center',fontName=FONT,fontSize=13,leading=24,alignment=TA_CENTER,wordWrap='CJK'),
    'eq':ParagraphStyle('equation',fontName='Courier',fontSize=8.5,leading=12,spaceBefore=3,spaceAfter=10),
    'ref':ParagraphStyle('reference',fontName=FONT,fontSize=10,leading=17,spaceAfter=10,wordWrap='CJK'),
}


class ReportDocument(BaseDocTemplate):
    def __init__(self,target):
        super().__init__(str(target),pagesize=A4,leftMargin=54,rightMargin=54,topMargin=48,bottomMargin=48,
                         title=TITLE,author='课程设计：学号后四位0795',allowSplitting=1)
        frame=Frame(self.leftMargin,self.bottomMargin,self.width,self.height,leftPadding=0,bottomPadding=0,rightPadding=0,topPadding=0)
        self.addPageTemplates(PageTemplate(id='report',frames=frame,onPage=self.decorate))
    def beforeDocument(self):self.heading_pages=[];self.key_counter=0
    def decorate(self,canvas,doc):
        if doc.page<3:return
        canvas.setFont(FONT,9);canvas.setFillColor(colors.HexColor('#555555'))
        canvas.drawString(54,A4[1]-31,'《智能流体传动与控制》课程设计 · 学号后四位0795')
        canvas.line(54,A4[1]-37,A4[0]-54,A4[1]-37)
        canvas.drawCentredString(A4[0]/2,26,str(doc.page-2))
    def afterFlowable(self,flowable):
        if isinstance(flowable,Paragraph) and flowable.style.name in ['course_h1','course_h2']:
            level=0 if flowable.style.name=='course_h1' else 1;text=flowable.getPlainText();key='h'+str(self.key_counter);self.key_counter+=1
            self.canv.bookmarkPage(key);self.canv.addOutlineEntry(text,key,level=level,closed=False)
            self.notify('TOCEntry',(level,text,self.page-2,key));self.heading_pages.append((level,text,self.page-2))


def para(text,style='p'):
    content=html.escape(str(text)).replace('\n','<br/>')
    content=content.replace('²','<super>2</super>').replace('³','<super>3</super>').replace('μm','微米').replace('−','-').replace('·','*')
    return Paragraph(content,STYLES[style])


def pdf_table(headers,rows,caption,width):
    cells=[[para(c,'cell') for c in headers]]+[[para(c,'cell') for c in row] for row in rows]
    table=Table(cells,colWidths=[width/len(headers)]*len(headers),repeatRows=1,hAlign='CENTER')
    table.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,0),1.,colors.black),('LINEBELOW',(0,0),(-1,0),.6,colors.black),
                               ('LINEBELOW',(0,-1),(-1,-1),1.,colors.black),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
                               ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    return [para(caption,'cap'),table,Spacer(1,9)]


def appendix_blocks():
    return [
        ('p','学号后四位：0795；Python整数种子：795。主程序和辨识模块显式调用np.random.seed(seed)，DE与带噪观测由该种子决定。独立验证噪声使用default_rng(796)局部派生生成器，全局np.random.seed仍设795；训练与验证激励由不同频率公式确定，原始数据与真值同时保留。'),
        ('e','python run_course_design.py --seed 0795\npython -m unittest -v\npython run_course_design.py --mode verify --seed 0795'),
        ('p','来源快照：cran506/a @ bae77ed1fbfb6d9711a50e2e7a9bad437945e5bb。第五章controllers.py提供增量输出记忆与滤波微分思想；第六章model_assisted_controller.py提供液压力G/H反演结构；第七章mpc_core.py提供仿射预测、增量决策和压力约束组织方式。本工程统一重构为四个物理状态，未导入第八章代码。'),
        ('p','数据字典：time_s为采样时刻；error_mm为参考减实际；u_v为已经过死区逆和限幅后的实际驱动电压；p1/p2_min/max_mpa为此前一个控制周期内的RK4子步节点极值；estimated_load_n为控制器独立因果估计；qp_status、qp_residual和qp_dual_residual保留求解诊断。'),
        ('p','控制峰压由全轨迹双腔子步记录计算，并以独立Radau轨迹复算。verification.py重放每一控制器自己的保存电压，不重新调用控制器来生成“参考答案”。附加制动模型辅助组因逆增益非正而终止，失败后没有填充后续轨迹。'),
        ('p','未完成个人信息请由本人填写：姓名、完整学号、学院和专业。未采集真实设备数据；没有完成或宣称完成SAC/HOCBF、SPRT、LSTM与RUL研究。报告可作为当前不含第八章范围的课程设计成果，不能冒充原任务书包含SAC的全量版本。'),
    ]


def build_pdf(root,pages):
    target=root/'课程设计报告_0795.pdf';doc=ReportDocument(target);story=[]
    story += [Spacer(1,70),para('《智能流体传动与控制》','center'),Spacer(1,25),para('综合课程设计报告','title'),Spacer(1,18),para(TITLE,'title'),Spacer(1,25),
              para('当前范围：课程第2—7章；第八章按要求暂缓','center'),Spacer(1,32)]
    for line in ['姓名：________________','完整学号：________________（后四位0795）','学院/专业：________________','指导教师：崔岩','完成日期：2026年11月18日']:
        story.append(para(line,'center'))
    story += [Spacer(1,20),para('正文20页 · 合成辨识数据 · 统一物理闭环 · 结果可复现','cap'),PageBreak(),para('目录','title')]
    toc=TableOfContents();toc.levelStyles=[ParagraphStyle('toc0',fontName=FONT,fontSize=11,leading=18,spaceBefore=5),
                                           ParagraphStyle('toc1',fontName=FONT,fontSize=9.5,leading=15,leftIndent=18)]
    story += [toc,PageBreak()]
    for i,page in enumerate(pages):
        if page['chapter']:story.append(para(page['chapter'],'h1'))
        story.append(para(page['title'],'h2'))
        for block in page['blocks']:story.extend(pdf_block(block,root,doc.width))
        story.append(PageBreak())
    story.append(para('参考文献','h1'))
    story.append(para('以下按GB/T 7714—2015的主要格式组织；课程报告不要求毕业设计附加的20篇综述。','p'))
    for entry in REFERENCES:story.append(para(entry,'ref'))
    story.extend([PageBreak(),para('附录A 运行、种子声明与数据字典','h1')])
    for block in appendix_blocks():story.extend(pdf_block(block,root,doc.width))
    doc.multiBuild(story)
    import fitz
    with fitz.open(target) as pdf:
        num_pages=len(pdf);text='\n'.join(p.get_text() for p in pdf)
        body_heading_pages=[(page['title'],next(number for level,title,number in doc.heading_pages if title==page['title'])) for page in pages]
        if [n for _,n in body_heading_pages]!=list(range(1,21)):
            raise RuntimeError('正文排版未保持20页：'+str(body_heading_pages))
        if num_pages!=24:raise RuntimeError(f'报告预期24页（含封面目录参考文献附录），实际{num_pages}')
        if '0795' not in text or '5000' not in text:raise RuntimeError('报告内容完整性检查失败')
    return target,doc.heading_pages,num_pages


def pdf_block(block,root,width):
    kind=block[0]
    if kind=='p':return [para(block[1])]
    if kind=='e':return [Preformatted(block[1],STYLES['eq'])]
    if kind=='t':return pdf_table(block[1],block[2],block[3],width)
    if kind=='f':
        image=Image(str(root/'figures'/f'{block[1]}.png'));ratio=min(width/image.imageWidth,250/image.imageHeight)
        image.drawWidth=image.imageWidth*ratio;image.drawHeight=image.imageHeight*ratio
        return [image,Spacer(1,4),para(block[2],'cap')]
    raise ValueError(kind)


def east_font(style,name='宋体'):
    style.font.name=name;style._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),name)


def set_borders(table):
    props=table._tbl.tblPr;borders=OxmlElement('w:tblBorders')
    for edge in ['top','bottom','left','right','insideH','insideV']:
        el=OxmlElement('w:'+edge);el.set(qn('w:val'),'single' if edge in ['top','bottom'] else 'nil');el.set(qn('w:sz'),'8');borders.append(el)
    props.append(borders)
    for cell in table.rows[0].cells:
        pr=cell._tc.get_or_add_tcPr();b=OxmlElement('w:tcBorders');el=OxmlElement('w:bottom');el.set(qn('w:val'),'single');el.set(qn('w:sz'),'5');b.append(el);pr.append(b)


def word_block(document,block,root):
    kind=block[0]
    if kind in ['p','e']:
        p=document.add_paragraph(block[1],style='Normal')
        if kind=='e':
            p.paragraph_format.first_line_indent=Cm(0)
            for run in p.runs:run.font.name='Consolas';run.font.size=Pt(8)
        return
    if kind=='f':
        from PIL import Image as PILImage
        path=root/'figures'/f'{block[1]}.png';w,h=PILImage.open(path).size
        width=min(6.2,3.4*w/h)
        p=document.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.first_line_indent=Cm(0)
        p.add_run().add_picture(str(path),width=Inches(width))
        p=document.add_paragraph(block[2],style='Caption');p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        return
    if kind=='t':
        p=document.add_paragraph(block[3],style='Caption');p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        headers,rows=block[1:3];table=document.add_table(rows=1,cols=len(headers));table.autofit=True
        for cell,text in zip(table.rows[0].cells,headers):cell.text=str(text)
        for values in rows:
            for cell,text in zip(table.add_row().cells,values):cell.text=str(text)
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.first_line_indent=Cm(0);p.paragraph_format.space_after=Pt(2);p.paragraph_format.line_spacing=1.
                    for run in p.runs:run.font.size=Pt(9)
        set_borders(table);document.add_paragraph()


def build_docx(root,pages,headings):
    doc=Document();section=doc.sections[0];section.page_height=Cm(29.7);section.page_width=Cm(21)
    section.top_margin=Cm(1.9);section.bottom_margin=Cm(1.9);section.left_margin=Cm(1.9);section.right_margin=Cm(1.9)
    normal=doc.styles['Normal'];east_font(normal);normal.font.size=Pt(10.5)
    normal.paragraph_format.line_spacing=1.25;normal.paragraph_format.space_after=Pt(5);normal.paragraph_format.first_line_indent=Cm(.74)
    for name,size in [('Heading 1',16),('Heading 2',12.5),('Title',22),('Caption',9)]:
        style=doc.styles[name];east_font(style);style.font.size=Pt(size);style.paragraph_format.first_line_indent=Cm(0)
    for line in ['《智能流体传动与控制》','综合课程设计报告',TITLE,'当前范围：第2—7章；第八章暂缓','姓名：________________','完整学号：________________（后四位0795）','学院/专业：________________','指导教师：崔岩','完成日期：2026年11月18日']:
        p=doc.add_paragraph(line,style='Title' if line in ['综合课程设计报告',TITLE] else 'Normal');p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.first_line_indent=Cm(0);p.paragraph_format.space_after=Pt(17)
    doc.add_page_break();p=doc.add_paragraph('目录',style='Title');p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p=doc.add_paragraph();p.paragraph_format.first_line_indent=Cm(0);p.paragraph_format.line_spacing=1.
    run=p.add_run();begin=OxmlElement('w:fldChar');begin.set(qn('w:fldCharType'),'begin');run._r.append(begin)
    instr=OxmlElement('w:instrText');instr.set(qn('xml:space'),'preserve');instr.text=' TOC \\o "1-2" \\h \\z \\u ';run._r.append(instr)
    separate=OxmlElement('w:fldChar');separate.set(qn('w:fldCharType'),'separate');run._r.append(separate)
    for level,title,number in headings:
        r=p.add_run(('    ' if level else '')+title+'  '+str(number));r.font.size=Pt(9)
        r.add_break()
    end=OxmlElement('w:fldChar');end.set(qn('w:fldCharType'),'end');p.add_run()._r.append(end)
    body_section=doc.add_section(WD_SECTION_START.NEW_PAGE);body_section.header.is_linked_to_previous=False;body_section.footer.is_linked_to_previous=False
    page_num=OxmlElement('w:pgNumType');page_num.set(qn('w:start'),'1');body_section._sectPr.append(page_num)
    hp=body_section.header.paragraphs[0];hp.text='《智能流体传动与控制》课程设计 · 学号后四位0795';hp.style='Caption'
    fp=body_section.footer.paragraphs[0];fp.alignment=WD_ALIGN_PARAGRAPH.CENTER
    f=OxmlElement('w:fldSimple');f.set(qn('w:instr'),'PAGE');fp._p.append(f)
    for index,page in enumerate(pages):
        if index:doc.add_page_break()
        if page['chapter']:doc.add_heading(page['chapter'],level=1)
        doc.add_heading(page['title'],level=2)
        for block in page['blocks']:word_block(doc,block,root)
    doc.add_page_break();doc.add_heading('参考文献',level=1)
    for entry in REFERENCES:doc.add_paragraph(entry)
    doc.add_page_break();doc.add_heading('附录A 运行、种子声明与数据字典',level=1)
    for block in appendix_blocks():word_block(doc,block,root)
    update=OxmlElement('w:updateFields');update.set(qn('w:val'),'true');doc.settings.element.append(update)
    doc.core_properties.title=TITLE;doc.core_properties.subject='不含第八章的课程设计；种子0795';doc.core_properties.author='课程设计：0795'
    target=root/'课程设计报告_0795.docx';doc.save(target)
    # 读回检查标题、章节、目录域与图片，不把“写出成功”当成内容完整。
    check=Document(target);text='\n'.join(p.text for p in check.paragraphs)
    assert all(page['title'] in text for page in pages)
    assert len(check.inline_shapes)>=10
    assert 'TOC' in check._element.xml
    return target


def build_report(results):
    root=Path(results);pages=make_pages(root)
    pdf,headings,count=build_pdf(root,pages);word=build_docx(root,pages,headings)
    markdown=['# '+TITLE,'','学号后四位：0795；种子：795。当前范围不含第八章。','']
    for page in pages:
        if page['chapter']:markdown.extend(['# '+page['chapter'],''])
        markdown.extend(['## '+page['title'],''])
        for block in page['blocks']:
            if block[0]=='p':markdown.extend([block[1],''])
            elif block[0]=='e':markdown.extend(['```text',block[1],'```',''])
            elif block[0]=='f':markdown.extend([f'![{block[2]}](figures/{block[1]}.png)',block[2],''])
            elif block[0]=='t':
                markdown.extend([block[3],'','| '+' | '.join(block[1])+' |','|'+'|'.join(['---']*len(block[1]))+'|'])
                markdown.extend('| '+' | '.join(str(v) for v in row)+' |' for row in block[2]);markdown.append('')
    markdown.extend(['# 参考文献','']+REFERENCES)
    (root/'REPORT.md').write_text('\n'.join(markdown)+'\n',encoding='utf-8')
    summary=dict(title=TITLE,pdf_total_pages=count,body_pages=20,front_matter_pages=2,references_pages=1,appendix_pages=1,
                 student_id_last_four='0795',seed_integer=795,pdf_auto_toc=True,word_auto_toc_field=True,
                 word_toc_note='已写入自动TOC域及PDF对应页码缓存；在Word全选F9可更新。不同字体/编辑会影响Word分页，固定页码以PDF为准。',
                 chapter_titles=[page['chapter'] for page in pages if page['chapter']],
                 pdf_file=pdf.name,word_file=word.name)
    (root/'report_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print('报告生成：',pdf,word,flush=True)
    return summary
