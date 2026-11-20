import json, zipfile, pathlib, xml.etree.ElementTree as ET
from pypdf import PdfReader

# 元資料の文字列と埋め込み画像を読み取り専用で抽出する。
root=pathlib.Path.cwd()
out=root/'.codex/pptx_build_20261025'
ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','p':'http://schemas.openxmlformats.org/presentationml/2006/main'}
records=[]
for p in (root/'研究関連書類').glob('第*回*.pptx'):
    d=out/p.stem; d.mkdir(exist_ok=True)
    with zipfile.ZipFile(p) as z:
        slides=sorted([n for n in z.namelist() if n.startswith('ppt/slides/slide') and n.endswith('.xml')],key=lambda x:int(x.split('slide')[-1].split('.')[0]))
        rec={'file':str(p),'slides':[]}
        for n in slides:
            x=ET.fromstring(z.read(n)); rel=n.replace('slides/','slides/_rels/')+'.rels'
            r=ET.fromstring(z.read(rel)) if rel in z.namelist() else []
            rec['slides'].append({'number':int(n.split('slide')[-1].split('.')[0]),'text':'\n'.join(t.text or '' for t in x.findall('.//a:t',ns)),'rels':[(t.attrib.get('Id'),t.attrib.get('Target')) for t in r]})
        for n in z.namelist():
            if n.startswith('ppt/media/'):
                (d/pathlib.Path(n).name).write_bytes(z.read(n))
        records.append(rec)
for name in ['塚田涼太郎＿進捗報告.pdf','岸知輝_進捗報告.pdf']:
    p=pathlib.Path('C:/Users/siomi/Downloads')/name
    pdf=PdfReader(p)
    records.append({'file':str(p),'pages':[{'number':i+1,'text':s.extract_text()} for i,s in enumerate(pdf.pages)]})
(out/'sources.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
for r in records:
    print('\nFILE',r['file'])
    for s in r.get('slides',r.get('pages',[])):
        print('\nPAGE',s['number'],s['text'])
