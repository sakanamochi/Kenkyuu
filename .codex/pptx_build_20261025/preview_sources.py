from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import pypdfium2 as pdfium

# 元資料の画像を一覧表示し、利用する図の内容を確認する。
out=Path.cwd()/'.codex/pptx_build_20261025'
for folder in out.glob('第*'):
    paths=[]
    for p in folder.iterdir():
        try:
            im=Image.open(p); im.verify(); paths.append(p)
        except Exception: pass
    canvas=Image.new('RGB',(1200,220*((len(paths)+3)//4)), 'white')
    draw=ImageDraw.Draw(canvas)
    for i,p in enumerate(paths):
        im=Image.open(p).convert('RGB'); im.thumbnail((290,185))
        x=(i%4)*300; y=(i//4)*220
        canvas.paste(im,(x,y+25)); draw.text((x+4,y+5),p.name,fill='black')
    canvas.save(folder/'montage.jpg')
for n in ['塚田涼太郎＿進捗報告.pdf','岸知輝_進捗報告.pdf']:
    pdf=pdfium.PdfDocument(str(Path('C:/Users/siomi/Downloads')/n))
    for i in [2,6,10]:
        pdf[i].render(scale=1.1).to_pil().save(out/(n[:2]+f'-{i+1}.png'))
