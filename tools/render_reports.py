"""Render every published page and contact sheets for manual layout review (Pillow)."""
from pathlib import Path
import argparse
import json
import subprocess
from report_publication import NAMES,check_publication
ROOT=Path(__file__).resolve().parents[1]

def main():
    from PIL import Image,ImageDraw
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('/tmp/co5085-pdf-review'))
    args=parser.parse_args();statuses=check_publication(ROOT)
    for name in NAMES:
        folder=args.output/name[:-4];folder.mkdir(parents=True,exist_ok=True)
        # Clean only renderer-owned page/sheet files; never touch source PDFs.
        for pattern in ['page-*.png','sheet-*.png']:
            for p in folder.glob(pattern):p.unlink()
        subprocess.run(['pdftoppm','-r','75','-png',str(ROOT/'reports'/name),str(folder/'page')],check=True)
        pages=sorted(folder.glob('page-*.png'))
        if len(pages)!=statuses[name]['pages']:raise ValueError('Render page count differs')
        for offset in range(0,len(pages),12):
            sheet=Image.new('RGB',(1440,1550),'#e2e8f0');draw=ImageDraw.Draw(sheet)
            for j,p in enumerate(pages[offset:offset+12]):
                image=Image.open(p).convert('RGB');image.thumbnail((350,485))
                x=j%4*360+5;y=j//4*515+23;sheet.paste(image,(x,y))
                draw.text((x,y-18),name+' / '+str(offset+j+1),fill='black')
            sheet.save(folder/f'sheet-{offset//12+1}.png')
        print(f'{name}: rendered all {len(pages)} pages -> {folder}',flush=True)
    (args.output/'render_record.json').write_text(json.dumps(statuses,indent=2)+'\n')

if __name__=='__main__':main()
