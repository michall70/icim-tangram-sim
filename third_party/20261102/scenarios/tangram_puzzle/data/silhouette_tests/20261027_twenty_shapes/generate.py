from pathlib import Path
import json, math, random, zipfile
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
source = json.loads((ROOT.parent / '20261026_chevron/solution_geometry.json').read_text())
base = [np.array(p, dtype=float) for p in source['pieces']]
base[0] -= [0, 4]
rng = random.Random(2026102720)

def area(p):
    q = np.roll(p, -1, axis=0)
    return abs((p[:,0]*q[:,1]-p[:,1]*q[:,0]).sum()) / 2

def overlaps(a, b):
    for p in (a, b):
        for edge in np.roll(p, -1, axis=0) - p:
            axis = np.array([-edge[1], edge[0]])
            x, y = a @ axis, b @ axis
            if min(x.max(), y.max()) - max(x.min(), y.min()) <= 1e-8:
                return False
    return True

def render(pieces, size=1024):
    pts = np.concatenate(pieces)
    lo, hi = pts.min(0), pts.max(0)
    scale = (size * 0.80) / max(hi-lo)
    offset = (size - (hi-lo)*scale)/2 - lo*scale
    pixels = [np.rint(p*scale+offset).astype(np.int32) for p in pieces]
    mask = np.zeros((size,size), np.uint8)
    for p in pixels:
        cv2.fillPoly(mask, [p], 255)
    return mask, pixels, scale, offset

def valid(pieces):
    if any(overlaps(a,b) for i,a in enumerate(pieces) for b in pieces[i+1:]):
        return False
    mask, _, _, _ = render(pieces, 1024)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours) == 1 and cv2.connectedComponents(mask, connectivity=4)[0] == 2

def assemble():
    order = [0] + rng.sample(list(range(1,7)),6)
    placed = {0: base[0].copy()}
    for idx in order[1:]:
        p = base[idx]
        candidates = []
        for q in placed.values():
            for a,b in zip(q,np.roll(q,-1,axis=0)):
                for c,d in zip(p,np.roll(p,-1,axis=0)):
                    v, w = b-a, c-d
                    angle = math.atan2(v[1],v[0])-math.atan2(w[1],w[0])
                    R = np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
                    turned = p @ R.T
                    for t in (0.,0.5,1.):
                        new = turned + (a+t*(b-a)) - ((d+t*(c-d))@R.T)
                        if any(overlaps(new,x) for x in placed.values()):
                            continue
                        allpts=np.concatenate(list(placed.values())+[new])
                        bbox=np.ptp(allpts,axis=0)
                        score=bbox.prod()*(1+rng.random()*0.65)
                        candidates.append((score,new))
        if not candidates:
            return None
        candidates.sort(key=lambda z:z[0])
        placed[idx] = candidates[rng.randrange(min(5,len(candidates)))][1]
    result=[placed[i] for i in range(7)]
    return result if valid(result) else None

sets=[]
signatures=set()
previous_masks=[]
for oldfile in sorted((ROOT.parent/'20261027_five_shapes/answers').glob('solution_*.json')):
    oldpieces=[np.array(p,float) for p in json.loads(oldfile.read_text())['pieces']]
    previous_masks.append(render(oldpieces,320)[0]>0)
previous_masks.append(render([np.array(p,float) for p in source['pieces']],320)[0]>0)
for attempt in range(600):
    if len(sets)==20:
        break
    result=assemble()
    if result is None:
        continue
    pts=np.concatenate(result)
    dims=np.sort(np.ptp(pts,axis=0))
    mask,_,_,_=render(result,320)
    contour=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0][0]
    corners=len(cv2.approxPolyDP(contour,1.5,True))
    signature=(tuple(np.round(dims,2)),corners)
    if signature in signatures or corners<6:
        continue
    binary=mask>0
    duplicate=False
    for old in previous_masks:
        for k in range(4):
            test=np.rot90(binary,k)
            if np.count_nonzero(test & old)/np.count_nonzero(test | old)>0.98:
                duplicate=True
    if duplicate:
        continue
    previous_masks.append(binary)
    signatures.add(signature)
    sets.append(result)
assert len(sets)==20
colors=['#60a5fa','#f87171','#fb923c','#c084fc','#facc15','#4ade80','#22d3ee']
font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',36)
thumbs=[]
for n,pieces in enumerate(sets,1):
    assert valid(pieces)
    assert np.allclose([area(p) for p in pieces],[4,4,2,1,1,2,2])
    mask,pixels,scale,offset=render(pieces)
    image=Image.fromarray(mask).convert('RGB')
    name=f'silhouette_{n:02}.png'
    image.save(ROOT/name)
    guide=Image.new('RGB',(1024,1024),'#111827')
    draw=ImageDraw.Draw(guide)
    for k,(p,col) in enumerate(zip(pixels,colors),1):
        xy=[tuple(map(int,v)) for v in p]
        draw.polygon(xy,fill=col)
        draw.line(xy+[xy[0]],fill='white',width=3)
        draw.text(tuple(p.mean(0)),str(k),font=font,anchor='mm',fill='#111827')
    guide.save(ROOT/'answers'/f'answer_{n:02}.png')
    (ROOT/'answers'/f'solution_{n:02}.json').write_text(json.dumps({
        'target_image':name,'piece_order':source['piece_order'],
        'pieces':[p.round(10).tolist() for p in pieces],
        'pixel_transform':{'scale':scale,'offset':offset.tolist()},
        'note':'One geometric solution; not robot coordinates. Rotations and translations only, no reflections.',
        'verified':{'piece_areas':[4,4,2,1,1,2,2],'no_interior_overlap':True,'one_connected_silhouette_no_holes':True}
    },indent=2),encoding='utf-8')
    thumbs.append(image.resize((300,300)))
sheet=Image.new('RGB',(1500,1360),'#202020')
draw=ImageDraw.Draw(sheet)
for i,thumb in enumerate(thumbs):
    sheet.paste(thumb,((i%5)*300,(i//5)*340+40))
    draw.text(((i%5)*300+150,(i//5)*340+20),f'{i+1:02}',font=font,anchor='mm',fill='white')
sheet.save(ROOT/'preview.png')
with zipfile.ZipFile(ROOT/'twenty_silhouettes_with_answers.zip','w',zipfile.ZIP_DEFLATED) as z:
    for path in sorted(ROOT.glob('silhouette_*.png')):
        z.write(path,path.name)
    for path in sorted((ROOT/'answers').iterdir()):
        z.write(path,'answers/'+path.name)
print('Created 20 independent silhouettes and answers; geometric checks passed.')
print(ROOT)

with zipfile.ZipFile(ROOT/'twenty_silhouettes_only.zip','w',zipfile.ZIP_DEFLATED) as z:
    for path in sorted(ROOT.glob('silhouette_*.png')):
        z.write(path,path.name)
checks=[]
for path in sorted(ROOT.glob('silhouette_*.png')):
    img=Image.open(path)
    arr=np.asarray(img)
    assert img.size==(1024,1024) and img.mode=='RGB'
    assert set(np.unique(arr))=={0,255}
    assert np.array_equal(arr[:,:,0],arr[:,:,1]) and np.array_equal(arr[:,:,1],arr[:,:,2])
    ys,xs=np.where(arr[:,:,0]>0)
    margins=[int(xs.min()),int(ys.min()),int(1023-xs.max()),int(1023-ys.max())]
    assert min(margins)>=100
    assert abs(margins[0]-margins[2])<=1 and abs(margins[1]-margins[3])<=1
    assert cv2.connectedComponents(arr[:,:,0],connectivity=4)[0]==2
    checks.append({'file':path.name,'size':[1024,1024],'pure_binary':True,'margins_ltrb':margins,'foreground_components':1})
assert len(checks)==20
(ROOT/'validation.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print('Verified 20 files: binary RGB, centered, margins >=100 px, one connected white region.')
