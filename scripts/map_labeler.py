#!/usr/bin/env python3
"""Hand-label the occupied cells of 2D lidar maps in the browser: keep / erase / unsure.

    python3 scripts/map_labeler.py [--labels data/map_cleaning/labels] [--maps maps] [--port 8507] [--host 127.0.0.1]
    open http://127.0.0.1:8507

Lists every map with a <name>_prelabel.png in the labels directory. Opening a map loads <name>_label.png
if it exists, else the prelabel; occupied cells that neither file labels start as unsure.
Label PNGs: 8-bit, same size as maps/<name>/maps_2d/<name>.pgm, 0 = not occupied, 1 = keep, 2 = erase, 3 = unsure.
Save writes <name>_label.png atomically (temp file + rename) after copying the old one to <name>_label.bak.png.
Needs only numpy and OpenCV (stdlib HTTP server, no CDN).

Second toolbar (needs ml/map_clean, and PyTorch for the ML buttons): a method marks the whole map keep / erase
(without ML: speckle filter, session agreement, ROSE2 from a file; with ML: the erase-only models), you correct the
marks by hand, then the clean-up buttons (cut door leak, fill unseen patches, straighten) and "Show result" give the
map those marks produce, scored against the saved hand labels. "Export map" writes it to data/map_cleaning/exports/.
"""
import argparse
import base64
import json
import os
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = MAPS = ''
DESKS, DESK_SIZE = 4, None
METHOD_LOCK = threading.Lock()                # one model call at a time
SAVE_LOCK = threading.Lock()


def pgm_path(n):
    return os.path.join(MAPS, n, 'maps_2d', n + '.pgm')


def lab_path(n, suffix):
    return os.path.join(LABELS, n + suffix + '.png')


def list_maps():
    if not os.path.isdir(LABELS):
        return []
    names = [f[:-len('_prelabel.png')] for f in os.listdir(LABELS) if f.endswith('_prelabel.png')]
    return sorted(n for n in names if n and os.path.isfile(pgm_path(n)))


def classes(n):
    """0 free, 1 unknown, 2 occupied"""
    img = cv2.imread(pgm_path(n), cv2.IMREAD_GRAYSCALE)
    c = np.ones(img.shape, np.uint8)
    c[img < 100], c[img > 250] = 2, 0
    return c


def load_labels(n, cls):
    occ = cls == 2
    lab, src = np.zeros(cls.shape, np.uint8), 'none'
    for suffix in ('_label', '_prelabel'):
        p = lab_path(n, suffix)
        if not os.path.exists(p):
            continue
        a = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if a is None or a.ndim != 2 or a.shape != cls.shape:
            print(f'warning: {p} is unreadable or not {cls.shape[1]}x{cls.shape[0]} single-channel, ignored')
            continue
        lab, src = a.astype(np.uint8), suffix[1:]
        break
    lab = np.where(occ, np.where((lab >= 1) & (lab <= 3), lab, 3), 0).astype(np.uint8)
    return lab, src


def methods():
    """ml/map_clean/editor_methods.py, or None when it cannot be imported (the labelling itself still works)."""
    try:
        if ROOT not in sys.path:
            sys.path.insert(0, ROOT)
        from ml.map_clean import editor_methods
        return editor_methods
    except Exception as e:                                   # noqa: BLE001 - any import problem just hides the buttons
        print('method buttons off:', type(e).__name__, e)
        return None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype='application/json', code=200):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def name(self):
        n = parse_qs(urlparse(self.path).query).get('name', [''])[0]
        return n if n in list_maps() else None

    def do_GET(self):
        p = urlparse(self.path).path
        if p == '/':
            return self.send(PAGE.encode(), 'text/html; charset=utf-8')
        if p == '/api/maps':
            return self.send(list_maps())
        if p == '/api/map':
            n = self.name()
            if n is None:
                return self.send({'error': 'no such map'}, code=404)
            cls = classes(n)
            lab, src = load_labels(n, cls)
            return self.send({'name': n, 'w': cls.shape[1], 'h': cls.shape[0], 'source': src,
                              'base': base64.b64encode(cls.tobytes()).decode(),
                              'label': base64.b64encode(lab.tobytes()).decode()})
        if p in ('/api/methods', '/api/propose'):
            n, em = self.name(), methods()
            if n is None or em is None:
                return self.send([] if p == '/api/methods' else {'ok': False, 'msg': 'methods are not available'})
            if p == '/api/methods':
                return self.send(em.available(n, LABELS))
            try:
                cls = classes(n)
                with METHOD_LOCK:
                    lab, msg = em.propose(n, parse_qs(urlparse(self.path).query).get('method', [''])[0], cls, LABELS)
                    out, _ = em.result(cls, lab, [])
                    sc = em.score_text(n, out, cls, LABELS)
                return self.send({'ok': True, 'label': base64.b64encode(lab.tobytes()).decode(), 'msg': msg, 'score': sc})
            except Exception as e:                           # noqa: BLE001
                return self.send({'ok': False, 'msg': f'{type(e).__name__}: {e}'})
        self.send({'error': 'not found'}, code=404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in ('/api/save', '/api/result', '/api/export'):
            return self.send({'ok': False, 'msg': 'not found'}, code=404)
        n = self.name()
        if n is None:
            return self.send({'ok': False, 'msg': 'no such map'}, code=404)
        cls = classes(n)
        size = int(self.headers.get('Content-Length') or 0)
        if size != cls.size:
            return self.send({'ok': False, 'msg': f'expected {cls.size} bytes, got {size}'}, code=400)
        a = np.frombuffer(self.rfile.read(size), np.uint8).reshape(cls.shape)
        if a.max() > 3:
            return self.send({'ok': False, 'msg': 'label values must be 0-3'}, code=400)
        if path != '/api/save':                              # the map the current marks give, after the chosen clean-up steps
            em = methods()
            if em is None:
                return self.send({'ok': False, 'msg': 'methods are not available'})
            try:
                steps = [t for t in parse_qs(urlparse(self.path).query).get('steps', [''])[0].split(',') if t]
                lab = np.where(cls == 2, a, 0).astype(np.uint8)
                with METHOD_LOCK:
                    out, msg = em.result(cls, lab, steps, DESKS, DESK_SIZE)
                    sc = em.score_text(n, out, cls, LABELS)
                    if path == '/api/export':
                        msg = 'Exported ' + em.export(n, out, pgm_path(n)[:-4] + '.yaml') + '. ' + msg
                return self.send({'ok': True, 'cls': base64.b64encode(em.to_labeler(out).tobytes()).decode(), 'msg': msg, 'score': sc})
            except Exception as e:                           # noqa: BLE001
                return self.send({'ok': False, 'msg': f'{type(e).__name__}: {e}'})
        out = np.where(cls == 2, a, 0).astype(np.uint8)
        dst, bak, tmp = lab_path(n, '_label'), lab_path(n, '_label.bak'), lab_path('.' + n, '_label.tmp')
        with SAVE_LOCK:
            if not cv2.imwrite(tmp, out) or not np.array_equal(cv2.imread(tmp, cv2.IMREAD_UNCHANGED), out):
                return self.send({'ok': False, 'msg': 'could not write ' + tmp}, code=500)
            if os.path.exists(dst):
                shutil.copy2(dst, bak)
            os.replace(tmp, dst)
        k = [int((out == v).sum()) for v in (1, 2, 3)]
        self.send({'ok': True, 'msg': f'Saved {dst} (keep {k[0]}, erase {k[1]}, unsure {k[2]})'})


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Map labeler</title><style>
html,body{margin:0;height:100%}
body{font:14px system-ui,sans-serif;background:#161b26;color:#e6edf7;display:flex;flex-direction:column}
.bar{background:#1f2636;padding:6px 10px;display:flex;flex-wrap:wrap;gap:8px;align-items:center;border-bottom:1px solid #2c3550}
#bar2{background:#1a2132;row-gap:6px}.grp{display:inline-flex;flex-wrap:wrap;gap:6px;align-items:center;margin-right:10px}#bar2 b{color:#9fb0cc;font-weight:600}#score{color:#7ee0a4}button:disabled{opacity:.5}
button,select,input{font:inherit;padding:4px 9px;border-radius:6px;border:1px solid #3a4560;background:#2a3348;color:inherit}
button.on{background:#3d7bfd;border-color:#3d7bfd}
.sw{display:inline-block;width:12px;height:12px;border:1px solid #888;vertical-align:-1px;margin-right:4px}
.sep{width:1px;height:22px;background:#3a4560}
#dirty{color:#ffb020;font-weight:600}#msg{color:#9fb0cc}
#help{padding:4px 10px;color:#8a98b3;font-size:12px;background:#1a2030}
#c{flex:1;min-height:0;width:100%;display:block;touch-action:none;cursor:crosshair}
</style></head><body>
<div class="bar">
<label>Map <select id="map"></select></label><span id="dirty"></span>
<button id="save">Save</button><button id="undo">Undo</button><button id="redo">Redo</button>
<span class="sep"></span>
<button class="cls" data-v="1"><span class="sw" style="background:#191919"></span>Keep (K)</button>
<button class="cls" data-v="2"><span class="sw" style="background:#e62828"></span>Erase (E)</button>
<button class="cls" data-v="3"><span class="sw" style="background:#ff9600"></span>Unsure (U)</button>
<span class="sep"></span>
<button class="tool" data-t="brush">Brush (B)</button><button class="tool" data-t="box">Box (X)</button><button class="tool" data-t="poly">Polygon (P)</button>
<label>Size <input id="size" type="range" min="1" max="40" value="3" style="width:100px;padding:0"> <span id="sizev">3</span></label>
<label><input id="round" type="checkbox"> round</label>
<span class="sep"></span>
<button id="next">Next unsure (N)</button><button id="hide">Hide labels (H)</button><button id="fit">Fit</button>
<span id="counts"></span>
</div>
<div class="bar" id="bar2">
<span class="grp"><b>1 Without ML</b><span id="m_noml" class="grp"></span></span>
<span class="grp"><b>2 With ML</b><span id="m_ml" class="grp"></span></span>
<span class="grp"><b>3 Clean-up</b><button class="stepb" data-s="leak">Cut door leak</button><button class="stepb" data-s="straighten">Straighten</button><button class="stepb" data-s="fill">Fill unseen</button>
<span class="sep"></span><button id="result">Show result (R)</button><button id="export">Export map</button></span>
<span id="score" style="flex-basis:100%"></span>
</div>
<div id="help">Wheel zoom, right-drag or Space+drag pan. K/E/U class, B brush (Shift = straight line), X box, P polygon (click points, Enter or double-click closes, Esc cancels), [ ] brush size, N next unsure cluster, H hide labels, R result view, Ctrl+Z undo, Ctrl+Y / Ctrl+Shift+Z redo, Ctrl+S save. Only occupied cells change. <span id="msg"></span></div>
<canvas id="c"></canvas>
<script>
const $=id=>document.getElementById(id),cv=$('c'),g=cv.getContext('2d'),off=document.createElement('canvas'),og=off.getContext('2d');
const PAL=[null,[25,25,25],[230,40,40],[255,150,0]],NAMES=['','keep','erase','unsure'];
let W=0,H=0,base=null,lab=null,img=null,cur='',cls=1,tool='brush',bsize=3,hide=false,resView=false,res=null,resVer=-1,resSteps='',busy=false;
let s=1,ox=0,oy=0,dpr=1,undo=[],redo=[],sid=0,savedId=0,stroke=null,dr=null,down=null,last=null,axis=null,pan=null,space=false,mouse=null,poly=[],boxA=null,hl=null,ver=0,clu=[],cluVer=-1,cluI=-1;
const cnt=[0,0,0,0];
const msg=t=>$('msg').textContent=t;
const topId=()=>undo.length?undo[undo.length-1].id:0;
function isDirty(){return W>0&&topId()!=savedId}
function showState(){$('dirty').textContent=isDirty()?'● unsaved changes':'';$('counts').textContent=`keep ${cnt[1]}  erase ${cnt[2]}  unsure ${cnt[3]}`}
function paint(i){const o=i*4,b=resView&&res?res[i]:base[i];let c;
  if(resView&&res)c=b==0?[255,255,255]:b==1?[205,205,205]:[0,0,0];else
  if(b==0)c=[255,255,255];else if(b==1)c=[205,205,205];else c=hide?[0,0,0]:(PAL[lab[i]]||PAL[3]);
  img.data[o]=c[0];img.data[o+1]=c[1];img.data[o+2]=c[2];img.data[o+3]=255}
function repaintAll(){for(let i=0;i<W*H;i++)paint(i);og.putImageData(img,0,0);draw()}
function touch(i){const x=i%W,y=(i-x)/W;if(!dr)dr=[x,y,x,y];else{dr[0]=Math.min(dr[0],x);dr[1]=Math.min(dr[1],y);dr[2]=Math.max(dr[2],x);dr[3]=Math.max(dr[3],y)}}
function flush(){if(dr){og.putImageData(img,0,0,dr[0],dr[1],dr[2]-dr[0]+1,dr[3]-dr[1]+1);dr=null}draw();showState()}
function put(i,v){cnt[lab[i]]--;cnt[v]++;lab[i]=v;paint(i);touch(i)}
function setCell(x,y){if(x<0||y<0||x>=W||y>=H)return;const i=y*W+x;if(base[i]!=2||lab[i]==cls)return;if(!stroke.has(i))stroke.set(i,lab[i]);put(i,cls)}
function begin(){stroke=new Map()}
function end(){if(stroke&&stroke.size){const idx=Int32Array.from(stroke.keys()),old=Uint8Array.from(stroke.values()),nw=new Uint8Array(idx.length);
    for(let k=0;k<idx.length;k++)nw[k]=lab[idx[k]];undo.push({id:++sid,idx,old,nw});if(undo.length>200)undo.shift();redo=[];ver++}
  stroke=null;flush()}
function step(st,back){const v=back?st.old:st.nw;for(let k=0;k<st.idx.length;k++)put(st.idx[k],v[k]);ver++;flush()}
function doUndo(){if(stroke)return;leaveResult();const st=undo.pop();if(!st)return msg('Nothing to undo.');step(st,true);redo.push(st)}
function doRedo(){if(stroke)return;leaveResult();const st=redo.pop();if(!st)return msg('Nothing to redo.');step(st,false);undo.push(st)}
// brush: the cells one dab covers (same rule as the map editor: square, or round from size 3 up)
function stamp(cx,cy){const h=(bsize-1)/2,bx=Math.round(cx-h),by=Math.round(cy-h),rnd=$('round').checked;
  for(let dy=0;dy<bsize;dy++)for(let dx=0;dx<bsize;dx++){const ex=dx-h,ey=dy-h;if(!rnd||bsize<3||ex*ex+ey*ey<=h*h+.5)setCell(bx+dx,by+dy)}}
function seg(a,b,c2,d){const n=Math.max(Math.abs(c2-a),Math.abs(d-b),1);for(let i=0;i<=n;i++)stamp(a+(c2-a)*i/n,b+(d-b)*i/n)}
function fillBox(a,b){begin();for(let y=Math.min(a[1],b[1]);y<=Math.max(a[1],b[1]);y++)for(let x=Math.min(a[0],b[0]);x<=Math.max(a[0],b[0]);x++)setCell(x,y);end()}
function closePoly(){if(poly.length<3){poly=[];draw();return msg('A polygon needs at least 3 points.')}
  const P=poly.map(p=>[p[0]+.5,p[1]+.5]),xs=P.map(p=>p[0]),ys=P.map(p=>p[1]);begin();
  for(let y=Math.max(0,Math.floor(Math.min(...ys)));y<=Math.min(H-1,Math.ceil(Math.max(...ys)));y++)
    for(let x=Math.max(0,Math.floor(Math.min(...xs)));x<=Math.min(W-1,Math.ceil(Math.max(...xs)));x++){const px=x+.5,py=y+.5;let ins=false;
      for(let i=0,j=P.length-1;i<P.length;j=i++){const[xi,yi]=P[i],[xj,yj]=P[j];if((yi>py)!=(yj>py)&&px<(xj-xi)*(py-yi)/(yj-yi)+xi)ins=!ins}
      if(ins)setCell(x,y)}
  poly=[];end()}
// view: screen = cell * s + (ox, oy), in CSS pixels
function resize(){dpr=window.devicePixelRatio||1;cv.width=Math.round(cv.clientWidth*dpr);cv.height=Math.round(cv.clientHeight*dpr);draw()}
function fit(){if(!W)return;s=Math.min(cv.clientWidth/W,cv.clientHeight/H)*.95;ox=(cv.clientWidth-W*s)/2;oy=(cv.clientHeight-H*s)/2;draw()}
function cell(e){const r=cv.getBoundingClientRect();return[Math.floor((e.clientX-r.left-ox)/s),Math.floor((e.clientY-r.top-oy)/s)]}
function draw(){g.setTransform(1,0,0,1,0,0);g.fillStyle='#10141d';g.fillRect(0,0,cv.width,cv.height);if(!W)return;
  g.setTransform(dpr,0,0,dpr,0,0);g.imageSmoothingEnabled=false;g.drawImage(off,ox,oy,W*s,H*s);g.lineWidth=1.5;
  if(hl){g.setLineDash([6,4]);g.strokeStyle='#22d3ee';g.strokeRect(ox+(hl.x0-1)*s,oy+(hl.y0-1)*s,(hl.x1-hl.x0+3)*s,(hl.y1-hl.y0+3)*s);g.setLineDash([])}
  const q=mouse;g.strokeStyle='#3d7bfd';
  if(tool=='brush'&&q&&!pan){const h=(bsize-1)/2,bx=Math.round(q[0]-h),by=Math.round(q[1]-h);g.beginPath();
    if($('round').checked&&bsize>=3)g.arc(ox+(bx+bsize/2)*s,oy+(by+bsize/2)*s,bsize/2*s,0,7);else g.rect(ox+bx*s,oy+by*s,bsize*s,bsize*s);
    g.stroke();g.strokeStyle='rgba(255,255,255,.6)';g.stroke()}
  if(tool=='box'&&boxA&&q){const x0=Math.min(boxA[0],q[0]),y0=Math.min(boxA[1],q[1]);g.fillStyle='rgba(61,123,253,.25)';
    g.fillRect(ox+x0*s,oy+y0*s,(Math.abs(q[0]-boxA[0])+1)*s,(Math.abs(q[1]-boxA[1])+1)*s);g.strokeRect(ox+x0*s,oy+y0*s,(Math.abs(q[0]-boxA[0])+1)*s,(Math.abs(q[1]-boxA[1])+1)*s)}
  if(tool=='poly'&&poly.length){const L=p=>[ox+(p[0]+.5)*s,oy+(p[1]+.5)*s];g.beginPath();poly.forEach((p,i)=>g[i?'lineTo':'moveTo'](...L(p)));if(q)g.lineTo(...L(q));
    g.strokeStyle='#22d3ee';g.stroke();g.fillStyle='#22d3ee';for(const p of poly){const[a,b]=L(p);g.fillRect(a-3,b-3,6,6)}}}
cv.addEventListener('contextmenu',e=>e.preventDefault());
cv.addEventListener('wheel',e=>{e.preventDefault();if(!W)return;const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top,
  s2=Math.max(.25,Math.min(64,s*(e.deltaY<0?1.2:1/1.2)));ox=mx-(mx-ox)*s2/s;oy=my-(my-oy)*s2/s;s=s2;mouse=cell(e);draw()},{passive:false});
cv.onpointerdown=e=>{if(!W)return;cv.setPointerCapture(e.pointerId);
  if(e.button==2||e.button==1||(e.button==0&&space)){pan={x:e.clientX,y:e.clientY,ox,oy};cv.style.cursor='grabbing';return}
  if(e.button!=0)return;if(resView)return msg('Result view: press R to go back to the marks before editing.');const q=cell(e);hl=null;
  if(tool=='brush'){begin();down=last=q;axis=null;stamp(...q);flush()}
  else if(tool=='box'){boxA=q;draw()}
  else if(tool=='poly'){const p=poly[poly.length-1];if(!p||p[0]!=q[0]||p[1]!=q[1])poly.push(q);draw()}};
cv.onpointermove=e=>{if(pan){ox=pan.ox+e.clientX-pan.x;oy=pan.oy+e.clientY-pan.y;draw();return}if(!W)return;const q=cell(e);
  // Shift while brushing: keep the stroke on the row or column it started on
  if(tool=='brush'&&down&&e.shiftKey){if(!axis&&(q[0]!=down[0]||q[1]!=down[1]))axis=Math.abs(q[0]-down[0])>=Math.abs(q[1]-down[1])?'h':'v';if(axis=='h')q[1]=down[1];if(axis=='v')q[0]=down[0]}
  mouse=q;if(tool=='brush'&&down&&stroke){seg(...last,...q);last=q;flush()}else draw()};
function up(e){if(pan){pan=null;cv.style.cursor=space?'grab':'crosshair';return}
  if(tool=='brush'&&stroke){end();down=null}
  if(tool=='box'&&boxA){fillBox(boxA,cell(e));boxA=null}}
cv.onpointerup=up;cv.onpointercancel=up;
cv.onpointerleave=()=>{if(!pan&&!down&&!boxA){mouse=null;draw()}};
cv.ondblclick=()=>{if(tool=='poly')closePoly()};
function setCls(v){cls=v;document.querySelectorAll('.cls').forEach(b=>b.classList.toggle('on',+b.dataset.v==v))}
function setTool(t){if(stroke)end();tool=t;poly=[];boxA=null;document.querySelectorAll('.tool').forEach(b=>b.classList.toggle('on',b.dataset.t==t));draw()}
function setSize(z){bsize=Math.max(1,Math.min(40,z));$('size').value=bsize;$('sizev').textContent=bsize;draw()}
function toggleHide(){hide=!hide;$('hide').classList.toggle('on',hide);if(W)repaintAll()}
// 8-connected clusters of unsure cells, largest first; recomputed after any label change
function clusters(){const n=W*H,seen=new Uint8Array(n),st=new Int32Array(n),out=[];
  for(let i=0;i<n;i++){if(lab[i]!=3||seen[i])continue;let top=0,c=0,x0=W,y0=H,x1=-1,y1=-1;st[top++]=i;seen[i]=1;
    while(top){const j=st[--top],x=j%W,y=(j-x)/W;c++;if(x<x0)x0=x;if(x>x1)x1=x;if(y<y0)y0=y;if(y>y1)y1=y;
      for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){const a=x+dx,b=y+dy;if(a<0||b<0||a>=W||b>=H)continue;const k=b*W+a;if(!seen[k]&&lab[k]==3){seen[k]=1;st[top++]=k}}}
    out.push({n:c,x0,y0,x1,y1})}
  return out.sort((a,b)=>b.n-a.n)}
function nextUnsure(){if(!W)return;if(cluVer!=ver){clu=clusters();cluVer=ver;cluI=-1}
  if(!clu.length){hl=null;draw();return msg('No unsure cells left.')}
  cluI=(cluI+1)%clu.length;const c=clu[cluI],bw=c.x1-c.x0+1,bh=c.y1-c.y0+1;
  s=Math.max(2,Math.min(24,Math.min(cv.clientWidth/(bw+20),cv.clientHeight/(bh+20))));
  ox=cv.clientWidth/2-(c.x0+bw/2)*s;oy=cv.clientHeight/2-(c.y0+bh/2)*s;hl=c;draw();
  msg(`Unsure cluster ${cluI+1} of ${clu.length}: ${c.n} cells.`)}
// ---- method buttons, clean-up steps and the result view
const steps=()=>[...document.querySelectorAll('.stepb.on')].map(b=>b.dataset.s).join(',');
function lock(v){busy=v;document.querySelectorAll('#bar2 button').forEach(b=>b.disabled=v)}
function setAll(nw){begin();for(let i=0;i<W*H;i++){if(base[i]!=2||lab[i]==nw[i]||nw[i]<1)continue;stroke.set(i,lab[i]);put(i,nw[i])}end()}
function leaveResult(){if(resView){resView=false;$('result').classList.remove('on');repaintAll()}}
async function applyMethod(id,btn){if(!W||busy)return;if(stroke)end();leaveResult();lock(true);msg('Running '+btn.textContent+'...');
  try{const d=await(await fetch('/api/propose?name='+encodeURIComponent(cur)+'&method='+id)).json();
    if(!d.ok){msg(d.msg)}else{setAll(b64(d.label));document.querySelectorAll('.meth').forEach(b=>b.classList.toggle('on',b==btn));msg(d.msg+' One undo step.');$('score').textContent=d.score}}
  catch(e){msg('Failed: '+e.message)}lock(false)}
async function showResult(exp){if(!W||busy)return;if(stroke)end();lock(true);msg(exp?'Exporting...':'Building the result...');
  try{const d=await(await fetch('/api/'+(exp?'export':'result')+'?name='+encodeURIComponent(cur)+'&steps='+steps(),{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:lab})).json();
    if(!d.ok){msg(d.msg)}else{res=b64(d.cls);resVer=ver;resSteps=steps();resView=true;$('result').classList.add('on');repaintAll();msg(d.msg);$('score').textContent=d.score}}
  catch(e){msg('Failed: '+e.message)}lock(false)}
function toggleResult(){if(resView)return leaveResult();showResult(false)}
async function loadMethods(){const l=await(await fetch('/api/methods?name='+encodeURIComponent(cur))).json();
  for(const gp of['noml','ml']){const box=$('m_'+gp);box.innerHTML='';for(const m of l.filter(x=>x.group==gp)){const b=document.createElement('button');b.className='meth';b.textContent=m.label;b.onclick=()=>{applyMethod(m.id,b);b.blur()};box.appendChild(b)}
    if(!box.children.length)box.textContent=' not available '}}
function b64(t){const b=atob(t),a=new Uint8Array(b.length);for(let i=0;i<b.length;i++)a[i]=b.charCodeAt(i);return a}
async function load(n){msg('Loading '+n+'...');const r=await fetch('/api/map?name='+encodeURIComponent(n));if(!r.ok)throw new Error('Could not load '+n);const d=await r.json();
  W=d.w;H=d.h;base=b64(d.base);lab=b64(d.label);off.width=W;off.height=H;img=new ImageData(W,H);cur=n;
  cnt.fill(0);for(let i=0;i<W*H;i++)if(base[i]==2)cnt[lab[i]]++;
  undo=[];redo=[];savedId=0;stroke=null;down=null;poly=[];boxA=null;hl=null;ver++;cluVer=-1;resView=false;res=null;$('result').classList.remove('on');$('score').textContent='';
  repaintAll();fit();showState();loadMethods().catch(()=>{});msg(`${n}: ${W} x ${H} cells, labels from ${d.source=='label'?n+'_label.png':d.source=='prelabel'?n+'_prelabel.png':'nothing (all unsure)'}.`)}
async function save(){if(!W)return;if(stroke)end();const id=topId();msg('Saving...');
  try{const r=await(await fetch('/api/save?name='+encodeURIComponent(cur),{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:lab})).json();
    msg(r.msg);if(r.ok){savedId=id;showState()}}catch(e){msg('Save failed: '+e.message)}}
$('map').onchange=async()=>{const n=$('map').value;$('map').blur();
  if(isDirty()&&!confirm('Discard the unsaved changes on '+cur+'?')){$('map').value=cur;return}
  try{await load(n)}catch(e){msg(e.message);$('map').value=cur}};
window.addEventListener('beforeunload',e=>{if(isDirty()){e.preventDefault();e.returnValue=''}});
window.addEventListener('resize',resize);
document.querySelectorAll('.cls').forEach(b=>b.onclick=()=>{setCls(+b.dataset.v);b.blur()});
document.querySelectorAll('.tool').forEach(b=>b.onclick=()=>{setTool(b.dataset.t);b.blur()});
$('size').oninput=()=>setSize(+$('size').value);$('round').onchange=()=>{$('round').blur();draw()};
document.querySelectorAll('.stepb').forEach(b=>b.onclick=()=>{b.classList.toggle('on');b.blur();showResult(false)});$('result').onclick=toggleResult;$('export').onclick=()=>showResult(true);
$('save').onclick=save;$('undo').onclick=doUndo;$('redo').onclick=doRedo;$('next').onclick=nextUnsure;$('hide').onclick=toggleHide;$('fit').onclick=fit;
document.querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>b.blur()));
document.addEventListener('keydown',e=>{if(e.target.tagName=='SELECT')return;const k=e.key.toLowerCase(),m=e.ctrlKey||e.metaKey;
  if(m){if(k=='z'){e.preventDefault();e.shiftKey?doRedo():doUndo()}else if(k=='y'){e.preventDefault();doRedo()}else if(k=='s'){e.preventDefault();save()}return}
  if(e.altKey)return;
  if(e.key==' '){e.preventDefault();if(!space){space=true;if(!pan)cv.style.cursor='grab'}return}
  const act={k:()=>setCls(1),e:()=>setCls(2),u:()=>setCls(3),b:()=>setTool('brush'),x:()=>setTool('box'),p:()=>setTool('poly'),
    '[':()=>setSize(bsize-1),']':()=>setSize(bsize+1),h:toggleHide,n:nextUnsure,r:toggleResult,enter:()=>{if(tool=='poly')closePoly()},
    escape:()=>{poly=[];boxA=null;hl=null;draw()}}[k];
  if(act){e.preventDefault();act()}});
document.addEventListener('keyup',e=>{if(e.key==' '){e.preventDefault();space=false;if(!pan)cv.style.cursor='crosshair'}});
window.addEventListener('blur',()=>{space=false});
setCls(1);setTool('brush');setSize(3);resize();
fetch('/api/maps').then(r=>r.json()).then(l=>{$('map').innerHTML=l.map(n=>`<option>${n}</option>`).join('');
  if(!l.length)return msg('No <name>_prelabel.png found in the labels directory.');$('map').value=l[0];load(l[0]).catch(e=>msg(e.message))});
</script></body></html>"""


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--labels', default=os.path.join(ROOT, 'data', 'map_cleaning', 'labels'))
    ap.add_argument('--maps', default=os.path.join(ROOT, 'maps'))
    ap.add_argument('--port', type=int, default=8507)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--desks', type=int, default=4, help='identical desks the Straighten button redraws (0 = none)')
    ap.add_argument('--desk-size', type=float, nargs=2, default=None, help='desk size in metres, long and short side; default: measured from the map')
    a = ap.parse_args()
    LABELS, MAPS = os.path.abspath(a.labels), os.path.abspath(a.maps)
    DESKS, DESK_SIZE = a.desks, a.desk_size
    print(f'Map labeler: http://{a.host}:{a.port}/   (labels in {LABELS}, maps in {MAPS}, {len(list_maps())} maps)', flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()
