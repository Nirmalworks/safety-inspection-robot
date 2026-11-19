#!/usr/bin/env python3
"""Interactive clean-up of a saved 2D map (Nav2 .pgm/.yaml) in the browser.

    scripts/map_editor.py [DATA_DIR] [PORT]        # default $AME505_DATA (the SSD on the robot), port 8506
    source scripts/sim_env.sh && scripts/map_editor.py     # on a laptop: maps in ~/ame505_sim_data
    open http://localhost:8506

Meant for a laptop: lining up two maps takes 15-20 s of full CPU, too much for the Nano during a run.

One map view: the map you chose, shown and edited as it is. Optionally choose a second map to compare
with: it is rotated and shifted to match, its walls can be overlaid in red, the cells where the two
differ in orange, and it is shown read-only below. "Use second map here" copies the selected area (or
every orange cell) from it; "Restore original here" puts the saved map back.
Tools: brush (Shift = straight), line, box, box / polygon select, fill, copy / paste (rotate, flip),
undo, zoom, and "Rotate whole map" (turns checkpoints and home too).
Save writes a NEW map name in the edited map's frame; the map you loaded is never overwritten and its
checkpoints, home and landmarks go with it (scripts/backup_map.sh then copies all of it to maps/).
Needs OpenCV, PyYAML and SciPy (SciPy only for aligning the second map). No ROS, no Flask.
"""
import json
import os
import re
import shutil
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import math

import cv2
import numpy as np
import yaml

DATA = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else os.environ.get('AME505_DATA', '/home/ubuntu/share/tmp/ame505_data'))
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8506
MAPS, CPS, HOME, LMK = (os.path.join(DATA, d) for d in ('maps_2d', 'checkpoints', 'home', 'landmarks'))
NAME = re.compile(r'^[A-Za-z0-9_\-]+$')


def meta(name):
    return yaml.safe_load(open(os.path.join(MAPS, name + '.yaml')))


def checkpoints(name):
    p = os.path.join(CPS, name + '.yaml')
    return (yaml.safe_load(open(p)) or {}).get('checkpoints', []) if os.path.exists(p) else []


def home(name):
    p = os.path.join(HOME, name + '.yaml')
    h = (yaml.safe_load(open(p)) or {}) if os.path.exists(p) else {}
    return {'x': float(h.get('x', 0.0)), 'y': float(h.get('y', 0.0)), 'yaw': float(h.get('yaw', 0.0))}


# ---- rotating the WHOLE map by `deg` (counter-clockwise) about the map origin (0, 0) ----
# The image, the yaml origin, the checkpoints and the home pose all turn together, so the saved map
# describes the same room and the same robot poses, only in a turned map frame.
def rot_geom(name, deg):
    m = meta(name)
    r, (ox, oy) = m['resolution'], m['origin'][:2]
    h, w = cv2.imread(os.path.join(MAPS, m['image']), cv2.IMREAD_GRAYSCALE).shape
    th = math.radians(deg)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    c = np.array([[ox, oy], [ox + w * r, oy], [ox, oy + h * r], [ox + w * r, oy + h * r]]) @ R.T
    x0, y0 = c.min(0)
    return R, float(x0), float(y0), int(math.ceil((c[:, 0].max() - x0) / r - 1e-6)), int(math.ceil((c[:, 1].max() - y0) / r - 1e-6))


def rotate_img(img, name, deg):
    """img is on the grid of map `name`; returns it on that map's rotated grid (nearest cell, no blending)"""
    if deg % 360 == 0:
        return img
    R, x0, y0, W_, H_ = rot_geom(name, deg)
    m = meta(name)
    r, (ox, oy) = m['resolution'], m['origin'][:2]
    cols, rows = np.meshgrid(np.arange(W_), np.arange(H_))
    P = np.stack([x0 + (cols + 0.5) * r, y0 + (H_ - rows - 0.5) * r], -1) @ R        # new frame -> old frame
    return cv2.remap(img, ((P[..., 0] - ox) / r - 0.5).astype(np.float32),
                     (img.shape[0] - (P[..., 1] - oy) / r - 0.5).astype(np.float32),
                     cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=205)


def rot_pose(p, name, deg):
    if deg % 360 == 0:
        return dict(p)
    R = rot_geom(name, deg)[0]
    x, y = R @ [p['x'], p['y']]
    yaw = p.get('yaw', 0.0) + math.radians(deg)
    return {**p, 'x': round(float(x), 3), 'y': round(float(y), 3), 'yaw': round(math.atan2(math.sin(yaw), math.cos(yaw)), 4)}


def wall_points(name):
    m = meta(name)
    img = cv2.imread(os.path.join(MAPS, m['image']), cv2.IMREAD_GRAYSCALE)
    r, (ox, oy) = m['resolution'], m['origin'][:2]
    ys, xs = np.nonzero(img < 100)
    return np.c_[ox + (xs + 0.5) * r, oy + (img.shape[0] - ys - 0.5) * r]


def align(A, B):
    """Rigid transform taking wall points A onto B: coarse search (3 deg, 0.5 m) + trimmed ICP.
    Same method as docs/progress/compare_maps.py in the robot repo."""
    from scipy.spatial import cKDTree
    rot = lambda th: np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    tb = cKDTree(B)
    sub, ca, cb = A[::4], A.mean(0), B.mean(0)
    best = (np.inf, None, None)
    for deg in range(0, 360, 3):
        R = rot(math.radians(deg))
        t0 = cb - R @ ca
        for dx in np.arange(-3, 3.01, 0.5):
            for dy in np.arange(-3, 3.01, 0.5):
                t = t0 + [dx, dy]
                sc = np.mean(np.minimum(tb.query(sub @ R.T + t)[0], 0.5))
                if sc < best[0]:
                    best = (sc, R, t)
    _, R, t = best
    for _ in range(60):
        d, i = tb.query(A @ R.T + t)
        k = d < np.percentile(d, 80)
        P, M = A[k], B[i[k]]
        pm, mm = P.mean(0), M.mean(0)
        U, _, Vt = np.linalg.svd((P - pm).T @ (M - mm))
        R = Vt.T @ np.diag([1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        t = mm - R @ pm
    return R, t


ALIGNED = {}


def aligned_png(name, to):
    """map `name` redrawn on the grid of map `to` (same size, origin and orientation)"""
    key = (name, to, os.path.getmtime(os.path.join(MAPS, name + '.pgm')), os.path.getmtime(os.path.join(MAPS, to + '.pgm')))
    if key not in ALIGNED:
        R, t = align(wall_points(name), wall_points(to))
        ma, mb = meta(name), meta(to)
        A = cv2.imread(os.path.join(MAPS, ma['image']), cv2.IMREAD_GRAYSCALE)
        H_, W_ = cv2.imread(os.path.join(MAPS, mb['image']), cv2.IMREAD_GRAYSCALE).shape
        r = mb['resolution']
        cols, rows = np.meshgrid(np.arange(W_), np.arange(H_))
        P = np.stack([mb['origin'][0] + (cols + 0.5) * r, mb['origin'][1] + (H_ - rows - 0.5) * r], -1)
        PA = (P - t) @ R
        ra = ma['resolution']
        out = cv2.remap(A, ((PA[..., 0] - ma['origin'][0]) / ra - 0.5).astype(np.float32),
                        (A.shape[0] - (PA[..., 1] - ma['origin'][1]) / ra - 0.5).astype(np.float32),
                        cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=205)
        ALIGNED[key] = cv2.imencode('.png', out)[1].tobytes()
    return ALIGNED[key]


class H(BaseHTTPRequestHandler):
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

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        name = q.get('name', [''])[0]
        if u.path == '/':
            return self.send(PAGE.encode(), 'text/html; charset=utf-8')
        if u.path == '/api/maps':
            return self.send(sorted(f[:-5] for f in os.listdir(MAPS) if f.endswith('.yaml')))
        if not NAME.match(name) or not os.path.exists(os.path.join(MAPS, name + '.yaml')):
            return self.send({'error': 'no such map'}, code=404)
        m = meta(name)
        rot = float(q.get('rot', ['0'])[0] or 0)
        if u.path == '/api/info':
            org = list(rot_geom(name, rot)[1:3]) if rot % 360 else m['origin'][:2]
            return self.send({'resolution': m['resolution'], 'origin': org, 'home': rot_pose(home(name), name, rot),
                              'checkpoints': [rot_pose(c, name, rot) for c in checkpoints(name)]})
        if u.path == '/api/png':
            to = q.get('align_to', [''])[0]
            if to and to != name and NAME.match(to) and os.path.exists(os.path.join(MAPS, to + '.yaml')):
                img, grid = cv2.imdecode(np.frombuffer(aligned_png(name, to), np.uint8), cv2.IMREAD_GRAYSCALE), to
            else:
                img, grid = cv2.imread(os.path.join(MAPS, m['image']), cv2.IMREAD_GRAYSCALE), name
            return self.send(cv2.imencode('.png', rotate_img(img, grid, rot))[1].tobytes(), 'image/png')
        self.send({'error': 'not found'}, code=404)

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        src, dst = q.get('name', [''])[0], q.get('as', [''])[0]
        if u.path != '/api/save' or not NAME.match(src) or not NAME.match(dst):
            return self.send({'ok': False, 'msg': 'Use letters, digits, _ and - in the name.'}, code=400)
        if dst == src:
            return self.send({'ok': False, 'msg': 'Choose a new name: the source map is never overwritten.'}, code=400)
        m = meta(src)
        rot = float(q.get('rot', ['0'])[0] or 0)
        turned = rot % 360 != 0
        old = cv2.imread(os.path.join(MAPS, m['image']), cv2.IMREAD_GRAYSCALE)
        want = rot_geom(src, rot)[4:2:-1] if turned else old.shape           # (rows, cols)
        img = cv2.imdecode(np.frombuffer(self.rfile.read(int(self.headers['Content-Length'])), np.uint8),
                           cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape != tuple(want):
            return self.send({'ok': False, 'msg': 'Image size does not match the source map.'}, code=400)
        out = np.full(img.shape, 205, np.uint8)             # back to the three Nav2 values
        out[img < 100], out[img > 230] = 0, 254
        cv2.imwrite(os.path.join(MAPS, dst + '.pgm'), out)
        m['image'] = dst + '.pgm'
        if turned:
            m['origin'] = [round(v, 3) for v in rot_geom(src, rot)[1:3]] + [0]
        yaml.safe_dump(m, open(os.path.join(MAPS, dst + '.yaml'), 'w'), default_flow_style=None, sort_keys=False)
        cps = checkpoints(src)
        if cps:                                             # same poses, in the (possibly turned) frame
            yaml.safe_dump({'frame_id': 'map', 'checkpoints': [rot_pose(c, src, rot) for c in cps]},
                           open(os.path.join(CPS, dst + '.yaml'), 'w'), sort_keys=False)
        if turned or os.path.exists(os.path.join(HOME, src + '.yaml')):
            os.makedirs(HOME, exist_ok=True)                # a turned map starts with the robot facing a new yaw
            yaml.safe_dump(rot_pose(home(src), src, rot), open(os.path.join(HOME, dst + '.yaml'), 'w'))
        if os.path.exists(os.path.join(LMK, src + '.yaml')):             # ArUco landmark poses (landmarks.py) are in the map frame too
            lm = yaml.safe_load(open(os.path.join(LMK, src + '.yaml'))) or {}
            lm['map'] = dst
            lm['markers'] = {k: rot_pose(v, src, rot) for k, v in (lm.get('markers') or {}).items()}
            yaml.safe_dump(lm, open(os.path.join(LMK, dst + '.yaml'), 'w'), sort_keys=False)
        note = f'rotated {rot:g} deg, checkpoints, home and landmarks turned with it' if turned else f'{int((out != old).sum())} cells changed'
        self.send({'ok': True, 'msg': f'Saved {dst} ({note}) in {MAPS}'})


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Map editor</title><style>
body{margin:0;font:14px system-ui,sans-serif;background:#161b26;color:#e6edf7}
.bar{background:#1f2636;padding:7px 10px;display:flex;flex-wrap:wrap;gap:8px;align-items:center;border-bottom:1px solid #2c3550}
button,select,input{font:inherit;padding:5px 9px;border-radius:6px;border:1px solid #3a4560;background:#2a3348;color:inherit}
button.on{background:#3d7bfd;border-color:#3d7bfd}
#wrap{overflow:auto;height:calc(100vh - 150px);touch-action:none}
#stack{position:relative;margin:12px;display:inline-block}
canvas{image-rendering:pixelated;display:block}#ov{position:absolute;left:0;top:0;pointer-events:none}
#msg{padding:6px 10px;color:#9fb0cc;min-height:18px}.sw{display:inline-block;width:12px;height:12px;border:1px solid #888;vertical-align:-1px;margin-right:4px}
.sep{width:1px;height:24px;background:#3a4560}
#src{display:flex;gap:10px;padding:0 12px 14px}#src>div{flex:1;min-width:0}#src canvas{width:100%;height:auto;border:1px solid #2c3550}.cap{padding:6px 2px;color:#9fb0cc}
</style></head><body>
<div class="bar">
<label>Map to edit <select id="m1"></select></label><label>Compare with <select id="m2"></select></label>
<span class="sep"></span><label title="turns the whole map, its checkpoints and home. Positive = counter-clockwise. Reloads the map: unsaved edits are lost.">Rotate whole map <input id="mrot" type="number" value="0" step="1" style="width:58px">&deg;</label><button id="mrl" title="counter-clockwise">&#8634;</button><button id="mrr" title="clockwise">&#8635;</button><select id="mstep" title="degrees per click"><option>1</option><option selected>5</option><option>15</option><option>45</option><option>90</option></select>
<span class="sep"></span>Show from the second map:
<label><input id="s_dif" type="checkbox" checked><span class="sw" style="background:#ffaa3c"></span>where it differs</label>
<label style="display:none"><input id="s_la" type="checkbox"></label>
<label><input id="s_lb" type="checkbox"><span class="sw" style="background:#eb143c"></span>its walls</label>
<input id="op" type="range" min="0.2" max="1" step="0.1" value="0.7" style="width:80px;padding:0" title="overlay strength">
<span class="sep"></span>
<button id="use1" title="With a selection: put that area back as it is in the saved map. Without: put back every orange cell.">Restore original here</button>
<button id="use2" title="With a selection: take that area from the second map. Without: fill every orange cell from the second map.">Use second map here</button>
<span class="sep"></span><label><input id="s_src" type="checkbox" checked> show the second map below</label>
</div>
<div class="bar">
<button data-v="254" class="paint on"><span class="sw" style="background:#fff"></span>Floor</button>
<button data-v="0" class="paint"><span class="sw" style="background:#000"></span>Wall</button>
<button data-v="205" class="paint"><span class="sw" style="background:#cdcdcd"></span>Unknown</button>
<span class="sep"></span>
<button id="brush" class="shape on" title="hold Shift while dragging to paint a straight horizontal or vertical line">Brush</button><button id="line" class="shape">Line</button><button id="rect" class="shape">Box</button>
<label>Size <input id="size" type="range" min="1" max="15" value="3" style="width:90px;padding:0"> <span id="sizev">3</span></label>
<span class="sep"></span>
<button id="sel" class="shape">Select</button><button id="poly" class="shape" title="click the corners; click the first corner again, double-click or press Enter to close">Polygon</button><button id="fill" title="fill the selected area with the chosen paint (Unknown = cut it out)">Fill selection</button><button id="copy">Copy</button><button id="paste">Paste</button><button id="inplace">Paste in place</button>
<button id="rl" title="rotate the selected / copied piece counter-clockwise">&#8634; Rotate</button><button id="rr" title="rotate clockwise">Rotate &#8635;</button><select id="rstep" title="degrees per click"><option>1</option><option>5</option><option selected>15</option><option>45</option><option>90</option></select><label>&deg; &nbsp;angle <input id="ang" type="number" value="0" step="1" style="width:58px">&deg;</label><button id="fh" class="flip">Flip &harr;</button><button id="fv" class="flip">Flip &varr;</button>
<label title="ticked: the unknown (grey) cells of the copy are not pasted, so what is under them stays"><input id="skip" type="checkbox"> do not paste unknown</label>
<span class="sep"></span>
<button id="undo">Undo</button><button id="zout">&minus;</button><span id="zv"></span><button id="zin">+</button>
<span class="sep"></span>
<input id="as" placeholder="save as name" size="20"><button id="save">Save</button>
</div>
<div id="msg"></div>
<div id="wrap"><div id="stack"><canvas id="c"></canvas><canvas id="ov"></canvas></div></div>
<div id="src"><div><div class="cap"></div><canvas id="cA"></canvas></div><div><div class="cap"></div><canvas id="cB"></canvas></div></div>
<script>
const $=id=>document.getElementById(id),c=$('c'),ov=$('ov'),x=c.getContext('2d',{willReadFrequently:true});
let val=254,shape='brush',zoom=4,undo=[],dirty=false,info=null,down=null,last=null,sel=null,clip=null,pasting=false,hover=null,
    fl={fh:false,fv:false},A=null,B=null,W=1,H=1,cur=['','','0'],poly=[],rot=0,axis=null;
const L={},LX={},COL={dif:[255,170,60],la:[40,110,255],lb:[235,20,60]},KEYS=['dif','la','lb'];
const msg=t=>$('msg').textContent=t,q3=v=>v<100?0:v>230?254:205;

// ---- the common map of two aligned maps (pure function: also run outside the browser as a test) ----
// A wall is common when the other map has a wall within 2 cells (10 cm): two mapping runs do not line
// up better than that. Map 1's cells are kept for a common wall, so walls stay one line thick.
function dil(s,w,h){const t=new Uint8Array(w*h),o=new Uint8Array(w*h);
  for(let y=0;y<h;y++)for(let i=0;i<w;i++){let v=0;for(let k=-2;k<=2&&!v;k++){const q=i+k;if(q>=0&&q<w&&s[y*w+q])v=1}t[y*w+i]=v}
  for(let y=0;y<h;y++)for(let i=0;i<w;i++){let v=0;for(let k=-2;k<=2&&!v;k++){const q=y+k;if(q>=0&&q<h&&t[q*w+i])v=1}o[y*w+i]=v}return o}
function common(a,b,w,h){const n=w*h,v=new Uint8Array(n),dm=new Uint8Array(n),oa=new Uint8Array(n),ob=new Uint8Array(n);
  for(let i=0;i<n;i++){oa[i]=a[i]<100;ob[i]=b[i]<100}
  const na=dil(oa,w,h),nb=dil(ob,w,h);
  for(let i=0;i<n;i++){const fa=a[i]>230,fb=b[i]>230;
    if(oa[i]&&nb[i])v[i]=0;                              // wall in both
    else if(ob[i]&&na[i]&&!oa[i])v[i]=fa?254:205;        // map 2's copy of a common wall: follow map 1
    else if(fa&&fb)v[i]=254;                             // floor in both
    else if(!oa[i]&&!fa&&!ob[i]&&!fb)v[i]=205;           // unknown in both
    else{v[i]=205;dm[i]=1}}                              // they differ: unknown until you decide
  return{v,dm,oa,ob}}

function gray(url){return new Promise((ok,no)=>{const im=new Image();im.onload=()=>{const t=document.createElement('canvas');t.width=im.width;t.height=im.height;
  const g=t.getContext('2d');g.drawImage(im,0,0);const d=g.getImageData(0,0,im.width,im.height).data,o=new Uint8Array(im.width*im.height);
  for(let i=0;i<o.length;i++)o[i]=d[i*4];ok({d:o,w:im.width,h:im.height})};im.onerror=()=>no(new Error('Could not load '+url));im.src=url+'&t='+Date.now()})}
function mkLayer(k,mask){const cv=document.createElement('canvas');cv.width=W;cv.height=H;const g=cv.getContext('2d',{willReadFrequently:true}),t=new ImageData(W,H),[r,gr,b]=COL[k];
  for(let i=0;i<W*H;i++)if(mask[i]){t.data[i*4]=r;t.data[i*4+1]=gr;t.data[i*4+2]=b;t.data[i*4+3]=255}g.putImageData(t,0,0);L[k]=cv;LX[k]=g}
async function build(){const n1=$('m1').value,n2=$('m2').value,two=n2&&n2!=n1;rot=+$('mrot').value||0;const rq='&rot='+rot;msg(two?'Aligning '+n2+' to '+n1+' (first time takes up to a minute)...':'Loading...');
  info=await(await fetch('/api/info?name='+n1+rq)).json();A=await gray('/api/png?name='+n1+rq);B=two?await gray('/api/png?name='+n2+'&align_to='+n1+rq):null;
  W=A.w;H=A.h;c.width=W;c.height=H;const n=W*H,img=new ImageData(W,H);let r;
  // the map on screen is map 1 itself; the second map is only compared against it (orange = they differ, red = its walls)
  r=B?common(A.d,B.d,W,H):{dm:new Uint8Array(n),oa:A.d.map(v=>v<100),ob:new Uint8Array(n)};r.v=A.d.map(q3);
  for(let i=0;i<n;i++){img.data[i*4]=img.data[i*4+1]=img.data[i*4+2]=r.v[i];img.data[i*4+3]=255}
  x.putImageData(img,0,0);mkLayer('dif',r.dm);mkLayer('la',r.oa);mkLayer('lb',r.ob);showSrc();
  undo=[];dirty=false;sel=null;clip=null;pasting=false;poly=[];$('as').value=n1.replace(/_edit\d*$/,'')+'_edit'+(rot?'_rot':'');setZoom(zoom);
  let wl=0,fr=0,df=0;for(let i=0;i<n;i++){wl+=r.v[i]==0;fr+=r.v[i]==254;df+=r.dm[i]}
  msg(`${n1}${rot?' turned '+rot+' deg':''}: ${W} x ${H} cells, ${wl} wall cells, ${(fr*info.resolution**2).toFixed(1)} m2 floor.`+(B?` Compared with ${n2}: ${df} cells differ (orange).`:' Choose a second map to compare it with.'))}

// size labels drawn next to a box / line while it is being dragged
const dims=(w,h)=>`${w} x ${h} cells = ${(w*info.resolution).toFixed(2)} x ${(h*info.resolution).toFixed(2)} m`;
function tag(o,t,px,py){o.font='13px sans-serif';const w=o.measureText(t).width+10,y=py>24?py-22:py+4;o.fillStyle='rgba(15,20,32,.92)';o.fillRect(px,y,w,18);o.fillStyle='#fff';o.fillText(t,px+5,y+13)}
// the two source maps, read-only, under the common map (map 2 as aligned to map 1; both turned if the map is rotated)
function showSrc(){const on=$('s_src').checked;$('src').style.display=on?'flex':'none';$('wrap').style.height=on&&B?'52vh':'calc(100vh - 150px)';$('cA').parentNode.style.display='none';if(!on||!A)return;
  for(const[id,m,t]of[['cB',B,'Second map, lined up with the map above: '+$('m2').value]]){const cv=$(id);cv.parentNode.style.display=m?'':'none';if(!m)continue;
    cv.width=W;cv.height=H;const d=new ImageData(W,H);for(let i=0;i<W*H;i++){d.data[i*4]=d.data[i*4+1]=d.data[i*4+2]=m.d[i];d.data[i*4+3]=255}
    cv.getContext('2d').putImageData(d,0,0);cv.previousElementSibling.textContent=t}}
function setZoom(z){zoom=Math.max(1,Math.min(12,z));for(const k of[c,ov]){k.style.width=W*zoom+'px';k.style.height=H*zoom+'px'}$('zv').textContent=zoom+'x';overlay()}
function overlay(pv){ov.width=W*zoom;ov.height=H*zoom;const o=ov.getContext('2d');o.imageSmoothingEnabled=false;o.globalAlpha=+$('op').value;
  for(const k of KEYS)if(L[k]&&$('s_'+k).checked)o.drawImage(L[k],0,0,W*zoom,H*zoom);o.globalAlpha=1;
  if(info){o.font='12px sans-serif';for(const q of[{id:'home',x:info.home.x,y:info.home.y}].concat(info.checkpoints)){
    const px=(q.x-info.origin[0])/info.resolution*zoom,py=(H-(q.y-info.origin[1])/info.resolution)*zoom;
    o.fillStyle=q.id=='home'?'#16a34a':'#9333ea';o.beginPath();o.arc(px,py,5,0,7);o.fill();o.fillText(q.id,px+8,py+4)}}
  // brush cursor: the exact cells one dab will paint, a box around them, and the size
  if(hover&&!pasting&&(shape=='brush'||shape=='line')){const z=+$('size').value,h=(z-1)/2,bx=Math.round(hover[0]-h),by=Math.round(hover[1]-h);o.fillStyle='rgba(61,123,253,.5)';
    for(let dy=0;dy<z;dy++)for(let dx=0;dx<z;dx++){const ex=dx-h,ey=dy-h;if(z<3||ex*ex+ey*ey<=h*h+.5)o.fillRect((bx+dx)*zoom,(by+dy)*zoom,zoom,zoom)}
    o.lineWidth=1;o.strokeStyle='#3d7bfd';o.strokeRect(bx*zoom+.5,by*zoom+.5,z*zoom,z*zoom);
    if(!down)tag(o,`brush ${z} x ${z} cells = ${(z*info.resolution*100).toFixed(0)} x ${(z*info.resolution*100).toFixed(0)} cm`,(bx+z)*zoom+6,by*zoom+26)}
  if(sel||poly.length){const P=(p,i)=>o[i?'lineTo':'moveTo']((p[0]+.5)*zoom,(p[1]+.5)*zoom);o.setLineDash([6,4]);o.lineWidth=2;o.strokeStyle=o.fillStyle='#22d3ee';
    if(poly.length){o.beginPath();poly.forEach(P);if(hover)P(hover,1);o.stroke();for(const p of poly)o.fillRect((p[0]+.5)*zoom-3,(p[1]+.5)*zoom-3,6,6)}
    else if(sel.pts){o.beginPath();sel.pts.forEach(P);o.closePath();o.stroke()}else o.strokeRect(sel.x*zoom,sel.y*zoom,sel.w*zoom,sel.h*zoom);o.setLineDash([]);if(sel&&!poly.length&&!sel.pts)tag(o,dims(sel.w,sel.h),sel.x*zoom,sel.y*zoom)}
  if(pasting&&clip&&hover){o.globalAlpha=.6;o.drawImage(clip.cv,(hover[0]-(clip.w>>1))*zoom,(hover[1]-(clip.h>>1))*zoom,clip.w*zoom,clip.h*zoom);o.globalAlpha=1}
  if(pv&&shape=='rect'){o.fillStyle='rgba(61,123,253,.35)';o.strokeStyle='#3d7bfd';const a=Math.min(pv[0],pv[2])*zoom,b=Math.min(pv[1],pv[3])*zoom,w=(Math.abs(pv[2]-pv[0])+1)*zoom,h=(Math.abs(pv[3]-pv[1])+1)*zoom;o.fillRect(a,b,w,h);o.strokeRect(a,b,w,h);tag(o,dims(Math.abs(pv[2]-pv[0])+1,Math.abs(pv[3]-pv[1])+1),a,b)}
  else if(pv){o.strokeStyle='#3d7bfd';o.lineWidth=Math.max(1,+$('size').value*zoom);o.globalAlpha=.6;o.beginPath();o.moveTo((pv[0]+.5)*zoom,(pv[1]+.5)*zoom);o.lineTo((pv[2]+.5)*zoom,(pv[3]+.5)*zoom);o.stroke();o.globalAlpha=1;
    tag(o,(Math.hypot(pv[2]-pv[0],pv[3]-pv[1])*info.resolution).toFixed(2)+' m long',Math.min(pv[0],pv[2])*zoom,Math.min(pv[1],pv[3])*zoom)}}
// painting settles a difference, so the orange goes; clearing to floor/unknown also removes the two maps' wall dots there
function wipe(a,b,w,h,all){LX.dif.clearRect(a,b,w,h);if(all){LX.la.clearRect(a,b,w,h);LX.lb.clearRect(a,b,w,h)}}
function stamp(cx,cy){const s=+$('size').value,h=(s-1)/2;x.fillStyle=`rgb(${val},${val},${val})`;
  for(let dy=0;dy<s;dy++)for(let dx=0;dx<s;dx++){const ex=dx-h,ey=dy-h;if(s<3||ex*ex+ey*ey<=h*h+.5){const a=Math.round(cx-h)+dx,b=Math.round(cy-h)+dy;x.fillRect(a,b,1,1);wipe(a,b,1,1,val!=0)}}}
function seg(a,b,c2,d){const n=Math.max(Math.abs(c2-a),Math.abs(d-b),1);for(let i=0;i<=n;i++)stamp(a+(c2-a)*i/n,b+(d-b)*i/n)}
function cell(e){const r=c.getBoundingClientRect();return[Math.floor((e.clientX-r.left)/zoom),Math.floor((e.clientY-r.top)/zoom)]}
function push(){const s={d:x.getImageData(0,0,W,H)};for(const k of KEYS)s[k]=LX[k].getImageData(0,0,W,H);undo.push(s);if(undo.length>25)undo.shift();dirty=true}
function box(a,b){const x0=Math.max(0,Math.min(a[0],b[0])),y0=Math.max(0,Math.min(a[1],b[1])),x1=Math.min(W-1,Math.max(a[0],b[0])),y1=Math.min(H-1,Math.max(a[1],b[1]));return{x:x0,y:y0,w:x1-x0+1,h:y1-y0+1}}
function doCopy(){if(!sel)return msg('Select a box first (Select tool, then drag on the map).');
  const src=x.getImageData(sel.x,sel.y,sel.w,sel.h);if(sel.mask)for(let i=0;i<sel.mask.length;i++)if(!sel.mask[i])src.data[i*4+3]=0;
  clip={src,cx:sel.x+sel.w/2,cy:sel.y+sel.h/2};applyRot();msg(`Copied ${sel.w} x ${sel.h} cells. Paste: click on the map. Paste in place: same position.`)}
function base(){const a=clip.src,w=a.width,h=a.height,b=new ImageData(w,h);
  for(let j=0;j<h;j++)for(let i=0;i<w;i++){const k=((fl.fv?h-1-j:j)*w+(fl.fh?w-1-i:i))*4;b.data.set(a.data.subarray(k,k+4),(j*w+i)*4)}return b}
// rotate the copied piece about its centre (positive = counter-clockwise), nearest cell, corners stay empty
function applyRot(){if(!clip)return;const a=(+$('ang').value||0)*Math.PI/180,cs=Math.cos(a),sn=Math.sin(a),s=base(),w0=s.width,h0=s.height,
  w=Math.max(1,Math.round(Math.abs(w0*cs)+Math.abs(h0*sn))),h=Math.max(1,Math.round(Math.abs(w0*sn)+Math.abs(h0*cs))),d=new ImageData(w,h);
  for(let j=0;j<h;j++)for(let i=0;i<w;i++){const u=i+.5-w/2,v=j+.5-h/2,sx=Math.floor(u*cs-v*sn+w0/2),sy=Math.floor(u*sn+v*cs+h0/2);
    if(sx<0||sy<0||sx>=w0||sy>=h0)continue;const k=(sy*w0+sx)*4,o=(j*w+i)*4;if(!s.data[k+3])continue;d.data[o]=d.data[o+1]=d.data[o+2]=s.data[k];d.data[o+3]=255}
  const cv=document.createElement('canvas');cv.width=w;cv.height=h;cv.getContext('2d').putImageData(d,0,0);
  Object.assign(clip,{d,cv,w,h,x:Math.round(clip.cx-w/2),y:Math.round(clip.cy-h/2)});overlay()}
function putClip(px,py){push();const w=clip.w,h=clip.h,dst=x.getImageData(px,py,w,h),s=clip.d.data,t=dst.data,skip=$('skip').checked;
  for(let i=0;i<w*h*4;i+=4){const v=s[i];if(!s[i+3]||(skip&&v>=100&&v<=230))continue;t[i]=t[i+1]=t[i+2]=v;t[i+3]=255}x.putImageData(dst,px,py);LX.dif.clearRect(px,py,w,h);overlay()}
// the selection is a box, or a polygon (bounding box + mask of the cells inside)
function eachSel(fn){for(let j=0;j<sel.h;j++)for(let k=0;k<sel.w;k++){const i=j*sel.w+k;if(!sel.mask||sel.mask[i])fn(i*4,k,j)}}
function clearSel(keys){for(const k of keys){const t=LX[k].getImageData(sel.x,sel.y,sel.w,sel.h);eachSel(o=>t.data[o+3]=0);LX[k].putImageData(t,sel.x,sel.y)}}
function closePoly(){if(poly.length<3){poly=[];overlay();return msg('A polygon needs at least 3 corners.')}
  const xs=poly.map(p=>p[0]),ys=poly.map(p=>p[1]),x0=Math.max(0,Math.min(...xs)),y0=Math.max(0,Math.min(...ys)),w=Math.min(W-1,Math.max(...xs))-x0+1,h=Math.min(H-1,Math.max(...ys))-y0+1;
  const t=document.createElement('canvas');t.width=w;t.height=h;const g=t.getContext('2d');g.beginPath();poly.forEach((p,i)=>g[i?'lineTo':'moveTo'](p[0]+.5-x0,p[1]+.5-y0));g.closePath();g.fill();
  const al=g.getImageData(0,0,w,h).data,mask=new Uint8Array(w*h);let n=0;for(let i=0;i<w*h;i++)if(al[i*4+3]>=128){mask[i]=1;n++}
  sel={x:x0,y:y0,w,h,mask,pts:poly};poly=[];overlay();msg(`Polygon selected: ${n} cells (${(n*info.resolution**2).toFixed(1)} m2). Fill selection, Copy, Rotate, or Use second map here.`)}
function fillSel(){if(!sel)return msg('Select an area first (Select or Polygon).');push();const d=x.getImageData(sel.x,sel.y,sel.w,sel.h);let n=0;
  eachSel(o=>{d.data[o]=d.data[o+1]=d.data[o+2]=val;d.data[o+3]=255;n++});x.putImageData(d,sel.x,sel.y);clearSel(val?KEYS:['dif']);overlay();
  msg(`Filled ${n} cells with ${val==254?'floor':val?'unknown':'wall'}. Undo takes it back.`)}
// take cells from one of the two source maps: the selected area, or (no selection) every orange cell
function useMap(i){const s=i==1?A:B;if(!s)return msg('Choose a second map to compare with first.');push();let n=0;
  if(sel){const d=x.getImageData(sel.x,sel.y,sel.w,sel.h);eachSel((o,k,j)=>{const v=q3(s.d[(sel.y+j)*W+sel.x+k]);if(d.data[o]!=v)n++;d.data[o]=d.data[o+1]=d.data[o+2]=v;d.data[o+3]=255});x.putImageData(d,sel.x,sel.y);clearSel(['dif'])}
  else{const d=x.getImageData(0,0,W,H),m=LX.dif.getImageData(0,0,W,H).data;for(let k=0;k<W*H;k++)if(m[k*4+3]){const v=q3(s.d[k]);if(d.data[k*4]!=v)n++;d.data[k*4]=d.data[k*4+1]=d.data[k*4+2]=v}x.putImageData(d,0,0);LX.dif.clearRect(0,0,W,H)}
  overlay();msg((sel?'Selected area ':'Every orange cell ')+(i==1?'put back as in the saved '+$('m1').value:'taken from '+$('m2').value)+': '+n+' cells changed. Undo takes it back.')}
// rotating shows the piece at once: it copies the selection if needed and turns Paste on, preview at the original spot
function showRot(){if(!clip&&sel)doCopy();if(!clip)return msg('Select an area first (Select or Polygon), then rotate it.');applyRot();
  if(!pasting){pasting=true;$('paste').classList.add('on');hover=[Math.round(clip.cx),Math.round(clip.cy)]}overlay();
  msg(`Piece rotated to ${+$('ang').value||0} deg (blue preview). Move it with the cursor and click to place it; Esc cancels.`)}
function rotBy(d){$('ang').value=(((+$('ang').value||0)+d)%360+360)%360;showRot()}
function setShape(id){shape=id;pasting=false;poly=[];hover=null;document.querySelectorAll('.shape').forEach(k=>k.classList.toggle('on',k.id==id));$('paste').classList.remove('on');overlay()}
c.onpointerdown=e=>{c.setPointerCapture(e.pointerId);const q=cell(e);
  if(pasting){if(clip)putClip(q[0]-(clip.w>>1),q[1]-(clip.h>>1));return}
  if(shape=='poly'){if(poly.length>2&&Math.hypot(q[0]-poly[0][0],q[1]-poly[0][1])<=Math.max(2,8/zoom))closePoly();else{if(!poly.length)sel=null;poly.push(q);overlay()}return}
  if(shape=='sel'){down=q;sel=box(q,q);overlay();return}
  push();down=last=q;axis=null;if(shape=='brush'){stamp(...q);overlay()}};
c.onpointermove=e=>{const q=cell(e);if(pasting){hover=q;overlay();return}if(shape=='poly'){hover=q;if(poly.length)overlay();return}
  // Shift while brushing: keep the stroke on the row or column it started on (whichever way it first moves)
  if(shape=='brush'&&down&&e.shiftKey){if(!axis&&(q[0]!=down[0]||q[1]!=down[1]))axis=Math.abs(q[0]-down[0])>=Math.abs(q[1]-down[1])?'h':'v';if(axis=='h')q[1]=down[1];if(axis=='v')q[0]=down[0]}
  if(shape=='brush'||shape=='line'){hover=q;if(!down){overlay();return}}if(!down)return;
  if(shape=='sel'){sel=box(down,q);overlay()}else if(shape=='brush'){seg(...last,...q);last=q;overlay()}else overlay([down[0],down[1],...q])};
c.onpointerup=e=>{if(!down)return;const q=cell(e);
  if(shape=='sel'){sel=box(down,q);if(sel.w<2&&sel.h<2){sel=null;msg('Selection cleared.')}else msg(`Selected ${sel.w} x ${sel.h} cells (${(sel.w*info.resolution).toFixed(1)} x ${(sel.h*info.resolution).toFixed(1)} m). Copy, Fill selection, or Use second map here.`)}
  if(shape=='line')seg(down[0],down[1],...q);
  if(shape=='rect'){x.fillStyle=`rgb(${val},${val},${val})`;const r=[Math.min(down[0],q[0]),Math.min(down[1],q[1]),Math.abs(q[0]-down[0])+1,Math.abs(q[1]-down[1])+1];x.fillRect(...r);wipe(...r,val!=0);
    // a wall box is an outline with unknown inside: the lidar sees the faces of a solid object, not its inside
    if(!val&&r[2]>2&&r[3]>2){const n=[r[0]+1,r[1]+1,r[2]-2,r[3]-2];x.fillStyle='rgb(205,205,205)';x.fillRect(...n);wipe(...n,true)}
    msg('Box painted: '+dims(r[2],r[3])+'.')}
  down=null;overlay()};
c.onpointerleave=()=>{if(!pasting&&(shape=='brush'||shape=='line')){hover=null;overlay()}};
c.ondblclick=()=>{if(shape=='poly'&&poly.length>3){poly.pop();closePoly()}};
for(const b of document.querySelectorAll('.paint'))b.onclick=()=>{val=+b.dataset.v;document.querySelectorAll('.paint').forEach(k=>k.classList.toggle('on',k==b))};
for(const b of document.querySelectorAll('.shape'))b.onclick=()=>setShape(b.id);
for(const b of document.querySelectorAll('.flip'))b.onclick=()=>{fl[b.id]=!fl[b.id];b.classList.toggle('on',fl[b.id]);applyRot()};
for(const k of KEYS)$('s_'+k).onchange=()=>overlay();$('op').oninput=()=>overlay();
$('s_src').onchange=showSrc;
$('size').oninput=()=>{$('sizev').textContent=$('size').value+' ('+(+$('size').value*(info?info.resolution*100:5)).toFixed(0)+' cm)'};
$('undo').onclick=()=>{const u=undo.pop();if(u){x.putImageData(u.d,0,0);for(const k of KEYS)LX[k].putImageData(u[k],0,0);overlay()}};
$('zin').onclick=()=>setZoom(zoom+1);$('zout').onclick=()=>setZoom(zoom-1);
$('use1').onclick=()=>useMap(1);$('use2').onclick=()=>useMap(2);
$('copy').onclick=doCopy;$('fill').onclick=fillSel;$('ang').oninput=showRot;$('rl').onclick=()=>rotBy(+$('rstep').value);$('rr').onclick=()=>rotBy(-$('rstep').value);
$('paste').onclick=()=>{if(!clip)return msg('Nothing copied yet.');pasting=!pasting;if(pasting)hover=[Math.round(clip.cx),Math.round(clip.cy)];$('paste').classList.toggle('on',pasting);msg(pasting?'Click on the map to place the copy. Esc or Paste again to stop.':'');overlay()};
$('inplace').onclick=()=>{if(!clip)return msg('Nothing copied yet.');putClip(clip.x,clip.y);msg('Pasted at the same position. Undo takes it back.')};
document.onkeydown=e=>{if(e.target.tagName=='INPUT'&&/text|number/.test(e.target.type))return;const m=e.ctrlKey||e.metaKey;
  if(e.key=='['||e.key==']')rotBy(e.key==']'?1:-1);if(e.key=='Enter'&&shape=='poly')closePoly();
  if(m&&e.key=='z'){e.preventDefault();$('undo').click()}if(m&&e.key=='c'){e.preventDefault();doCopy()}if(m&&e.key=='v'){e.preventDefault();if(!pasting)$('paste').click()}
  if(e.key=='Escape'){pasting=false;sel=null;poly=[];$('paste').classList.remove('on');overlay()}};
async function pick(){if(dirty&&!confirm('Discard the unsaved edits on this map?')){$('m1').value=cur[0];$('m2').value=cur[1];$('mrot').value=cur[2];return}
  cur=[$('m1').value,$('m2').value,$('mrot').value];try{await build()}catch(e){msg(e.message)}}
$('m1').onchange=pick;$('m2').onchange=pick;$('mrot').onchange=pick;
for(const[b,sg]of[['mrl',1],['mrr',-1]])$(b).onclick=()=>{$('mrot').value=((((+$('mrot').value||0)+sg*$('mstep').value)%360)+360)%360;pick()};
$('save').onclick=()=>c.toBlob(async b=>{const r=await(await fetch('/api/save?name='+$('m1').value+'&rot='+rot+'&as='+encodeURIComponent($('as').value),{method:'POST',body:b})).json();msg(r.msg);
  if(r.ok){dirty=false;await maps()}},'image/png');
async function maps(){const l=await(await fetch('/api/maps')).json(),h=l.map(n=>`<option>${n}</option>`).join(''),a=$('m1').value,b=$('m2').value;
  $('m1').innerHTML=h;$('m2').innerHTML='<option value="">(none)</option>'+h;if(a){$('m1').value=a;$('m2').value=b}return l}
maps().then(l=>{$('m1').value=l.includes('lab2d_11_18_2')?'lab2d_11_18_2':l[0];$('m2').value='';pick()});
</script></body></html>"""


if __name__ == '__main__':
    print(f'Map editor: http://localhost:{PORT}   (maps in {MAPS})')
    ThreadingHTTPServer(('0.0.0.0', PORT), H).serve_forever()
