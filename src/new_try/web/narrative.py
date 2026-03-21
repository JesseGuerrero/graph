"""
Programmatic narrative extractor and storyline generator.
Extracts entities, sections, and events from a STORM report
and produces an embeddable D3.js storyline <div>.
"""
import re
import json
import html as html_mod
from collections import Counter

# Color palette matching the report's blue-based scheme
COLORS = [
    '#1e3a5f', '#2e86ab', '#a23b72', '#f18f01', '#3c896d',
    '#5c4d7d', '#c44536', '#4a8fe7', '#7b6d8d', '#e8871e',
    '#2d6a4f', '#774936', '#457b9d', '#9b2226', '#606c38',
]

# D3.js shape symbols for entity differentiation
SHAPES = [
    'symbolCircle', 'symbolSquare', 'symbolTriangle', 'symbolDiamond',
    'symbolStar', 'symbolCross', 'symbolWye',
]


def extract_entities(text, max_entities=12):
    """Extract key entities (proper nouns appearing frequently) from text."""
    # Remove citations and markdown
    clean = re.sub(r'\[+\d+\]+', '', text)
    clean = re.sub(r'!\[.*?\]\(.*?\)', '', clean)
    clean = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', clean)
    clean = re.sub(r'[#*_`>]', '', clean)

    # Find capitalized multi-word sequences (2-4 words)
    patterns = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b', clean)
    # Also single capitalized words that are likely entities (3+ chars, not sentence starters)
    singles = re.findall(r'(?<=[.!?]\s)[A-Z][a-z]+\b|(?<=\n)[A-Z][a-z]+\b', clean)
    # Filter: known non-entities
    skip = {'The', 'This', 'That', 'These', 'Those', 'However', 'According',
            'Although', 'While', 'After', 'Before', 'During', 'Between',
            'Within', 'Without', 'Under', 'Since', 'First', 'Second',
            'Third', 'Fourth', 'Fifth', 'Meanwhile', 'Furthermore',
            'Additionally', 'Moreover', 'Nevertheless', 'Consequently',
            'Report Summary', 'Table Contents', 'Stage One', 'Stage Two',
            'Stage Three', 'Stage Four', 'Stage Five'}

    counts = Counter()
    for p in patterns:
        if p not in skip and len(p) > 3:
            counts[p] += 1
    # Merge similar (one contains the other)
    merged = {}
    for name, count in counts.most_common(50):
        found = False
        for existing in list(merged.keys()):
            if name in existing or existing in name:
                longer = name if len(name) > len(existing) else existing
                shorter = name if len(name) <= len(existing) else existing
                if shorter != longer:
                    merged[longer] = merged.pop(existing, 0) + count
                    found = True
                    break
        if not found:
            merged[name] = count

    # Take top entities by frequency (minimum 2 mentions)
    top = [(name, c) for name, c in sorted(merged.items(), key=lambda x: -x[1]) if c >= 2]
    return [name for name, _ in top[:max_entities]]


def parse_report_sections(text):
    """Parse markdown into sections with their content."""
    lines = text.split('\n')
    sections = []
    current = None
    content_lines = []

    for line in lines:
        if line.strip().startswith('#'):
            if current is not None:
                sections.append({
                    'name': current['name'],
                    'level': current['level'],
                    'content': '\n'.join(content_lines),
                })
            hashes = len(line) - len(line.lstrip('#'))
            name = line.lstrip('#').strip()
            current = {'name': name, 'level': hashes}
            content_lines = []
        else:
            content_lines.append(line)

    if current is not None:
        sections.append({
            'name': current['name'],
            'level': current['level'],
            'content': '\n'.join(content_lines),
        })

    return sections


def build_narrative_data(article_text, topic=''):
    """Build narrative data structure from report text."""
    sections = parse_report_sections(article_text)
    entities = extract_entities(article_text)

    # Assign colors and shapes to entities
    characters = []
    for i, name in enumerate(entities):
        characters.append({
            'id': name,
            'color': COLORS[i % len(COLORS)],
            'shape': SHAPES[i % len(SHAPES)],
        })

    # Build events from top-level sections (h1)
    # with characters being entities mentioned in that section
    events = []
    section_groups = []

    for sec in sections:
        if sec['level'] == 1:
            section_groups.append(sec['name'])

        # Only create events from h1 and h2 sections
        if sec['level'] > 2:
            continue

        # Find which entities appear in this section
        present = []
        for ent in entities:
            if ent.lower() in sec['content'].lower() or ent.lower() in sec['name'].lower():
                present.append(ent)

        if not present and sec['level'] == 2:
            continue

        # Estimate tension from content length and keywords
        tension = 5
        content_lower = sec['content'].lower()
        high_tension = ['attack', 'strike', 'war', 'crisis', 'kill', 'destroy',
                        'threat', 'escalat', 'missile', 'bomb', 'invasi', 'conflict']
        low_tension = ['negotiat', 'peace', 'diplomat', 'agreement', 'cooperat',
                       'resolution', 'ceasefire', 'de-escalat']
        for word in high_tension:
            if word in content_lower:
                tension = min(10, tension + 1)
        for word in low_tension:
            if word in content_lower:
                tension = max(1, tension - 1)

        # Find parent section name
        parent_section = ''
        for sg in section_groups:
            parent_section = sg

        events.append({
            'name': sec['name'][:40],
            'time': '',
            'characters': present[:6],
            'location': '',
            'description': sec['content'].strip().replace('\n', ' ')[:300],
            'depends': [],
            'parallel': [],
            'merges': '',
            'tension': tension,
            'branch': 'main',
            'theme': '',
            'section': parent_section if sec['level'] == 2 else sec['name'],
        })

    # Build section list for the visualization
    seen_sections = []
    for e in events:
        if e['section'] and e['section'] not in seen_sections:
            seen_sections.append(e['section'])

    nav_sections = [{'name': s, 'timerange': ''} for s in seen_sections]

    return {
        'title': topic or 'Report Storyline',
        'timeline': '',
        'characters': characters,
        'events': events,
        'sections': nav_sections,
        'locations': [],
    }


def generate_storyline_div(data):
    """Generate embeddable storyline HTML div with D3.js (storyline only)."""
    dj = json.dumps(data, ensure_ascii=False)
    ne = len(data['events'])
    nc = len(data['characters'])
    uid = 'sl'  # unique prefix for IDs

    return f'''<div id="narrative-storyline" style="background:#f8fafc;border-radius:12px;overflow:hidden;margin-bottom:2rem;border:1px solid #e2e8f0;">
<div style="padding:20px 24px 12px;border-bottom:1px solid #e2e8f0;">
  <div style="font-family:system-ui,sans-serif;font-weight:700;font-size:1.1rem;color:#1e3a5f;">Report Storyline</div>
  <div style="font-family:monospace;font-size:.7rem;color:#64748b;margin-top:2px;">{ne} events &bull; {nc} entities &bull; scroll to explore</div>
</div>
<div id="{uid}-outer" style="display:flex;overflow:hidden;height:auto;">
  <div id="{uid}-cpanel" style="width:180px;flex-shrink:0;background:#f1f5f9;border-right:1px solid #e2e8f0;overflow-y:auto;">
    <div style="font-family:monospace;font-size:.6rem;color:#64748b;letter-spacing:.12em;text-transform:uppercase;padding:12px 12px 6px;border-bottom:1px solid #e2e8f0;position:sticky;top:0;background:#f1f5f9;z-index:5;">Entities</div>
    <div id="{uid}-clist"></div>
  </div>
  <div id="{uid}-wrap" style="flex:1;min-width:0;overflow-x:auto;overflow-y:hidden;cursor:grab;position:relative;">
    <svg id="{uid}-svg"></svg>
  </div>
</div>
<script>
(function(){{
const D={dj};
const chars=D.characters,evts=D.events,secs=D.sections;
function cc(id){{const c=chars.find(x=>x.id===id);return c?.color||'#666'}}
function cs(id){{const c=chars.find(x=>x.id===id);return c?.shape||'symbolCircle'}}
const shapeMap={{symbolCircle:d3.symbolCircle,symbolSquare:d3.symbolSquare,symbolTriangle:d3.symbolTriangle,symbolDiamond:d3.symbolDiamond,symbolStar:d3.symbolStar,symbolCross:d3.symbolCross,symbolWye:d3.symbolWye}};
function shapePath(id,size){{return d3.symbol().type(shapeMap[cs(id)]||d3.symbolCircle).size(size)()}}

const wrap=document.getElementById('{uid}-wrap');
const svg=d3.select('#{uid}-svg');

const ES=140,PL=50,PR=50,PT=80,PB=160;
const W=PL+Math.max(evts.length,1)*ES+PR;

const co=[];const cset=new Set();
evts.forEach(e=>e.characters.forEach(c=>{{if(!cset.has(c)){{cset.add(c);co.push(c)}}}}));

const CG=36;
const H=PT+Math.max(co.length,1)*CG+PB+40;
svg.attr('width',W).attr('height',H).attr('viewBox','0 0 '+W+' '+H);
document.getElementById('{uid}-outer').style.height=Math.max(H,co.length*48+100)+'px';

const by={{}};co.forEach((c,i)=>{{by[c]=PT+25+i*CG}});
const ex=evts.map((_,i)=>PL+i*ES+ES/2);

const ey=evts.map((e,ei)=>{{
  const pr=e.characters.filter(c=>co.includes(c));
  const ab=co.filter(c=>!pr.includes(c));
  const m={{}};
  if(pr.length){{const ctr=pr.reduce((s,c)=>s+by[c],0)/pr.length;const g=Math.min(16,CG*.45);pr.forEach((c,i)=>{{m[c]=ctr+(i-(pr.length-1)/2)*g}});}}
  ab.forEach(c=>{{m[c]=by[c]}});
  return m;
}});

co.forEach(c=>{{let last=by[c];for(let i=0;i<evts.length;i++){{if(evts[i].characters.includes(c)){{last=ey[i][c]}}else{{ey[i][c]=last+(by[c]-last)*.25;last=ey[i][c]}}}}}});

const defs=svg.append('defs');
const gl=defs.append('filter').attr('id','{uid}-gl').attr('x','-50%').attr('y','-50%').attr('width','200%').attr('height','200%');
gl.append('feGaussianBlur').attr('stdDeviation','5').attr('result','b');
gl.append('feMerge').selectAll('feMergeNode').data(['b','SourceGraphic']).join('feMergeNode').attr('in',d=>d);

const scols=['#1e3a5f','#2e86ab','#a23b72','#f18f01','#3c896d','#5c4d7d','#c44536','#4a8fe7'];
const sg=svg.append('g');
let ss2=0,ls2=evts[0]?.section||'',si2=0;
function drawSec(start,end,name,idx){{
  const x1=ex[start]-ES/2,x2=ex[end]+ES/2;
  sg.append('rect').attr('x',x1).attr('y',0).attr('width',x2-x1).attr('height',H).attr('fill',scols[idx%scols.length]).attr('opacity',.02);
  sg.append('text').attr('x',(x1+x2)/2).attr('y',22).attr('text-anchor','middle').attr('fill',scols[idx%scols.length]).attr('opacity',.6).attr('font-family','system-ui,sans-serif').attr('font-size','11px').attr('font-weight',700).attr('letter-spacing','.08em').text(name.toUpperCase().substring(0,35));
  sg.append('line').attr('x1',x2).attr('y1',0).attr('x2',x2).attr('y2',H).attr('stroke',scols[idx%scols.length]).attr('stroke-width',.5).attr('opacity',.1);
}}
evts.forEach((e,i)=>{{if(e.section!==ls2&&i>0){{drawSec(ss2,i-1,ls2,si2);si2++;ss2=i}}ls2=e.section}});
if(evts.length) drawSec(ss2,evts.length-1,ls2,si2);

evts.forEach((_,i)=>{{svg.append('line').attr('x1',ex[i]).attr('y1',PT-15).attr('x2',ex[i]).attr('y2',H-PB+15).attr('stroke','#1e3a5f').attr('stroke-width',.3).attr('opacity',.08)}});

const lg=svg.append('g');
const lineGen=d3.line().curve(d3.curveBasis);
co.forEach(cid=>{{
  const col=cc(cid);
  const pts=evts.map((_,i)=>[ex[i],ey[i][cid]]);
  lg.append('path').datum(cid).attr('d',lineGen(pts)).attr('fill','none').attr('stroke',col).attr('stroke-width',1.5).attr('opacity',.25).attr('class','charline cl-'+cid.replace(/\\s/g,'_'));
  let seg=[];
  evts.forEach((e,i)=>{{if(e.characters.includes(cid)){{seg.push([ex[i],ey[i][cid]])}}else{{if(seg.length>1){{lg.append('path').datum(cid).attr('d',d3.line().curve(d3.curveBasis)(seg)).attr('fill','none').attr('stroke',col).attr('stroke-width',3.5).attr('opacity',.75).attr('stroke-linecap','round').attr('class','charline cl-'+cid.replace(/\\s/g,'_'))}}seg=[]}}}});
  if(seg.length>1){{lg.append('path').datum(cid).attr('d',d3.line().curve(d3.curveBasis)(seg)).attr('fill','none').attr('stroke',col).attr('stroke-width',3.5).attr('opacity',.75).attr('stroke-linecap','round').attr('class','charline cl-'+cid.replace(/\\s/g,'_'))}}
}});

// Initial entity shapes on the far left so lines are easy to follow
const igp=svg.append('g');
co.forEach(c=>{{
  const y0=by[c];
  igp.append('path').attr('d',shapePath(c,120)).attr('transform','translate('+(PL-20)+','+y0+')').attr('fill',cc(c)).attr('stroke','#fff').attr('stroke-width',1.5).attr('opacity',.9);
}});

const eg=svg.append('g');
evts.forEach((e,i)=>{{
  const x=ex[i];
  const pr=e.characters.filter(c=>co.includes(c));
  const allYs=co.map(c=>ey[i][c]);
  const ys=pr.length?pr.map(c=>ey[i][c]):allYs;
  const minY=Math.min(...ys),maxY=Math.max(...ys);
  const tn=e.tension/10;
  const gc=d3.interpolateRgb('#94a3b8','#1e3a5f')(tn);

  const grp=eg.append('g');

  // Small faded shapes for non-participating entities
  const absent=co.filter(c=>!pr.includes(c));
  absent.forEach(c=>{{
    const cy2=ey[i][c];
    grp.append('path').attr('d',shapePath(c,40)).attr('transform','translate('+x+','+cy2+')').attr('fill',cc(c)).attr('opacity',.15);
  }});
  // Full shapes for participating entities
  pr.forEach(c=>{{
    const cy2=ey[i][c];
    grp.append('path').attr('d',shapePath(c,150)).attr('transform','translate('+x+','+cy2+')').attr('fill',cc(c)).attr('stroke','#fff').attr('stroke-width',1.5).attr('opacity',.9);
  }});

  const ly=maxY+24;
  // Word-wrap event name at 23 chars
  const wrapText=(str,lim)=>{{const words=str.split(' ');const lines=[];let cur='';words.forEach(w=>{{if((cur+' '+w).trim().length>lim){{lines.push(cur.trim());cur=w}}else{{cur+=' '+w}}}});if(cur.trim())lines.push(cur.trim());return lines}};
  const nameLines=wrapText(e.name,23);
  nameLines.forEach((ln,li)=>{{
    grp.append('text').attr('x',x).attr('y',ly+li*13).attr('text-anchor','middle').attr('fill','#1e293b').attr('opacity',.7).attr('font-family','system-ui,sans-serif').attr('font-size','11px').attr('font-weight',600).text(ln);
  }});

  grp.append('rect').attr('x',x-10).attr('y',minY-24).attr('width',20).attr('height',14).attr('rx',3).attr('fill',gc).attr('opacity',.8);
  grp.append('text').attr('x',x).attr('y',minY-14).attr('text-anchor','middle').attr('fill','#fff').attr('font-size','8px').attr('font-family','monospace').attr('font-weight',700).text(e.tension);

  // Tooltip on hover
  grp.append('rect').attr('x',x-ES/2).attr('y',PT-20).attr('width',ES).attr('height',H-PT-PB+40).attr('fill','transparent').attr('cursor','pointer')
    .on('mouseenter',function(ev){{
      const tip=document.getElementById('{uid}-tip');
      if(!tip) return;
      const ch=pr.map(c=>'<span style="background:'+cc(c)+';padding:2px 6px;border-radius:2px;font-size:10px;color:#fff;margin:1px;">'+c+'</span>').join(' ');
      const descRaw=e.description;
      const descHtml=descRaw.replace(/!\[([^\]]*)\]\(([^)]+)\)/g,'<img src="$2" alt="$1" style="max-width:100%;border-radius:4px;margin:4px 0;">').trim();
      tip.innerHTML='<div style="font-weight:700;color:#1e293b;margin-bottom:4px;">'+e.name+'</div>'+(descHtml?'<div style="font-size:11px;color:#64748b;margin-bottom:6px;">'+descHtml+'</div>':'')+'<div>'+ch+'</div>';
      tip.style.opacity='1';
      tip.style.left=Math.min(ev.clientX+12,innerWidth-300)+'px';
      tip.style.top=(ev.clientY+12)+'px';
    }})
    .on('mousemove',function(ev){{
      const tip=document.getElementById('{uid}-tip');
      if(tip){{tip.style.left=Math.min(ev.clientX+12,innerWidth-300)+'px';tip.style.top=(ev.clientY+12)+'px';}}
    }})
    .on('mouseleave',function(){{
      const tip=document.getElementById('{uid}-tip');
      if(tip) tip.style.opacity='0';
    }});
}});

// Drag scroll
let dn=false,sx3,sl3;
wrap.addEventListener('mousedown',e=>{{if(e.button===0){{dn=true;sx3=e.pageX-wrap.offsetLeft;sl3=wrap.scrollLeft}}}});
wrap.addEventListener('mouseleave',()=>{{dn=false}});
wrap.addEventListener('mouseup',()=>{{dn=false}});
wrap.addEventListener('mousemove',e=>{{if(!dn)return;e.preventDefault();wrap.scrollLeft=sl3-(e.pageX-wrap.offsetLeft-sx3)*1.5}});

// Character panel
const charList=document.getElementById('{uid}-clist');
let activeC=null;
co.forEach(c=>{{
  const row=document.createElement('div');
  row.style.cssText='display:flex;align-items:center;gap:7px;padding:0 12px;height:'+CG+'px;cursor:pointer;transition:background .15s,opacity .15s;';
  row.dataset.char=c;
  const cls=c.replace(/\\s/g,'_');
  const shapeEl=document.createElementNS('http://www.w3.org/2000/svg','svg');
  shapeEl.setAttribute('width','14');shapeEl.setAttribute('height','14');shapeEl.setAttribute('viewBox','-8 -8 16 16');shapeEl.style.flexShrink='0';
  const sp=document.createElementNS('http://www.w3.org/2000/svg','path');
  sp.setAttribute('d',shapePath(c,80));sp.setAttribute('fill',cc(c));
  shapeEl.appendChild(sp);
  row.appendChild(shapeEl);
  const nm2=document.createElement('span');nm2.style.cssText='font-family:monospace;font-size:.65rem;color:'+cc(c)+';';nm2.textContent=c;
  row.appendChild(nm2);
  row.addEventListener('click',()=>{{
    if(activeC===c){{activeC=null;svg.selectAll('.charline').style('opacity',function(){{return d3.select(this).attr('stroke-width')==='1.5'?.25:.75}});charList.querySelectorAll('[data-char]').forEach(r=>r.style.opacity='1');}}
    else{{activeC=c;svg.selectAll('.charline').style('opacity',function(){{return d3.select(this).datum()===c?(d3.select(this).attr('stroke-width')==='1.5'?.35:.9):.05}});charList.querySelectorAll('[data-char]').forEach(r=>{{r.style.opacity=r.dataset.char===c?'1':'.25'}});}}
  }});
  charList.appendChild(row);
}});
charList.style.paddingTop=(PT+25-CG/2)+'px';
}})();
</script>
<div id="{uid}-tip" style="position:fixed;pointer-events:none;background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:12px 16px;max-width:280px;opacity:0;transition:opacity .12s;z-index:300;box-shadow:0 4px 16px rgba(0,0,0,.12);font-family:system-ui,sans-serif;font-size:12px;color:#334155;"></div>
</div>'''
