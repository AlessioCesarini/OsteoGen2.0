"""
report.py

Trasforma l'output di compatibility.genera_report_compatibilita in due
formati consumabili:
- un JSON grezzo (per uso programmatico / da passare a render/generate.py)
- una pagina HTML autonoma (grafico a barre delle specie compatibili,
  thumbnail dei donatori per segmento) da aprire in un browser qualunque,
  senza server e senza dipendenze esterne (le immagini sono incorporate
  come base64).
"""
import base64
import io
import json
import os

from PIL import Image


def _thumbnail_base64(path, max_size=160):
    """Ritorna una data-URI base64 di una thumbnail JPEG, o None se il file
    non esiste o non e' leggibile (il report deve poter degradare senza
    rompersi se manca una texture)."""
    if not path or not os.path.exists(path):
        return None
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_size, max_size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


def genera_report_json(report, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    return output_path


_CSS = """
:root { color-scheme: light dark; }
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 900px;
       margin: 2rem auto; padding: 0 1rem; background:#0f1115; color:#e8e8e8; }
h1 { font-size: 1.4rem; }
h2 { font-size: 1.05rem; margin-top: 2rem; border-bottom: 1px solid #333; padding-bottom:.3rem;}
.header { display:flex; gap:1.5rem; align-items:center; }
.header img { width:140px; height:140px; object-fit:contain; background:#000; border-radius:8px; }
.bar-row { display:flex; align-items:center; gap:.6rem; margin:.35rem 0; }
.bar-label { width:150px; font-size:.85rem; flex-shrink:0; text-align:right; }
.bar-track { flex:1; background:#22252b; border-radius:4px; overflow:hidden; height:20px; }
.bar-fill { background:linear-gradient(90deg,#4f8cff,#7ad1ff); height:100%; }
.bar-pct { width:52px; font-size:.8rem; color:#aaa; }
.segmento { margin-bottom:1.4rem; }
.segmento h3 { font-size:.95rem; margin-bottom:.4rem; text-transform:capitalize; }
.donatori { display:flex; gap:.7rem; flex-wrap:wrap; }
.donatore { text-align:center; font-size:.75rem; width:78px; }
.donatore img { width:70px; height:70px; object-fit:contain; background:#000; border-radius:6px; }
.assente { color:#888; font-style:italic; font-size:.85rem; }
footer { margin-top:2.5rem; font-size:.75rem; color:#777; }
"""


def genera_report_html(report, db, fossile_path, output_path, titolo="Report di compatibilita' OsteoGen"):
    fossile_b64 = _thumbnail_base64(fossile_path, max_size=220)

    righe_bar = []
    for r in report["ranking_specie"]:
        nome = os.path.splitext(r["animale"])[0]
        righe_bar.append(f"""
        <div class="bar-row">
          <div class="bar-label">{nome}</div>
          <div class="bar-track"><div class="bar-fill" style="width:{r['compatibilita_pct']}%"></div></div>
          <div class="bar-pct">{r['compatibilita_pct']:.1f}%</div>
        </div>""")

    blocchi_segmento = []
    for parte, dati in report["per_segmento"].items():
        if not dati["presente_nel_target"]:
            blocchi_segmento.append(f"""
            <div class="segmento"><h3>{parte.replace('_',' ')}</h3>
            <div class="assente">Assente nel target: nessun dato da confrontare.</div></div>""")
            continue

        donatori_html = []
        for d in dati["donatori"][:5]:
            info = db.get(d["animale"], {})
            img_b64 = _thumbnail_base64(info.get("path_texture"))
            img_tag = f'<img src="{img_b64}">' if img_b64 else '<div style="width:70px;height:70px;background:#000;border-radius:6px;"></div>'
            nome = os.path.splitext(d["animale"])[0]
            donatori_html.append(f"""
            <div class="donatore">{img_tag}<div>{nome}</div><div>{d['compatibilita_pct']:.1f}%</div></div>""")

        blocchi_segmento.append(f"""
        <div class="segmento"><h3>{parte.replace('_',' ')}</h3>
        <div class="donatori">{''.join(donatori_html)}</div></div>""")

    fossile_img_tag = f'<img src="{fossile_b64}">' if fossile_b64 else ''

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{titolo}</title><style>{_CSS}</style></head>
<body>
  <div class="header">
    {fossile_img_tag}
    <div>
      <h1>{titolo}</h1>
      <div>Confrontato con {report['n_animali_confrontati']} animali del database geometrico.</div>
    </div>
  </div>

  <h2>Compatibilita' complessiva</h2>
  {''.join(righe_bar)}

  <h2>Donatori per segmento anatomico</h2>
  {''.join(blocchi_segmento)}

  <footer>Generato da OsteoGen 2.0 - report.py. Le percentuali sono una similarita'
  geometrica tra keypoint normalizzati, non una classificazione tassonomica.</footer>
</body></html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return output_path
