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
.confidenza-bassa { background:#3a2c10; border:1px solid #6b4e14; border-radius:6px;
       padding:.7rem 1rem; margin:1rem 0; font-size:.85rem; color:#f0c674; }
footer { margin-top:2.5rem; font-size:.75rem; color:#777; }
"""


def genera_report_html(report, db, fossile_path, output_path, titolo="OsteoGen compatibility report"):
    from display_names import species_name, segment_name
    from dataset_paths import resolve_dataset_path

    fossile_b64 = _thumbnail_base64(fossile_path, max_size=220)

    righe_bar = []
    for r in report["ranking_specie"]:
        nome = species_name(r["animale"])
        righe_bar.append(f"""
        <div class="bar-row">
          <div class="bar-label">{nome}</div>
          <div class="bar-track"><div class="bar-fill" style="width:{r['compatibilita_pct']}%"></div></div>
          <div class="bar-pct">{r['compatibilita_pct']:.1f}%</div>
        </div>""")

    blocchi_segmento = []
    for parte, dati in report["per_segmento"].items():
        etichetta = segment_name(parte)
        if not dati["presente_nel_target"]:
            blocchi_segmento.append(f"""
            <div class="segmento"><h3>{etichetta}</h3>
            <div class="assente">Absent in target: nothing to compare.</div></div>""")
            continue

        donatori_html = []
        for d in dati["donatori"][:5]:
            info = db.get(d["animale"], {})
            texture_path = resolve_dataset_path(info.get("path_texture"), "target_y", d["animale"])
            img_b64 = _thumbnail_base64(texture_path)
            img_tag = f'<img src="{img_b64}">' if img_b64 else '<div style="width:70px;height:70px;background:#000;border-radius:6px;"></div>'
            nome = species_name(d["animale"])
            donatori_html.append(f"""
            <div class="donatore">{img_tag}<div>{nome}</div><div>{d['compatibilita_pct']:.1f}%</div></div>""")

        blocchi_segmento.append(f"""
        <div class="segmento"><h3>{etichetta}</h3>
        <div class="donatori">{''.join(donatori_html)}</div></div>""")

    fossile_img_tag = f'<img src="{fossile_b64}">' if fossile_b64 else ''

    confidenza = report.get("confidenza_keypoint")
    banner_confidenza = ""
    if confidenza:
        from ui import LOW_CONFIDENCE_AVG
        if confidenza["media"] < LOW_CONFIDENCE_AVG:
            banner_confidenza = f"""
      <div class="confidenza-bassa">Low average keypoint confidence (avg {confidenza['media']:.2f},
      min {confidenza['minima']:.2f}): the target's shape may be too different from the training
      set for the network to place all 14 joints reliably. Treat the percentages below with caution.</div>"""

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{titolo}</title><style>{_CSS}</style></head>
<body>
  <div class="header">
    {fossile_img_tag}
    <div>
      <h1>{titolo}</h1>
      <div>Compared against {report['n_animali_confrontati']} animals in the geometric database.</div>
      {banner_confidenza}
    </div>
  </div>

  <h2>Overall compatibility</h2>
  {''.join(righe_bar)}

  <h2>Donors per anatomical segment</h2>
  {''.join(blocchi_segmento)}

  <footer>Generated by OsteoGen 2.0 - report.py. These percentages are a geometric
  similarity between normalized keypoints, not a taxonomic classification.</footer>
</body></html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return output_path
