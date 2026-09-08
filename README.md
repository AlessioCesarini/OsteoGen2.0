# OsteoGen 2.0

Da un fossile/scheletro a (1) un report di compatibilita' biologica con gli
animali di un database geometrico e (2) un render generativo dell'animale
ricostruito, mescolando le texture dei donatori pesate per compatibilita'.
Funziona anche per animali estinti mai fotografati da vivi (es. dinosauri),
perche' il retrieval si basa solo sulla forma dello scheletro, mai su una
foto del target.

Vedi `OsteoGen_2/documentazione_finale.txt` per l'architettura dettagliata
di ogni modulo, e `OsteoGen_2/dataset_tools/SOURCES.md` per ampliare il
dataset.

## Uso rapido

```bash
cd OsteoGen_2
pip install -r requirements.txt

# Solo report di compatibilita' (nessuna GPU richiesta oltre a quella usata
# per l'inferenza dei keypoint, che gira anche su CPU pur se piu' lenta):
python pipeline.py --fossile input/mio_fossile.jpg --no-render

# Report + render generativo (richiede GPU con CUDA):
python pipeline.py --fossile input/mio_fossile.jpg
```

---

A Dinosaur Project
