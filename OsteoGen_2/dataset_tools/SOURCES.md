# Fonti per ampliare il dataset (input_x / target_y)

Ogni animale nel dataset richiede DUE immagini isolate su sfondo nero,
512x512 (o comunque un lato massimo gestibile da `preprocess_target.py`):
- `input_x/<nome>.png`: lo scheletro/fossile (illustrazione, render 3D, o
  foto di un montaggio museale)
- `target_y/<nome>.png`: una foto dell'animale vivo corrispondente

Prima di aggiungere qualunque immagine: controlla la licenza e annotala in
`attributions.csv` (autore, fonte, licenza). Non scaricare in blocco senza
controllare licenza per licenza: e' per questo che `fetch_wikimedia.py`
elenca i candidati invece di scaricarli automaticamente.

## Scheletri / fossili (input_x)

- **Wikimedia Commons**, categoria [Category:Skeletons](https://commons.wikimedia.org/wiki/Category:Skeletons)
  e sottocategorie per taxon — molte foto museali CC-BY/CC-BY-SA/CC0.
- **MorphoSource** (morphosource.org) — scansioni 3D di scheletri museali,
  molte a licenza CC0/CC-BY; permette di esportare screenshot/render puliti
  su sfondo trasparente.
- **Sketchfab**, filtro "Downloadable" + licenza CC — molti modelli 3D di
  scheletri (alcuni gia' usati probabilmente per gli scheletri "3D render"
  gia' presenti nel dataset, es. gli uccelli).
- **Bone Clones** (boneclones.com) — cataloghi fotografici di calchi ossei,
  utile come *riferimento visivo* per ri-fotografare/renderizzare i propri
  scan, ma verificare i termini d'uso prima di riusare le loro foto dirette.

## Animali vivi (target_y)

- **Wikimedia Commons** — stessa fonte, cercare per nome scientifico o
  comune, filtrare per licenza libera.
- **iNaturalist** (api iNaturalist, `research-grade` observations con
  licenza CC) — ottimo per specie meno comuni, foto naturalistiche di
  qualita' variabile ma abbondanti.
- **GBIF** (gbif.org) — aggregatore di osservazioni con immagini, spesso
  licenza aperta, utile per specie rare.

## Priorita' di ampliamento (rispetto alle 65 specie attuali)

Per rendere il retrieval piu' utile su target estinti (dinosauri e affini),
conviene privilegiare:
1. **Rettili e coccodrilli** (varani, alligatori, caimani, camaleonti extra) —
   parenti anatomici piu' stretti dei dinosauri non-aviani.
2. **Ratiti e uccelli terrestri** (emu, nandu', tacchino gia' presente,
   fagiano gia' presente) — la parentela uccelli-dinosauri e' il motivo per
   cui questi donatori contano di piu' per un target come il T-Rex.
3. **Pesci e anfibi aggiuntivi** — utile per target acquatici (plesiosauri,
   ittiosauri) di cui gia' ci sono un paio di proxy (celacanto, axolotl).
4. **Mammiferi di taglia/andatura insolita** (formichiere gigante, pangolino,
   okapi) — aumentano la varieta' di proporzioni disponibili.

## Registro attribuzioni

Vedi `attributions.csv` in questa cartella: una riga per ogni immagine
aggiunta al dataset, con fonte/autore/licenza. Necessario se in futuro il
dataset o dei render con quelle texture venissero condivisi pubblicamente.
