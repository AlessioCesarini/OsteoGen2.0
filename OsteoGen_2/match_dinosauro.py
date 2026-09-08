import json
from inference import estrai_coordinate

def calcola_distanza(p1, p2):
    if p1 is None or p2 is None: return 0.0
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def calcola_proporzioni(coords):
    # Calcola le lunghezze base
    parti = {
        "testa": calcola_distanza(coords.get("punta_muso"), coords.get("base_collo")),
        "torso": calcola_distanza(coords.get("base_collo"), coords.get("base_coda")),
        "arto_anteriore": calcola_distanza(coords.get("spalla"), coords.get("zampa_ant")),
        "arto_posteriore": calcola_distanza(coords.get("anca"), coords.get("zampa_post")),
        "coda": calcola_distanza(coords.get("base_coda"), coords.get("punta_coda"))
    }
    
    # Normalizza rispetto al torso (invarianza di scala)
    torso_len = parti["torso"] if parti["torso"] > 0 else 1.0
    proporzioni = {k: v / torso_len for k, v in parti.items()}
    return proporzioni

def esegui_retrieval(trex_img_path, pesi_modello, db_json_path):
    print("Estrazione coordinate del target in corso...")
    coords_trex = estrai_coordinate(trex_img_path, pesi_modello, threshold=0.30)
    prop_trex = calcola_proporzioni(coords_trex)

    print("\nProporzioni Target calcolate (relative al torso):")
    for k, v in prop_trex.items():
        print(f" - {k}: {v:.2f}x")

    with open(db_json_path, 'r') as f:
        db = json.load(f)

    match_finali = {}

    print("\n--- RISULTATI MATCH BIOLOGICO ---")
    for nome_parte in prop_trex.keys():
        if prop_trex[nome_parte] == 0.0:
            print(f"{nome_parte.upper()}: Assente nel target.")
            continue

        best_match = None
        min_diff = float('inf')

        for nome_animale, dati_animale in db.items():
            torso_anim = dati_animale["segmentazione"]["torso"]["lunghezza_base"]
            if torso_anim == 0: continue

            len_parte_anim = dati_animale["segmentazione"][nome_parte]["lunghezza_base"]
            prop_anim = len_parte_anim / torso_anim
            # Distanza euclidea 1D tra le proporzioni
            diff = abs(prop_trex[nome_parte] - prop_anim)
            
            if diff < min_diff:
                min_diff = diff
                best_match = nome_animale

        match_finali[nome_parte] = best_match
        print(f"{nome_parte.upper()}: {best_match} (Diff: {min_diff:.3f})")

    return match_finali

if __name__ == "__main__":
    # 1. SCARICA UNO SCHELETRO DI T-REX SU SFONDO NERO
    # 2. INSERISCI QUI IL PERCORSO:
    trex_path = r"C:\Users\alexc\Desktop\OsteoGen2.0\OsteoGen2.0\OsteoGen_2\tests\t-rex.jpg"
    
    pesi = r"C:\Users\alexc\Desktop\OsteoGen2.0\OsteoGen2.0\OsteoGen_2\training_outputs_2\Weights\best_keypoint_detector.pth"
    db_path = r"C:\Users\alexc\Desktop\OsteoGen2.0\OsteoGen2.0\OsteoGen_2\data\processed\geometric_database.json"

    esegui_retrieval(trex_path, pesi, db_path)