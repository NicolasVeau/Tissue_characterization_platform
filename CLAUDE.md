# vna_app — acquisition S11/S21 LiteVNA 64 sur Jetson Xavier

## Contexte
Logiciel d'acquisition et de post-traitement de paramètres S pour un banc de
spectroscopie diélectrique (bande utile 0,4–3 GHz). Tourne sur un Jetson Xavier
(JetPack 5.1.x, Ubuntu 20.04, aarch64, Python 3.8) avec un LiteVNA 64 sur USB.
L'utilisateur est ingénieur RF/embarqué : pas besoin de vulgariser, être précis.

## Matériel
- VNA : LiteVNA 64 (hugen), 50 kHz–6,3 GHz, USB 2.0 Full Speed, CDC-ACM.
  Device : `/dev/litevna` (symlink udev vers `/dev/ttyACM0`).
- Protocole : binaire NanoVNA V2 / SAA2 → voir `docs/protocol.md` (référence
  obligatoire avant de toucher au driver).
- Débit : ~550 points/s en S11+S21. Un sweep 1001 pts ≈ 2 s. Max 65535 pts.
- Les données renvoyées sont BRUTES (non calibrées). La calibration SOLT est
  faite côté hôte, jamais dans l'appareil.
- Au-delà de 3 GHz la dynamique chute (>50 dB au lieu de >70 dB) ; la bande
  prioritaire est 0,4–3 GHz.

## Stack imposée
- Python 3.8, venv `~/venvs/vna` (créé avec `--system-site-packages`).
- GUI : **PyQt5 du paquet apt** (`python3-pyqt5`), pas PySide6, pas PyQt6.
- Tracés : `pyqtgraph`. Pas de matplotlib dans la GUI (trop lent sur Jetson) ;
  matplotlib autorisé seulement pour des figures d'export hors ligne.
- pyqtgraph épinglé en 0.12.4 (Qt système = 5.12.8, pyqtgraph ≥ 0.13 exige Qt ≥ 5.15). Ne pas mettre à jour.
- Série : `pyserial`. Calcul : `numpy`, `scipy`. S-params : `scikit-rf`
  (`skrf.Network`, calibration, Touchstone).
- Tests : `pytest`. Lint : `ruff`.

## Architecture
```
vna_app/
  driver/litevna.py      # protocole, thread d'acquisition, queue de frames
  calibration/           # SOLT hôte, kits, sauvegarde/chargement .cal
  processing/            # S11/S21 → Z, ε*, filtres, export
  gui/                   # Qt : paramètres, graphes, contrôle acquisition
  cli.py                 # acquisition sans GUI
  tests/                 # tests unitaires (décodeur de trames sur captures)
docs/protocol.md         # référence protocole USB LiteVNA
```

## Règles de conception
- Toute I/O série vit dans un thread dédié (`QThread` ou `threading.Thread`),
  JAMAIS dans le thread Qt. Communication par `queue.Queue` / signaux Qt.
- Le driver expose une API synchrone simple (`connect`, `configure_sweep`,
  `sweep() -> np.ndarray complexe (N,2)`) + une API asynchrone en streaming.
- Les mesures sont stockées en `skrf.Network` (fréquences en Hz, z0 = 50 Ω).
  Export Touchstone `.s1p` / `.s2p` + métadonnées JSON (date, points, averaging,
  firmware, fichier de calibration utilisé).
- Toujours lire `valuesPerFrequency` et `sweepPoints` après écriture pour
  vérifier que l'appareil a bien accepté les paramètres.
- Gérer proprement : appareil absent, débranchement en cours de sweep, timeout,
  frames FIFO désordonnées (utiliser l'index de fréquence, pas l'ordre d'arrivée).
- Pas de dépendance nouvelle sans la justifier et vérifier qu'elle s'installe
  sur aarch64 / Python 3.8.

## Workflow de validation
Avant de proposer une modification comme terminée :
1. `ruff check .` et `pytest` passent.
2. Si le driver ou le CLI est touché et que le VNA est branché :
   `python -m vna_app.cli info` puis
   `python -m vna_app.cli sweep --start 400e6 --stop 3e9 --points 201 --out /tmp/test.s2p`
   et vérifier que le fichier est lisible par `skrf.Network('/tmp/test.s2p')`.
3. Pour la GUI : lancer `python -m vna_app.gui` et confirmer que la fenêtre
   s'ouvre sans erreur (écran local du Xavier ou DISPLAY=:0).

## Commandes utiles
- Activer l'env : `source ~/venvs/vna/bin/activate`
- Identifier le VNA : `ls -l /dev/litevna`, `lsusb | grep -i 04b4:0008` (ZeenKo.tech « LiteVNA6 »)
- Test de vie : `python -c "import serial;s=serial.Serial('/dev/litevna',timeout=1);s.write(b'\x0d');print(s.read(1))"` → doit afficher `b'2'` (0x32)

## Hors périmètre pour l'instant
- Pas de calcul GPU (CuPy) tant que le post-traitement CPU suffit.
- Pas de serveur web / API réseau ; GUI locale uniquement.
