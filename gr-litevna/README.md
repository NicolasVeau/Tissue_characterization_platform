# gr-litevna — le LiteVNA 64 comme source temps réel pour GNU Radio

Blocs GNU Radio Companion qui délivrent S11/S21 en continu, pour explorer des
stratégies de traitement du signal. Enveloppe mince autour de `vna_app` :
streaming du driver (`LiteVNA.start_streaming()`), calibration (`Calibration.apply()`),
simulateur (`vna_app.driver.sim`) et enregistrements (`vna_app.processing.record`).

Blocs en Python pur, sans compilation : ni CMake ni `gr_modtool`. Le module se
charge par deux variables d'environnement.

## Mise en route

```bash
sudo apt install gnuradio python3-serial   # Ubuntu 20.04 : GNU Radio 3.8
bash gr-litevna/install.sh                 # installe puis affiche un diagnostic
gnuradio-companion flowgraphs/01_visu_s11_s21.grc
```

`install.sh` lie les définitions de blocs dans `~/.grc_gnuradio` (dossier que GRC lit
toujours) et ajoute un fichier `.pth` au site-packages utilisateur, pour que
`gr_litevna` et `vna_app` soient importables par le Python de GNU Radio.
`bash gr-litevna/install.sh --check` rejoue le diagnostic sans rien modifier,
`--remove` désinstalle.

`gnuradio-companion` et les flowgraphs qu'il génère tournent avec `/usr/bin/python3`,
pas avec le Python du venv, même si le venv est actif. Les dépendances des blocs
(`numpy`, `pyserial`) doivent donc exister pour le Python système.

Alternative sans installation, valable pour le seul terminal courant :
`source gr-litevna/env.sh` puis `gnuradio-companion` depuis ce même terminal.

Les blocs apparaissent dans la catégorie **LiteVNA**. Sans banc, mettre `sim` comme port :

```bash
grcc -o /tmp flowgraphs/01_visu_s11_s21.grc
python /tmp/litevna_01_visu.py --port sim:drift=4e6,noise=2e-3
```

La GUI `vna_app` et un flowgraph ne peuvent pas tenir `/dev/litevna` en même temps.

## Blocs

| Bloc | Sorties | Usage |
|---|---|---|
| LiteVNA Sweep Source | S11, S21 : vecteurs complex64 de `points` valeurs ; PDU `sweep` | Un item = un sweep |
| LiteVNA CW Source | S11, S21 : flux scalaires complex64, bruts | Fréquence fixe (start = stop) |
| LiteVNA Replay Source | comme Sweep Source, depuis un `.npz` | Développer et comparer hors banc |
| LiteVNA Sweep Sink | — (écrit un `.npz`) | Enregistrer depuis un flowgraph |

Tags posés sur S11 à chaque sweep : `rx_time` (tuple secondes/fraction, heure de fin
du sweep), `sweep_idx`, `f_start`, `f_step`, `calibrated`. Le PDU `sweep` porte les
mêmes métadonnées et un vecteur c32 de 2N valeurs (S11 puis S21).

Port message `cmd` : dictionnaire `{f_start, f_stop, averaging}` (Sweep Source) ou
`{freq, averaging}` (CW Source). `points` est fixe : c'est la longueur des vecteurs.

Port simulé : `sim` ou `sim:rate=300,noise=1e-3,drift=1e5,seed=1` (résonateur à
1,5 GHz dont la fréquence dérive de `drift` Hz/s ; `rate=0` = sans limite de débit).

### Comportements à connaître

- **Débit** : celui de l'appareil, ~300 points/s. 201 points ≈ 1,5 sweep/s.
- **Sweeps écartés** : si le flowgraph consomme moins vite que l'appareil ne produit,
  le driver écarte les plus anciens ; les trous se lisent dans `sweep_idx`.
- **Calibration** : le `.cal` doit être sur la même grille que le sweep, sinon la
  construction du bloc est refusée. Après un changement de bande à chaud, la sortie
  repasse en brut (`calibrated = False`). Le bloc CW sort toujours du brut.
- **Erreur d'appareil** (absent, débranché) : journalisée, gardée dans `bloc.error`,
  et le flowgraph se termine. Les blocs ne lèvent jamais d'exception dans `start()` :
  GNU Radio 3.10 s'y bloque.
- **QT GUI Vector Sink** : constaté sous GNU Radio 3.10.9, il ne trace rien s'il
  reçoit un seul vecteur par appel, ce qui est le cas à quelques sweeps par seconde.
  Les flowgraphs fournis dupliquent chaque sweep (`Repeat` ×2) juste avant l'affichage.
  Faire de même dans tout nouveau flowgraph ; les autres afficheurs (Time Sink,
  Time Raster) ne sont pas concernés.
- **CW** : l'échantillonnage est régulier dans une acquisition de `block_len`
  échantillons ; un court trou sépare deux acquisitions (relance du balayage).

## Flowgraphs (`flowgraphs/`)

| Fichier | Contenu |
|---|---|
| `01_visu_s11_s21.grc` | Module (dB) et phase de S11 et S21 |
| `02_waterfall.grc` | \|S11\| au fil des sweeps (raster) et dernier sweep |
| `03_ifft_gating.grc` | Réponse temporelle (IFFT), porte réglable, retour en fréquence |
| `04_diff.grc` | S11 moins une ligne de base glissante (IIR par point de fréquence) |
| `05_cw.grc` | Module et phase de S11 à fréquence fixe, en fonction du temps |
| `06_replay.grc` | Relecture d'un enregistrement, même affichage que 01 |

Paramètres en ligne de commande : `--port`, `--f-start`, `--f-stop`, `--points`,
`--averaging`, `--cal-file` (01 à 04) ; `--filename` (06).

## Enregistrer puis rejouer

```bash
# série brute, double précision, sans GNU Radio
python -m vna_app.cli record --start 400e6 --stop 3e9 --points 201 --count 100 --out data/serie.npz
# jeu de démonstration sans banc
python -m vna_app.cli --port sim:rate=0,drift=3e6,noise=2e-3 record \
    --start 400e6 --stop 3e9 --points 201 --count 60 --out data/demo_sim.npz
python /tmp/litevna_06_replay.py --filename data/demo_sim.npz
```

LiteVNA Sweep Sink enregistre au même format depuis un flowgraph, en complex64.

## Tests

```bash
pytest                                   # vna_app : simulateur, enregistreur, CLI (sans GNU Radio)
pytest gr-litevna/tests/qa_blocks.py     # blocs, avec le Python de GNU Radio, sur simulateur
```

## État de validation

Vérifié sous GNU Radio 3.10.9 (Ubuntu 24.04, Python 3.12) avec le simulateur : tests
des blocs, compilation `grcc` des six flowgraphs, exécution et affichage.

Reste à vérifier sur le banc :

1. **GNU Radio 3.8** (paquet apt du Xavier) : les blocs n'utilisent que l'API Python
   commune aux deux versions, mais rien n'a été exécuté sous 3.8. Les `.grc` ont été
   écrits par GRC 3.10 ; en cas de refus à l'ouverture, les régénérer sous 3.8.
2. **Matériel** : aucun test sur le LiteVNA réel.
3. **Fréquence fixe** : `python gr-litevna/tools/check_fixed_freq.py` dit si le
   firmware accepte un pas nul et donne la cadence obtenue.
