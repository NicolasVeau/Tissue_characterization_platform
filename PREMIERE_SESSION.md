# Mise en place sur le Xavier et première session Claude Code

## 1. Déposer les fichiers
Depuis le PC Windows (PowerShell), une fois SSH fonctionnel :
```powershell
scp -r vna_kit nicolas@xavier.local:~/vna_app
```
(ou coller le contenu des deux fichiers dans VS Code Remote-SSH.)

## 2. Initialiser le dépôt
Sur le Xavier :
```bash
cd ~/vna_app
git init
mkdir -p vna_app/{driver,calibration,processing,gui} tests
python3 -m venv --system-site-packages ~/venvs/vna
source ~/venvs/vna/bin/activate
pip install pyserial numpy scipy scikit-rf pyqtgraph pytest ruff
git add -A && git commit -m "Bootstrap: CLAUDE.md + protocole LiteVNA"
claude
```

## 3. Prompt de la première session (à coller tel quel dans Claude Code)
```
Lis CLAUDE.md et docs/protocol.md avant tout.

Implémente la première brique du projet :

1. vna_app/driver/litevna.py : classe LiteVNA (connect/close, device_info,
   configure_sweep(start_hz, stop_hz, points, averaging), sweep() -> (freqs,
   s_raw) avec s_raw complexe shape (N,2) [S11, S21]). Suis exactement la
   séquence décrite dans docs/protocol.md, avec timeouts et gestion du
   désordre des frames par freqIndex.
2. tests/test_frames.py : décodeur de frame testé sur des octets construits
   à la main (struct "<iiiiiiH6x"), y compris averaging > 1 et frames
   désordonnées. Pas besoin du VNA pour ces tests.
3. vna_app/cli.py avec deux sous-commandes : `info` (affiche variant,
   firmware) et `sweep --start --stop --points --avg --out fichier.s2p`
   (écrit un skrf.Network 2 ports, S12 et S22 à zéro, + fichier .json de
   métadonnées à côté).
4. pyproject.toml minimal (ruff, pytest).

Le VNA est branché sur /dev/litevna : une fois les tests unitaires verts,
lance `python -m vna_app.cli info` puis un sweep 400 MHz–3 GHz, 201 points,
et montre-moi |S11| et |S21| en dB aux points 0, 100 et 200. Une charge 50 Ω
est sur le port 1 : S11 doit être nettement en dessous de -10 dB.
Ne passe pas à la GUI dans cette session.
```

## 4. Sessions suivantes (ordre suggéré)
- Calibration SOLT hôte + sauvegarde/chargement `.cal`.
- Thread d'acquisition continue + API streaming.
- GUI PyQt5/pyqtgraph : panneau de paramètres, tracés dB/phase/Smith,
  boutons start/stop/single/save, chargement de calibration.
- Post-traitement spécifique (S11 → impédance → permittivité, fits).

## Astuces Claude Code utiles ici
- `claude --continue` reprend la session précédente.
- `@docs/protocol.md` dans un prompt force la relecture du fichier.
- `/memory` permet d'éditer CLAUDE.md sans quitter la session ; ajoutez-y
  chaque décision prise en cours de route (ex. format du fichier .cal).
- Lancer `claude` dans `tmux` pour survivre à une coupure SSH.
