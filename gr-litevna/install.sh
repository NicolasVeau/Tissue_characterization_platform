#!/usr/bin/env bash
# Installation permanente des blocs LiteVNA pour GNU Radio Companion (3.8 et 3.10).
#
#   bash gr-litevna/install.sh            installe (liens symboliques, rien n'est copié)
#   bash gr-litevna/install.sh --check    diagnostic seul, ne modifie rien
#   bash gr-litevna/install.sh --remove   désinstalle
#
# Contrairement à env.sh, l'effet ne dépend pas du terminal d'où GRC est lancé :
#  1. les .block.yml sont liés dans ~/.grc_gnuradio, dossier que GRC lit toujours ;
#  2. un fichier .pth dans le site-packages utilisateur rend gr_litevna et vna_app
#     importables par le Python de GNU Radio (/usr/bin/python3, PAS celui du venv :
#     gnuradio-companion et les flowgraphs générés ignorent le venv actif).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
PY="${GR_PYTHON:-/usr/bin/python3}"
GRC_DIR="${GRC_HIER_PATH:-$HOME/.grc_gnuradio}"
MODE="${1:-install}"
FAIL=0
ok()  { printf '  [ok]    %s\n' "$*"; }
bad() { printf '  [ECHEC] %s\n' "$*"; FAIL=1; }

if ! "$PY" -c 'import sys' 2>/dev/null; then
    echo "Python introuvable : $PY (définir GR_PYTHON=/chemin/vers/python3)"
    exit 2
fi
USER_SITE="$("$PY" -c 'import site; print(site.getusersitepackages())')"
PTH="$USER_SITE/gr_litevna_dev.pth"

if [ "$MODE" = "--remove" ]; then
    for f in "$HERE"/grc/*.block.yml; do
        link="$GRC_DIR/$(basename "$f")"
        [ -L "$link" ] && unlink "$link"
    done
    [ -f "$PTH" ] && unlink "$PTH"
    echo "Blocs LiteVNA désinstallés."
    exit 0
fi

if [ "$MODE" != "--check" ]; then
    mkdir -p "$GRC_DIR" "$USER_SITE"
    for f in "$HERE"/grc/*.block.yml; do ln -sf "$f" "$GRC_DIR/$(basename "$f")"; done
    printf '%s\n%s\n' "$HERE" "$REPO" > "$PTH"
    echo "Installé : liens dans $GRC_DIR, chemins dans $PTH"
fi

echo "Diagnostic (Python de GNU Radio : $PY)"
n=0
for f in "$HERE"/grc/*.block.yml; do
    [ -e "$GRC_DIR/$(basename "$f")" ] && n=$((n + 1))
done
if [ "$n" -eq 4 ]; then
    ok "4 définitions de blocs visibles dans $GRC_DIR"
elif echo ":${GRC_BLOCKS_PATH:-}:" | grep -q ":$HERE/grc:"; then
    ok "blocs visibles par GRC_BLOCKS_PATH (dans ce terminal seulement)"
else
    bad "blocs absents de $GRC_DIR et de GRC_BLOCKS_PATH : GRC ne les affichera pas"
fi

"$PY" -c 'from gnuradio import gr; print("  [ok]    GNU Radio", gr.version())' 2>/dev/null \
    || bad "gnuradio non importable par $PY (sudo apt install gnuradio)"
for mod in numpy serial; do
    if "$PY" -c "import $mod" 2>/dev/null; then
        ok "module $mod"
    else
        bad "module $mod absent pour $PY (sudo apt install python3-$mod)"
    fi
done
if (cd / && "$PY" -c 'import vna_app.driver.sim, vna_app.processing.record' 2>/dev/null); then
    ok "vna_app importable"
else
    bad "vna_app non importable par $PY"
fi
ERR="$(cd / && "$PY" -c 'import gr_litevna' 2>&1)"
if [ -z "$ERR" ]; then
    ok "gr_litevna importable"
else
    bad "gr_litevna non importable par $PY :"
    printf '%s\n' "$ERR" | tail -2 | sed 's/^/          /'
fi

if [ "$FAIL" -eq 0 ]; then
    echo "Tout est en place. Relancer gnuradio-companion : catégorie « LiteVNA » dans la liste des blocs."
else
    echo "Corriger les lignes [ECHEC] ci-dessus, puis relancer ce script."
fi
exit "$FAIL"
