import argparse
import sys

# cv2 AVANT PyQt5/numpy/scipy : sur aarch64, libgomp chargée par dlopen (import tardif)
# échoue avec « cannot allocate memory in static TLS block » si d'autres bibliothèques
# ont déjà consommé le TLS statique. Importée en premier, elle réserve sa place.
# Absente : la GUI reste utilisable, l'onglet Carto le signalera au démarrage caméra.
try:
    import cv2  # noqa: F401
except ImportError:
    pass

from PyQt5.QtWidgets import QApplication

from vna_app.gui.main_window import MainWindow


def main() -> int:
    ap = argparse.ArgumentParser(prog="vna_app.gui")
    ap.add_argument("--port", default="/dev/litevna")
    args, qt_args = ap.parse_known_args()
    app = QApplication([sys.argv[0]] + qt_args)
    w = MainWindow(args.port)
    w.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
