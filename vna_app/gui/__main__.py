import argparse
import sys

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
