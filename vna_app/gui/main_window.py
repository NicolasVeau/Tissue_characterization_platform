from __future__ import annotations

import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from vna_app.calibration import CalibrationError
from vna_app.gui.scanview import ScanView
from vna_app.gui.smith import SmithChart
from vna_app.gui.waterfall import Waterfall
from vna_app.gui.worker import Worker
from vna_app.processing.classifier import (
    Dataset,
    ZoneClassifier,
    cross_validate,
    features_for,
    format_report,
    grid_signature,
)
from vna_app.processing.compare import compare, mod_db, snapshot, unwrap_deg
from vna_app.processing.export import write_s2p
from vna_app.processing.impedance import gamma_to_z
from vna_app.processing.permittivity import REFERENCE_LIQUIDS, to_eps_columns
from vna_app.processing.pipeline import Pipeline
from vna_app.processing.pressure import contact_level, normalize_shape, passes_contact
from vna_app.processing.sweepfilter import SweepFilter
from vna_app.vision.fusion import PositionBuffer, ScanMap, summarize_eps

EPS_MODES = [("off", "Désactivé"), ("capacitive", "Capacitif (C0)"), ("three_ref", "3 références")]
CAL_LABELS = {"open": "Open", "short": "Short", "load": "Load", "thru": "Thru (S21)"}
EPS_LABELS = {"air": "Air", "probe_short": "Court-circuit", "ref": "Liquide de réf."}


def _db(x):
    with np.errstate(divide="ignore"):
        return 20 * np.log10(np.abs(x))


def _phase_unwrapped_deg(x):
    return np.degrees(np.unwrap(np.angle(x)))


ZONES = [
    "nez",
    "joue droite",
    "joue gauche",
    "index droit",
    "majeur droit",
    "index gauche",
    "majeur gauche",
    "avant-bras droit",
    "avant-bras gauche",
    "cuisse droite",
    "cuisse gauche",
    "paume",
]

LAST_CAL = Path(
    os.environ.get("VNA_APP_LAST_CAL") or Path.home() / ".config" / "vna_app" / "last_cal"
)


class MainWindow(QMainWindow):
    _connect = pyqtSignal()
    _disconnect = pyqtSignal()
    _configure = pyqtSignal(float, float, int, int)
    _run = pyqtSignal(bool)
    _power = pyqtSignal(int, int)

    def __init__(self, port: str = "/dev/litevna"):
        super().__init__()
        self.setWindowTitle("vna_app — LiteVNA 64")
        self.resize(1500, 800)
        self.info = None
        self.last = None  # (freqs, s_raw, params)
        self.result = None  # processing.pipeline.Result
        self.connected = False
        self.pipeline = Pipeline()
        self.pending = None  # standard à capturer au prochain sweep

        self.mem = {1: None, 2: None}  # Snapshot ou None
        tabs = QTabWidget()
        tabs.addTab(self._build_acq_tab(), "Acq.")
        tabs.addTab(self._build_cal_tab(), "Cal.")
        self.eps_tab_index = tabs.addTab(self._build_eps_tab(), "ε*")
        self.cmp_tab_index = tabs.addTab(self._build_cmp_tab(), "Comp.")
        self.cls_tab_index = tabs.addTab(self._build_cls_tab(), "Zone")
        self.map_tab_index = tabs.addTab(self._build_map_tab(), "Carto")
        tabs.setFixedWidth(310)
        self.status = QLabel("Déconnecté")
        self.status.setWordWrap(True)
        side = QVBoxLayout()
        side.addWidget(tabs, 1)
        side.addWidget(self.status)
        side_w = QWidget()
        side_w.setLayout(side)
        side_w.setFixedWidth(310)

        self.plot11 = pg.PlotWidget(title="|Z| (brut)")
        self.plot21 = pg.PlotWidget(title="Phase Z déroulée (brut)")
        self.plot_eps = pg.PlotWidget(title="ε*")
        for p in (self.plot11, self.plot21, self.plot_eps):
            p.setLabel("bottom", "Fréquence", units="Hz")
            p.showGrid(x=True, y=True)
        self.plot11.setLabel("left", "|Z|", units="dBΩ")
        self.plot21.setLabel("left", "Phase Z déroulée", units="°")
        self.plot_eps.setLabel("left", "ε', ε''")
        self.plot21.setXLink(self.plot11)
        self.plot_eps.addLegend()
        self.c11 = self.plot11.plot(pen=pg.mkPen("y", width=2))
        self.c21 = self.plot21.plot(pen=pg.mkPen("c", width=2))
        self.ceps1 = self.plot_eps.plot(pen=pg.mkPen("g", width=2), name="ε'")
        self.ceps2 = self.plot_eps.plot(pen=pg.mkPen("m", width=2), name="ε''")
        self.plot_eps.setVisible(False)
        # onglet ε* : colonne gauche = Smith + spectre ε* ; colonne droite = UN waterfall au
        # choix : ε' en couleur + isocontours ε'' (opt. 1) ou tan δ (opt. 2)
        depth = self.wf_depth.value()
        self.wf_iso = Waterfall(
            "ε' (couleur) + isocontours ε''",
            "ε'",
            depth=depth,
            aux_label="ε''",
            n_iso=self.wf_niso.value(),
        )
        self.wf_tan = Waterfall("tan δ = ε''/ε' par sweep", "tan δ", depth=depth)
        self.waterfalls = (self.wf_iso, self.wf_tan)
        self.on_wf_scale()
        self.smith_eps = SmithChart()
        left = QVBoxLayout()
        left.addWidget(self.smith_eps, 2)
        left.addWidget(self.plot_eps, 1)
        right = QVBoxLayout()
        for w in (self.wf_iso, self.wf_tan):
            right.addWidget(w, 1)
        eps_cols = QHBoxLayout()
        eps_cols.addLayout(left, 1)
        eps_cols.addLayout(right, 1)
        eps_page = QWidget()
        eps_page.setLayout(eps_cols)
        self.wf_select.currentIndexChanged.connect(self.on_wf_select)
        self.on_wf_select(self.wf_select.currentIndex())

        plots = QVBoxLayout()
        for p in (self.plot11, self.plot21):
            plots.addWidget(p)

        self.smith = SmithChart()
        root = QHBoxLayout()
        root.addWidget(side_w)
        live = QWidget()
        live.setLayout(plots)
        self.stack = QStackedWidget()
        self.stack.addWidget(live)  # 0 : acquisition / calibration
        self.stack.addWidget(self._build_cmp_plots())  # 1 : comparaison
        self.stack.addWidget(eps_page)  # 2 : ε*
        self.stack.addWidget(self._build_map_page())  # 3 : cartographie caméra
        root.addWidget(self.stack, 3)
        tabs.currentChanged.connect(self.on_tab_changed)
        root.addWidget(self.smith, 2)
        c = QWidget()
        c.setLayout(root)
        self.setCentralWidget(c)

        # Thread d'acquisition
        self.thread = QThread(self)
        self.worker = Worker(port)
        self.worker.moveToThread(self.thread)
        self._connect.connect(self.worker.connect_device)
        self._disconnect.connect(self.worker.disconnect_device)
        self._configure.connect(self.worker.configure)
        self._run.connect(self.worker.run)
        self._power.connect(self.worker.set_power)
        self.worker.connected.connect(self.on_connected)
        self.worker.disconnected.connect(self.on_disconnected)
        self.worker.sweep_done.connect(self.on_sweep)
        self.worker.error.connect(self.on_error)
        self.thread.start()
        self.load_last_cal()
        self.refresh_buttons()

    # -- construction des onglets ----------------------------------------------
    def _build_acq_tab(self):
        self.start_mhz = QDoubleSpinBox()
        self.start_mhz.setRange(0.05, 6300)
        self.start_mhz.setValue(400)
        self.stop_mhz = QDoubleSpinBox()
        self.stop_mhz.setRange(0.05, 6300)
        self.stop_mhz.setValue(3000)
        for w in (self.start_mhz, self.stop_mhz):
            w.setDecimals(6)
            w.setSuffix(" MHz")
        self.points = QSpinBox()
        self.points.setRange(2, 65535)
        self.points.setValue(201)
        self.avg = QSpinBox()
        self.avg.setRange(1, 255)
        self.avg.setValue(1)
        self.pwr_high = self._power_combo(3, "Puissance de la source au-dessus de ~140 MHz")
        self.pwr_low = self._power_combo(1, "Puissance de la source en dessous de ~140 MHz")
        self.btn_conn = QPushButton("Connecter")
        self.btn_single = QPushButton("Sweep unique")
        self.btn_cont = QPushButton("Continu")
        self.btn_cont.setCheckable(True)
        self.btn_save = QPushButton("Enregistrer .s2p…")
        self.btn_conn.clicked.connect(self.toggle_connect)
        self.btn_single.clicked.connect(lambda: self.start(False))
        self.btn_cont.toggled.connect(self.toggle_continuous)
        self.btn_save.clicked.connect(self.save)
        form = QFormLayout()
        form.addRow("Début", self.start_mhz)
        form.addRow("Fin", self.stop_mhz)
        form.addRow("Points", self.points)
        form.addRow("Moyenne", self.avg)
        form.addRow("Puissance > 140 MHz", self.pwr_high)
        form.addRow("Puissance < 140 MHz", self.pwr_low)
        self.cal_lock = QLabel("")
        self.cal_lock.setWordWrap(True)
        self.filt = SweepFilter()
        self.chk_filt = QCheckBox("Rejeter les sweeps incohérents")
        self.chk_filt.setToolTip(
            "Rejette un sweep si |Γ_fin − Γ_début| sort de [dmin, dmax] ou si un point "
            "intermédiaire est à plus de R des deux extrémités (S11, calibré si appliqué)."
        )
        self.filt_sp = {}
        fform = QFormLayout()
        for key, label, val in (
            ("d_min", "Distance extrémités min", self.filt.d_min),
            ("d_max", "Distance extrémités max", self.filt.d_max),
            ("radius", "Voisinage R", self.filt.radius),
        ):
            sp = QDoubleSpinBox()
            sp.setRange(0.0, 10.0)
            sp.setDecimals(3)
            sp.setSingleStep(0.05)
            sp.setValue(val)
            sp.valueChanged.connect(self.on_filter)
            self.filt_sp[key] = sp
            fform.addRow(label, sp)
        self.filt_info = QLabel("")
        self.chk_filt.toggled.connect(self.on_filter)
        lay = QVBoxLayout()
        lay.addWidget(self.btn_conn)
        lay.addLayout(form)
        lay.addWidget(self.cal_lock)
        lay.addWidget(self.chk_filt)
        lay.addLayout(fform)
        lay.addWidget(self.filt_info)
        lay.addWidget(self.btn_single)
        lay.addWidget(self.btn_cont)
        lay.addWidget(self.btn_save)
        lay.addStretch(1)
        w = QWidget()
        w.setLayout(lay)
        return w

    def _build_cal_tab(self):
        lay = QVBoxLayout()
        lay.addWidget(QLabel("Un sweep par standard (mêmes paramètres que la mesure) :"))
        self.cal_buttons = {}
        for k, label in CAL_LABELS.items():
            b = QPushButton(f"Capturer {label}")
            b.clicked.connect(lambda _=False, k=k: self.capture(k))
            self.cal_buttons[k] = b
            lay.addWidget(b)
        self.cal_captured = QLabel("Capturés : —")
        self.cal_captured.setWordWrap(True)
        self.btn_build = QPushButton("Calculer la calibration…")
        self.btn_load_cal = QPushButton("Charger un .cal…")
        self.chk_cal = QCheckBox("Appliquer la calibration")
        self.cal_info = QLabel("Aucune calibration")
        self.cal_info.setWordWrap(True)
        self.delay_ns = QDoubleSpinBox()
        self.delay_ns.setRange(0.0, 100.0)
        self.delay_ns.setDecimals(3)
        self.delay_ns.setSingleStep(0.1)
        self.delay_ns.setSuffix(" ns")
        self.delay_ns.setToolTip(
            "Délai simple trajet du câble coaxial entre le plan de calibration et le capteur "
            "(≈ 5 ns/m pour un câble PE, VF 0,66). Appliqué à S11, aller-retour."
        )
        self.delay_ns.valueChanged.connect(self.on_delay)
        self.delay_len = QLabel("")
        dform = QFormLayout()
        dform.addRow("Délai câble (simple trajet)", self.delay_ns)
        dform.addRow("≈ longueur (VF 0,66)", self.delay_len)
        self.on_delay(0.0)
        self.btn_build.clicked.connect(self.build_cal)
        self.btn_load_cal.clicked.connect(self.load_cal)
        self.chk_cal.toggled.connect(self.on_apply_cal)
        for w in (
            self.cal_captured,
            self.btn_build,
            self.btn_load_cal,
            self.chk_cal,
            self.cal_info,
        ):
            lay.addWidget(w)
        lay.addLayout(dform)
        lay.addStretch(1)
        w = QWidget()
        w.setLayout(lay)
        return w

    def _build_eps_tab(self):
        self.eps_mode = QComboBox()
        for key, label in EPS_MODES:
            self.eps_mode.addItem(label, key)
        self.c0_ff = QDoubleSpinBox()
        self.c0_ff.setRange(0.001, 100000)
        self.c0_ff.setDecimals(3)
        self.c0_ff.setValue(80.0)
        self.c0_ff.setSuffix(" fF")
        self.ref_liquid = QComboBox()
        self.ref_liquid.addItems(sorted(REFERENCE_LIQUIDS))
        self.smooth = QSpinBox()
        self.smooth.setRange(1, 51)
        self.smooth.setSingleStep(2)
        form = QFormLayout()
        form.addRow("Méthode", self.eps_mode)
        form.addRow("C0 (capacitif)", self.c0_ff)
        form.addRow("Liquide de réf.", self.ref_liquid)
        form.addRow("Lissage (impair)", self.smooth)
        self.chk_norm = QCheckBox("ε* normalisé (invariant au contact partiel)")
        self.chk_norm.setToolTip(
            "Trace u = (ε*−1)/(ε*−1)(f_réf) : Re u et −Im u. Supprime le facteur d'échelle "
            "dû à la fraction de contact (pression). Sans effet sur les waterfalls."
        )
        self.f_ref_mhz = QDoubleSpinBox()
        self.f_ref_mhz.setRange(0.05, 6300)
        self.f_ref_mhz.setDecimals(1)
        self.f_ref_mhz.setValue(1000.0)
        self.f_ref_mhz.setSuffix(" MHz")
        self.contact_min = QDoubleSpinBox()
        self.contact_min.setRange(0.0, 1000.0)
        self.contact_min.setDecimals(2)
        self.contact_min.setSingleStep(0.5)
        self.contact_min.setToolTip(
            "Rejette les sweeps dont le niveau de contact médian |ε*−1| est inférieur (0 = off)"
        )
        self.contact_lbl = QLabel("Contact : —")
        self.n_contact_rej = 0
        form.addRow("f réf. normalisation", self.f_ref_mhz)
        form.addRow("Contact min |ε*−1|", self.contact_min)
        lay = QVBoxLayout()
        lay.addLayout(form)
        lay.addWidget(self.chk_norm)
        lay.addWidget(self.contact_lbl)
        lay.addWidget(QLabel("Références (méthode 3 références, S11 brut) :"))
        self.eps_buttons = {}
        for k, label in EPS_LABELS.items():
            b = QPushButton(f"Capturer {label}")
            b.clicked.connect(lambda _=False, k=k: self.capture(k))
            self.eps_buttons[k] = b
            lay.addWidget(b)
        self.eps_captured = QLabel("Capturées : —")
        self.eps_captured.setWordWrap(True)
        self.btn_save_eps = QPushButton("Enregistrer ε* (.csv)…")
        lay.addWidget(self.eps_captured)
        lay.addWidget(self.btn_save_eps)

        # waterfalls : historique et calibration des couleurs
        lay.addWidget(QLabel("Waterfall (colonne de droite) :"))
        self.wf_select = QComboBox()
        self.wf_select.addItem("ε' + isocontours ε''")
        self.wf_select.addItem("tan δ = ε''/ε'")
        self.wf_depth = QSpinBox()
        self.wf_depth.setRange(5, 2000)
        self.wf_depth.setValue(100)
        self.wf_auto = QCheckBox("Couleurs auto (1–99 %)")
        self.wf_auto.setChecked(True)
        self.wf_lim = {}
        wform = QFormLayout()
        wform.addRow("Affichage", self.wf_select)
        wform.addRow("Sweeps conservés", self.wf_depth)
        for key, lo, hi in (("re", 1.0, 100.0), ("tan", 0.0, 1.0)):
            for end, v in (("min", lo), ("max", hi)):
                sp = QDoubleSpinBox()
                sp.setRange(-1e5, 1e5)
                sp.setDecimals(2)
                sp.setValue(v)
                sp.setButtonSymbols(QAbstractSpinBox.NoButtons)
                sp.setEnabled(False)
                sp.valueChanged.connect(self.on_wf_scale)
                self.wf_lim[key, end] = sp
        for key, name in (("re", "ε'"), ("tan", "tan δ")):
            lo, hi = self.wf_lim[key, "min"], self.wf_lim[key, "max"]
            wform.addRow(f"{name} min / max", self._pair(lo, hi))
        self.wf_niso = QSpinBox()
        self.wf_niso.setRange(0, 30)
        self.wf_niso.setValue(6)
        self.wf_niso.setToolTip("Isocontours de ε'' sur le waterfall ε' (0 = aucun)")
        wform.addRow("Isocontours ε''", self.wf_niso)
        btn_wf_clear = QPushButton("Effacer les waterfalls")
        btn_wf_clear.clicked.connect(self.clear_waterfalls)
        lay.addWidget(self.wf_auto)
        lay.addLayout(wform)
        lay.addWidget(btn_wf_clear)
        self.wf_auto.toggled.connect(self.on_wf_scale)
        self.wf_depth.valueChanged.connect(self.on_wf_depth)
        self.wf_niso.valueChanged.connect(lambda n: self.wf_iso.set_n_iso(n))
        lay.addStretch(1)
        self.eps_mode.currentIndexChanged.connect(self.on_eps_params)
        self.c0_ff.valueChanged.connect(self.on_eps_params)
        self.ref_liquid.currentIndexChanged.connect(self.on_eps_params)
        self.smooth.valueChanged.connect(self.on_eps_params)
        self.chk_norm.toggled.connect(self.on_norm)
        self.f_ref_mhz.valueChanged.connect(self.on_norm)
        self.contact_min.valueChanged.connect(self.on_contact_min)
        self.btn_save_eps.clicked.connect(self.save_eps)
        w = QWidget()
        w.setLayout(lay)
        return w

    # -- cartographie caméra (RealSense RGB + bague verte + ε* au contact)
    def _build_map_tab(self):
        self.cam_stream = None  # vision.camera.CameraStream ou None
        self.cam_tracker = None  # vision.tracker.RingTracker pendant la capture
        self.scan_positions = PositionBuffer()
        self.scan_map = ScanMap()
        self.trend = deque(maxlen=1000)  # (t, ε', ε'') par sweep
        self.cam_timer = QTimer(self)
        self.cam_timer.setInterval(66)  # ~15 Hz d'affichage ; la capture tourne à 30
        self.cam_timer.timeout.connect(self.poll_camera)

        self.cam_device = QComboBox()
        self.cam_device.setEditable(True)
        self.cam_device.addItem("/dev/video4")  # nœud UVC couleur de la RealSense D4xx
        self.btn_cam = QPushButton("Démarrer la caméra")
        self.btn_cam.setCheckable(True)
        self.btn_cam.toggled.connect(self.toggle_camera)
        self.chk_cam_auto = QCheckBox("Expo / WB auto")
        self.chk_cam_auto.setToolTip(
            "Déconseillé : en auto la teinte dérive et le seuil HSV de la bague décroche."
        )
        self.cam_expo = QSpinBox()
        self.cam_expo.setRange(1, 5000)
        self.cam_expo.setValue(156)
        self.cam_expo.setToolTip("Exposition V4L2 (unités de 100 µs)")
        self.cam_wb = QSpinBox()
        self.cam_wb.setRange(2000, 8000)
        self.cam_wb.setSingleStep(100)
        self.cam_wb.setValue(4600)
        self.cam_wb.setSuffix(" K")
        for w in (self.chk_cam_auto, self.cam_expo, self.cam_wb):
            (w.toggled if w is self.chk_cam_auto else w.valueChanged).connect(self.on_cam_controls)

        self.trk_sp = {}
        tform = QFormLayout()
        # défauts = mesures sur la bague turquoise réelle (voir vision/tracker.py)
        for key, label, lo, hi, val in (
            ("h_center", "Teinte H centre (turquoise = 95)", 0, 179, 95),
            ("h_width", "Teinte H ± (gaine bleue à H≈105 !)", 1, 60, 5),
            ("s_min", "Saturation min", 0, 255, 120),
            ("v_min", "Valeur min", 0, 255, 100),
            ("min_area", "Aire min (px²)", 1, 100000, 40),
        ):
            sp = QSpinBox()
            sp.setRange(lo, hi)
            sp.setValue(val)
            sp.valueChanged.connect(self.on_tracker_params)
            self.trk_sp[key] = sp
            tform.addRow(label, sp)
        self.tip_dx = QSpinBox()
        self.tip_dy = QSpinBox()
        for sp in (self.tip_dx, self.tip_dy):
            sp.setRange(-400, 400)
        self.tip_dx.setToolTip("Décalage bague → extrémité de la canule, en pixels image")
        tform.addRow("Décalage pointe dx / dy (px)", self._pair(self.tip_dx, self.tip_dy))

        self.map_flo = QDoubleSpinBox()
        self.map_fhi = QDoubleSpinBox()
        for sp, v in ((self.map_flo, 500.0), (self.map_fhi, 2500.0)):
            sp.setRange(0.05, 6300)
            sp.setDecimals(1)
            sp.setValue(v)
            sp.setSuffix(" MHz")
        self.map_contact = QDoubleSpinBox()
        self.map_contact.setRange(0.0, 1000.0)
        self.map_contact.setDecimals(2)
        self.map_contact.setSingleStep(0.5)
        self.map_contact.setValue(2.0)
        self.map_contact.setToolTip(
            "Contact détecté par le signal VNA : médiane |ε*−1| ≥ seuil. "
            "Seuls les sweeps en contact sont cartographiés."
        )
        self.map_pt = QSpinBox()
        self.map_pt.setRange(2, 40)
        self.map_pt.setValue(9)
        mform = QFormLayout()
        mform.addRow("Bande résumé ε*", self._pair(self.map_flo, self.map_fhi))
        mform.addRow("Contact min |ε*−1|", self.map_contact)
        mform.addRow("Taille des points (px)", self.map_pt)

        self.btn_map_clear = QPushButton("Effacer le chemin")
        self.btn_map_clear.clicked.connect(self.clear_map)
        self.btn_map_csv = QPushButton("Exporter le chemin (.csv)…")
        self.btn_map_csv.clicked.connect(self.export_map)
        self.cam_status = QLabel("Caméra arrêtée")
        self.cam_status.setWordWrap(True)

        lay = QVBoxLayout()
        lay.addWidget(QLabel("Caméra (RGB UVC de la RealSense) :"))
        lay.addWidget(self.cam_device)
        lay.addWidget(self.btn_cam)
        lay.addWidget(self.chk_cam_auto)
        eform = QFormLayout()
        eform.addRow("Exposition", self.cam_expo)
        eform.addRow("Balance des blancs", self.cam_wb)
        lay.addLayout(eform)
        lay.addWidget(QLabel("Bague vert clair (HSV) :"))
        lay.addLayout(tform)
        lay.addWidget(QLabel("Cartographie (ε* du sweep en contact) :"))
        lay.addLayout(mform)
        lay.addWidget(self.btn_map_clear)
        lay.addWidget(self.btn_map_csv)
        lay.addWidget(self.cam_status)
        lay.addStretch(1)
        w = QWidget()
        w.setLayout(lay)
        return w

    def _build_map_page(self):
        self.scan_view = ScanView()
        self.map_pt.valueChanged.connect(self.scan_view.set_point_size)
        self.map_live = QLabel("ε' : —   ε'' : —")
        self.map_live.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.map_live.setWordWrap(True)
        self.trend_plot = pg.PlotWidget(title="ε', ε'' médians (bande résumé)")
        self.trend_plot.setLabel("bottom", "t", units="s")
        self.trend_plot.showGrid(x=True, y=True)
        self.trend_plot.addLegend()
        self.trend1 = self.trend_plot.plot(pen=pg.mkPen("g", width=2), name="ε'")
        self.trend2 = self.trend_plot.plot(pen=pg.mkPen("m", width=2), name="ε''")
        self.trend_plot.setMaximumHeight(220)  # le bandeau ε* reste bas : l'image prend le reste
        bottom = QHBoxLayout()
        bottom.addWidget(self.map_live)
        bottom.addWidget(self.trend_plot, 1)
        lay = QVBoxLayout()
        lay.addWidget(self.scan_view, 1)  # image maximisée en haut
        lay.addLayout(bottom)  # ε', ε'' (valeurs + tendance) en dessous
        w = QWidget()
        w.setLayout(lay)
        return w

    def toggle_camera(self, on):
        if not on:
            self.cam_timer.stop()
            if self.cam_stream is not None:
                self.cam_stream.stop()
                self.cam_stream = None
            self.cam_tracker = None
            self.btn_cam.setText("Démarrer la caméra")
            if "échouée" not in self.cam_status.text():
                self.cam_status.setText("Caméra arrêtée")
            return
        try:  # import paresseux : la GUI reste utilisable sans OpenCV
            from vna_app.vision.camera import CameraError, CameraStream
            from vna_app.vision.tracker import RingTracker
        except ImportError as e:
            self._cam_fail(f"OpenCV indisponible : {e}")
            return
        self.cam_tracker = RingTracker()
        self.on_tracker_params()
        stream = CameraStream(
            self.cam_tracker, self.scan_positions, device=self.cam_device.currentText().strip()
        )
        try:
            stream.open()
        except CameraError as e:
            self.cam_tracker = None
            self._cam_fail(str(e))
            return
        stream.set_controls(
            self.chk_cam_auto.isChecked(), self.cam_expo.value(), self.cam_wb.value()
        )
        stream.start()
        self.cam_stream = stream
        self.btn_cam.setText("Arrêter la caméra")
        self.cam_status.setText(f"Caméra : {stream.device} ({stream.width}×{stream.height})")
        self.cam_timer.start()

    def _cam_fail(self, msg):
        self.btn_cam.blockSignals(True)
        self.btn_cam.setChecked(False)
        self.btn_cam.blockSignals(False)
        self.btn_cam.setText("Démarrer la caméra")
        self.cam_status.setText(f"Caméra : {msg}")
        QMessageBox.warning(self, "Caméra", msg)

    def poll_camera(self):
        s = self.cam_stream
        if s is None:
            return
        if s.error:
            err = s.error
            self.btn_cam.setChecked(False)  # déclenche toggle_camera(False)
            self.cam_status.setText(f"Caméra : {err}")
            return
        latest = s.latest()
        if latest is None:
            return
        _, rgb, tip = latest
        self.scan_view.set_frame(rgb)
        self.scan_view.set_tip(tip)

    def on_tracker_params(self, *_):
        trk = self.cam_tracker
        if trk is None:
            return
        trk.h_center = float(self.trk_sp["h_center"].value())
        trk.h_width = float(self.trk_sp["h_width"].value())
        trk.s_min = self.trk_sp["s_min"].value()
        trk.v_min = self.trk_sp["v_min"].value()
        trk.min_area = float(self.trk_sp["min_area"].value())

    def on_cam_controls(self, *_):
        auto = self.chk_cam_auto.isChecked()
        self.cam_expo.setEnabled(not auto)
        self.cam_wb.setEnabled(not auto)
        if self.cam_stream is not None:
            self.cam_stream.set_controls(auto, self.cam_expo.value(), self.cam_wb.value())

    def map_on_sweep(self, freqs, params):
        """Fusion : la mesure ε* du sweep reçoit la position de la pointe à son instant."""
        r = self.result
        if r is None or r.eps is None:
            return
        e1, e2 = summarize_eps(
            freqs, r.eps, self.map_flo.value() * 1e6, self.map_fhi.value() * 1e6
        )
        if not np.isfinite(e1):
            return
        t = params.get("t_mono", time.monotonic())
        lvl = contact_level(r.eps)
        contact = bool(np.isfinite(lvl) and lvl >= self.map_contact.value())
        self.map_live.setText(
            f"ε' = {e1:.2f}   ε'' = {e2:.2f}\n"
            f"|ε*−1| = {lvl:.2f} — {'contact' if contact else 'pas de contact'}"
            f"   ({len(self.scan_map)} pts)"
        )
        self.trend.append((t, e1, e2))
        ts = np.array([p[0] for p in self.trend])
        self.trend1.setData(ts - ts[0], [p[1] for p in self.trend])
        self.trend2.setData(ts - ts[0], [p[2] for p in self.trend])
        if not contact:
            return
        pos = self.scan_positions.at(t)
        if pos is None:  # pas de position caméra contemporaine : pas de point inventé
            return
        self.scan_map.add(t, pos[0] + self.tip_dx.value(), pos[1] + self.tip_dy.value(), e1, e2)
        self.update_map_path()

    def update_map_path(self):
        _, x, y, e1, _ = self.scan_map.arrays()
        if x.size == 0:
            self.scan_view.clear_path()
            return
        if e1.size >= 10:
            lo, hi = np.percentile(e1, [5, 95])
        else:
            lo, hi = float(e1.min()), float(e1.max())
        if hi <= lo:
            hi = lo + 1e-6 + abs(lo) * 1e-6
        self.scan_view.set_path(x, y, e1, float(lo), float(hi))

    def clear_map(self):
        self.scan_map.clear()
        self.trend.clear()
        self.trend1.setData([], [])
        self.trend2.setData([], [])
        self.scan_view.clear_path()
        self.map_live.setText("ε' : —   ε'' : —")

    def export_map(self):
        if len(self.scan_map) == 0:
            self.status.setText("Chemin vide : rien à exporter")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter le chemin", "carto.csv", "CSV (*.csv)"
        )
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        self.scan_map.save_csv(path)
        self.status.setText(f"Enregistré : {path}")

    @staticmethod
    def _pair(a, b):
        w = QWidget()
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(a)
        row.addWidget(b)
        w.setLayout(row)
        return w

    def _build_cmp_tab(self):
        self.btn_mem = {}
        self.mem_labels = {}
        lay = QVBoxLayout()
        for n in (1, 2):
            b = QPushButton(f"Mémoire {n}")
            b.setToolTip("Mémorise la mesure affichée (Z, et ε* si actif)")
            b.clicked.connect(lambda _=False, n=n: self.store_memory(n))
            c = QPushButton("Effacer")
            c.clicked.connect(lambda _=False, n=n: self.clear_memory(n))
            row = QHBoxLayout()
            row.addWidget(b, 2)
            row.addWidget(c, 1)
            self.btn_mem[n] = b
            self.mem_labels[n] = QLabel(f"Mémoire {n} : vide")
            self.mem_labels[n].setWordWrap(True)
            lay.addLayout(row)
            lay.addWidget(self.mem_labels[n])
        note = QLabel("Différences = mémoire 2 − mémoire 1 (phase : arg(Z2/Z1) déroulé).")
        note.setWordWrap(True)
        lay.addWidget(note)
        lay.addStretch(1)
        w = QWidget()
        w.setLayout(lay)
        return w

    def _build_cmp_plots(self):
        grid = QGridLayout()
        rows = [
            ("|Z| — mémoires 1 et 2", "|Z|", "dBΩ", "Δ|Z| (2 − 1)", "dB"),
            ("arg Z déroulé — mémoires 1 et 2", "arg Z", "°", "Δarg Z (2 − 1)", "°"),
            ("ε', ε'' — mémoires 1 et 2", "ε', ε''", None, "Δε', Δε'' (2 − 1)", None),
        ]
        plots = []
        first = None
        for row, (t1, l1, u1, t2, u2) in enumerate(rows):
            left = pg.PlotWidget(title=t1)
            right = pg.PlotWidget(title=t2)
            left.setLabel("left", l1, units=u1)
            right.setLabel("left", "Δ", units=u2)
            for p in (left, right):
                p.setLabel("bottom", "Fréquence", units="Hz")
                p.showGrid(x=True, y=True)
                p.addLegend()
                if first is None:
                    first = p
                else:
                    p.setXLink(first)
            grid.addWidget(left, row, 0)
            grid.addWidget(right, row, 1)
            plots.append((left, right))
        w = QWidget()
        w.setLayout(grid)
        dash = pg.QtCore.Qt.DashLine
        p1 = pg.mkPen("y", width=2)
        p2 = pg.mkPen("c", width=2)
        pd = pg.mkPen("r", width=2)
        p1b = pg.mkPen("y", width=2, style=dash)
        p2b = pg.mkPen("c", width=2, style=dash)
        pdb = pg.mkPen("m", width=2, style=dash)
        (zl, zr), (al, ar), (el, er) = plots
        self.cmp_curves = {
            "mod1": zl.plot(pen=p1, name="Mémoire 1"),
            "mod2": zl.plot(pen=p2, name="Mémoire 2"),
            "dmod": zr.plot(pen=pd, name="Δ|Z|"),
            "arg1": al.plot(pen=p1, name="Mémoire 1"),
            "arg2": al.plot(pen=p2, name="Mémoire 2"),
            "darg": ar.plot(pen=pd, name="Δarg Z"),
            "e1re": el.plot(pen=p1, name="ε' mém. 1"),
            "e2re": el.plot(pen=p2, name="ε' mém. 2"),
            "e1im": el.plot(pen=p1b, name="ε'' mém. 1"),
            "e2im": el.plot(pen=p2b, name="ε'' mém. 2"),
            "dere": er.plot(pen=pd, name="Δε'"),
            "deim": er.plot(pen=pdb, name="Δε''"),
        }
        return w

    # -- identification de la zone
    def _build_cls_tab(self):
        self.ds = Dataset()
        self.model = None
        self.rec = None  # {"zone", "session", "left"} pendant un enregistrement
        self.zone_box = QComboBox()
        self.zone_box.setEditable(True)
        self.zone_box.addItems(ZONES)
        self.rec_n = QSpinBox()
        self.rec_n.setRange(1, 500)
        self.rec_n.setValue(10)
        self.btn_rec = QPushButton("Enregistrer une pose")
        self.btn_rec.setToolTip("Enregistre les N prochains sweeps acceptés (ε* actif requis)")
        self.ds_info = QLabel("")
        self.btn_ds_save = QPushButton("Enregistrer le jeu…")
        self.btn_ds_load = QPushButton("Charger un jeu…")
        self.btn_ds_clear = QPushButton("Vider")
        self.shrink = QDoubleSpinBox()
        self.shrink.setRange(0.0, 1.0)
        self.shrink.setSingleStep(0.05)
        self.shrink.setValue(0.2)
        self.shrink.setToolTip(
            "Rétrécissement de la covariance (0 = LDA pur, 1 = distances euclidiennes)"
        )
        self.btn_train = QPushButton("Entraîner et valider")
        self.chk_live = QCheckBox("Prédiction en direct")
        self.cls_label = QLabel("Zone : —")
        self.cls_label.setWordWrap(True)
        self.cls_label.setStyleSheet("font-size: 15px; font-weight: bold;")
        self.cls_report = QPlainTextEdit()
        self.cls_report.setReadOnly(True)
        self.cls_report.setStyleSheet("font-family: monospace; font-size: 11px;")
        form = QFormLayout()
        form.addRow("Zone", self.zone_box)
        form.addRow("Sweeps / pose", self.rec_n)
        form.addRow("Rétrécissement λ", self.shrink)
        lay = QVBoxLayout()
        lay.addLayout(form)
        for w in (
            self.btn_rec,
            self.ds_info,
            self.btn_ds_save,
            self.btn_ds_load,
            self.btn_ds_clear,
            self.btn_train,
            self.chk_live,
            self.cls_label,
            self.cls_report,
        ):
            lay.addWidget(w)
        self.btn_rec.clicked.connect(self.start_recording)
        self.btn_ds_save.clicked.connect(self.save_dataset)
        self.btn_ds_load.clicked.connect(self.load_dataset)
        self.btn_ds_clear.clicked.connect(self.clear_dataset)
        self.btn_train.clicked.connect(self.train_model)
        self.update_ds_info()
        w = QWidget()
        w.setLayout(lay)
        return w

    def update_ds_info(self):
        d = self.ds
        zones = sorted(set(d.zones))
        txt = f"Jeu : {len(d)} sweeps, {len(zones)} zones, {len(set(d.sessions))} poses"
        if self.rec:
            txt += f"\nEnregistrement {self.rec['session']} : {self.rec['left']} restants"
        self.ds_info.setText(txt)

    def _cls_params(self):
        return float(self.ds.meta.get("f_ref", self.f_ref_mhz.value() * 1e6))

    def start_recording(self):
        zone = self.zone_box.currentText().strip()
        if not zone:
            QMessageBox.warning(self, "Zone", "Donne un nom de zone.")
            return
        if self.pipeline.eps_mode == "off":
            QMessageBox.warning(
                self, "Zone", "Active une méthode ε* (onglet ε*) : les descripteurs en dépendent."
            )
            return
        if self.last is None:
            QMessageBox.warning(self, "Zone", "Aucun sweep reçu : lance l'acquisition.")
            return
        sig = grid_signature(self.last[0])
        if len(self.ds) == 0:
            self.ds.meta = {"f_ref": self.f_ref_mhz.value() * 1e6, "n_bands": 8, "grid": sig}
        elif not np.allclose(self.ds.meta.get("grid", sig), sig, atol=1.0):
            QMessageBox.warning(self, "Zone", "Grille de fréquences différente de celle du jeu.")
            return
        self.rec = {"zone": zone, "session": self.ds.new_session(zone), "left": self.rec_n.value()}
        self.status.setText(f"Enregistrement {self.rec['session']}…")
        self.update_ds_info()

    def collect_sweep(self, freqs):
        if self.rec is None or self.result is None:
            return
        x = features_for(freqs, self.result.eps, self._cls_params(), self.ds.meta.get("n_bands", 8))
        if x is None:
            self.status.setText("Descripteurs indisponibles (ε* actif ? contact ?)")
            return
        self.ds.add(x, self.rec["zone"], self.rec["session"])
        self.rec["left"] -= 1
        if self.rec["left"] <= 0:
            self.status.setText(f"Pose {self.rec['session']} enregistrée")
            self.rec = None
        self.update_ds_info()

    def train_model(self):
        if len(self.ds) == 0:
            QMessageBox.warning(self, "Zone", "Jeu vide.")
            return
        X, z, s = self.ds.arrays()
        lam = self.shrink.value()
        try:
            res = cross_validate(X, z, s, shrinkage=lam)
            model = ZoneClassifier(lam).fit(X, z)
        except ValueError as e:
            self.cls_report.setPlainText(f"Entraînement impossible : {e}")
            return
        model.d2_max = res["d2_max"]
        self.model = model
        self.cls_report.setPlainText(format_report(res, lam))
        self.status.setText("Modèle entraîné")

    def classify_sweep(self, freqs):
        if not self.chk_live.isChecked() or self.model is None or self.result is None:
            return
        meta = self.ds.meta
        if not np.allclose(
            meta.get("grid", grid_signature(freqs)), grid_signature(freqs), atol=1.0
        ):
            self.cls_label.setText("Zone : grille différente de l'apprentissage")
            return
        x = features_for(freqs, self.result.eps, self._cls_params(), meta.get("n_bands", 8))
        if x is None:
            self.cls_label.setText("Zone : — (pas de contact / ε* indisponible)")
            return
        best, proba, dmin = self.model.predict_one(x)
        top = sorted(proba.items(), key=lambda kv: -kv[1])[:3]
        detail = " | ".join(f"{k} {100 * v:.0f} %" for k, v in top)
        head = best if best is not None else "Inconnue"
        self.cls_label.setText(f"Zone : {head}\n{detail}\n(d² = {dmin:.1f})")

    def save_dataset(self):
        if len(self.ds) == 0:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Enregistrer le jeu", "zones.npz", "NumPy (*.npz)"
        )
        if path:
            self.ds.save(path if path.endswith(".npz") else path + ".npz")
            self.status.setText(f"Jeu enregistré : {path}")

    def load_dataset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Charger un jeu", "", "NumPy (*.npz)")
        if not path:
            return
        try:
            self.ds = Dataset.load(path)
        except (OSError, ValueError, KeyError) as e:
            QMessageBox.warning(self, "Zone", f"Jeu illisible : {e}")
            return
        self.model = None
        self.update_ds_info()

    def clear_dataset(self):
        self.ds = Dataset()
        self.model = None
        self.rec = None
        self.update_ds_info()

    # -- comparaison
    def on_tab_changed(self, idx):
        if idx in (self.eps_tab_index, self.cls_tab_index):  # page ε* : son propre Smith
            self.stack.setCurrentIndex(2)
            self.smith.setVisible(False)
        elif idx == self.map_tab_index:  # toute la largeur pour l'image
            self.stack.setCurrentIndex(3)
            self.smith.setVisible(False)
        else:
            self.stack.setCurrentIndex(1 if idx == self.cmp_tab_index else 0)
            self.smith.setVisible(True)

    # -- waterfalls (ε' + isocontours ε'', ou tan δ)
    def on_wf_scale(self, *_):
        auto = self.wf_auto.isChecked()
        for key, wf in (("re", self.wf_iso), ("tan", self.wf_tan)):  # re : isocontours = ε''
            lo, hi = self.wf_lim[key, "min"], self.wf_lim[key, "max"]
            lo.setEnabled(not auto)
            hi.setEnabled(not auto)
            wf.set_scale(lo.value(), hi.value(), auto)  # un seul rendu
        self.sync_wf_spins()

    def sync_wf_spins(self):
        """En auto : affiche les bornes réellement appliquées (sans nouveau rendu)."""
        if not self.wf_auto.isChecked():
            return
        for key, wf in (("re", self.wf_iso), ("tan", self.wf_tan)):
            lo, hi = self.wf_lim[key, "min"], self.wf_lim[key, "max"]
            for sp, v in ((lo, wf.levels[0]), (hi, wf.levels[1])):
                sp.blockSignals(True)
                sp.setValue(v)
                sp.blockSignals(False)

    def on_wf_select(self, idx):
        """Un seul waterfall visible (les deux reçoivent les données : l'historique reste)."""
        self.wf_iso.setVisible(idx == 0)
        self.wf_tan.setVisible(idx == 1)

    def on_wf_depth(self, n):
        for wf in self.waterfalls:
            wf.set_depth(n)

    def clear_waterfalls(self):
        for wf in self.waterfalls:
            wf.clear_data()

    def push_waterfalls(self, freqs):
        r = self.result
        if r is None or r.eps is None:
            return
        e1, e2, tan = to_eps_columns(r.eps)
        self.wf_iso.push(freqs, e1, aux=e2)
        self.wf_tan.push(freqs, tan)
        self.sync_wf_spins()

    def store_memory(self, n):
        if self.result is None or self.last is None:
            self.status.setText("Rien à mémoriser : faire d'abord un sweep")
            return
        r, freqs = self.result, self.last[0]
        tag = "calibré" if r.calibrated else "brut"
        eps = "ε* oui" if r.eps is not None else "ε* non"
        label = f"{len(freqs)} pts, {tag}, {eps}, {time.strftime('%H:%M:%S')}"
        self.mem[n] = snapshot(freqs, r.s[:, 0], r.eps, label)
        self.mem_labels[n].setText(f"Mémoire {n} : {label}")
        self.update_comparison()

    def clear_memory(self, n):
        self.mem[n] = None
        self.mem_labels[n].setText(f"Mémoire {n} : vide")
        self.update_comparison()

    def update_comparison(self):
        a, b = self.mem[1], self.mem[2]
        c = self.cmp_curves
        for cur in c.values():
            cur.setData([], [])
        for snap, k in ((a, "1"), (b, "2")):
            if snap is None:
                continue
            c["mod" + k].setData(snap.freqs, mod_db(snap.z))
            c["arg" + k].setData(snap.freqs, unwrap_deg(snap.z))
            if snap.eps is not None:
                c[f"e{k}re"].setData(snap.freqs, snap.eps.real)
                c[f"e{k}im"].setData(snap.freqs, -snap.eps.imag)
        if a is not None and b is not None:
            d = compare(a, b)
            f = d["freqs"]
            # mémoire 2 réaffichée sur la grille de la mémoire 1
            c["mod2"].setData(f, d["mod2_db"])
            c["arg2"].setData(f, d["arg2_deg"])
            c["dmod"].setData(f, d["d_mod_db"])
            c["darg"].setData(f, d["d_arg_deg"])
            if d["eps1_re"] is not None:
                c["e2re"].setData(f, d["eps2_re"])
                c["e2im"].setData(f, d["eps2_im"])
                c["dere"].setData(f, d["d_eps_re"])
                c["deim"].setData(f, d["d_eps_im"])

    # -- état
    def refresh_buttons(self):
        self.btn_conn.setText("Déconnecter" if self.connected else "Connecter")
        for b in [
            self.btn_single,
            self.btn_cont,
            *self.cal_buttons.values(),
            *self.eps_buttons.values(),
        ]:
            b.setEnabled(self.connected)
        self.btn_save.setEnabled(self.last is not None)
        self.btn_save_eps.setEnabled(self.result is not None and self.result.eps is not None)
        self.chk_cal.setEnabled(self.pipeline.cal is not None)

    def toggle_connect(self):
        self.btn_conn.setEnabled(False)
        self.status.setText("Connexion…" if not self.connected else "Déconnexion…")
        (self._disconnect if self.connected else self._connect).emit()

    def on_connected(self, info):
        self.connected, self.info = True, info
        self.status.setText(f"Connecté — fw {info['firmware']}")
        self.btn_conn.setEnabled(True)
        self.refresh_buttons()

    def on_disconnected(self):
        self.connected = False
        self.pending = None
        self.btn_cont.blockSignals(True)
        self.btn_cont.setChecked(False)
        self.btn_cont.blockSignals(False)
        self.status.setText("Déconnecté")
        self.btn_conn.setEnabled(True)
        self.refresh_buttons()

    def on_error(self, msg):
        self.pending = None
        self.status.setText(f"Erreur : {msg}")
        self.btn_conn.setEnabled(True)
        self.refresh_buttons()
        QMessageBox.warning(self, "vna_app", msg)

    # -- acquisition
    def _power_combo(self, default, tip):
        cb = QComboBox()
        for lvl, txt in enumerate(("0 (min)", "1", "2", "3 (max)")):
            cb.addItem(txt, lvl)
        cb.setCurrentIndex(default)
        cb.setToolTip(
            tip + ". Niveaux 0 à 3 du registre de l'appareil (≈ 2,5 dB de bruit de fond S21 "
            "par pas mesurés). Un niveau plus bas réduit le SNR ; la calibration reste "
            "valable seulement pour la puissance à laquelle elle a été faite."
        )
        cb.currentIndexChanged.connect(self.on_power)
        return cb

    def on_power(self, *_):
        if self.connected:
            self._power.emit(self.pwr_high.currentData(), self.pwr_low.currentData())

    def align_to_cal(self):
        """Début/fin/points = grille de la calibration appliquée (sauf capture de standard)."""
        p = self.pipeline
        if p.cal is None or not p.apply_cal:
            self.cal_lock.setText("")
            for w in (self.start_mhz, self.stop_mhz, self.points):
                w.setEnabled(True)
            return
        f = p.cal.freqs
        for w, v in (
            (self.start_mhz, f[0] / 1e6),
            (self.stop_mhz, f[-1] / 1e6),
            (self.points, len(f)),
        ):
            w.blockSignals(True)
            w.setValue(v)
            w.blockSignals(False)
            w.setEnabled(False)
        self.cal_lock.setText(
            f"Grille imposée par la calibration : {f[0] / 1e6:.3f}–{f[-1] / 1e6:.3f} MHz, "
            f"{len(f)} pts (décocher « Appliquer la calibration » pour la libérer)"
        )

    def on_norm(self, *_):
        self.plot_eps.setLabel("left", "Re u, −Im u" if self.chk_norm.isChecked() else "ε', ε''")
        self.redisplay()

    def on_contact_min(self, *_):
        self.n_contact_rej = 0
        self.update_contact_label()

    def update_contact_label(self, level=None):
        txt = "Contact : —" if level is None else f"Contact |ε*−1| = {level:.2f}"
        if self.contact_min.value() > 0:
            txt += f" (seuil {self.contact_min.value():.2f}, rejetés {self.n_contact_rej})"
        self.contact_lbl.setText(txt)

    def on_filter(self, *_):
        f = self.filt
        f.enabled = self.chk_filt.isChecked()
        f.d_min = self.filt_sp["d_min"].value()
        f.d_max = self.filt_sp["d_max"].value()
        f.radius = self.filt_sp["radius"].value()
        f.reset_counts()
        self.update_filter_info()

    def update_filter_info(self, why=None):
        f = self.filt
        txt = f"Acceptés {f.accepted} / rejetés {f.rejected}" if f.enabled else ""
        if why:
            txt += f"\nDernier rejet : {why}"
        self.filt_info.setText(txt)

    def start(self, continuous):
        if self.pending is None:
            self.align_to_cal()
        self._power.emit(self.pwr_high.currentData(), self.pwr_low.currentData())
        self._configure.emit(
            self.start_mhz.value() * 1e6,
            self.stop_mhz.value() * 1e6,
            self.points.value(),
            self.avg.value(),
        )
        self._run.emit(continuous)
        self.status.setText("Acquisition…")

    def toggle_continuous(self, on):
        if on:
            self.start(True)
        else:
            self.worker.stop()
            self.status.setText("Arrêté")

    def capture(self, name):
        """Capture brute de `name` au prochain sweep (sweep unique)."""
        self.btn_cont.blockSignals(True)
        self.btn_cont.setChecked(False)
        self.btn_cont.blockSignals(False)
        self.worker.stop()
        self.pending = name
        self.start(False)
        self.status.setText(f"Capture {name}…")

    def on_sweep(self, freqs, s_raw, params):
        try:
            self._on_sweep(freqs, s_raw, params)
        finally:
            self.worker.ack()  # débloque le worker (contre-pression), même sur sweep rejeté

    def _on_sweep(self, freqs, s_raw, params):
        is_capture = self.pending is not None
        if not is_capture and (self.filt.enabled or self.contact_min.value() > 0):
            r = self.pipeline.process(freqs, s_raw)
            if self.filt.enabled:
                ok, why = self.filt.check(r.s[:, 0])
                self.update_filter_info(why)
                if not ok:
                    self.status.setText(f"Sweep rejeté : {why}")
                    return
            if passes_contact(r.eps, self.contact_min.value()) is False:
                self.n_contact_rej += 1
                self.update_contact_label(contact_level(r.eps))
                self.status.setText("Sweep rejeté : contact insuffisant")
                return
        self.last = (freqs, s_raw, params)
        if self.pending is not None:
            name, self.pending = self.pending, None
            self.pipeline.capture(name, freqs, s_raw)  # toujours le BRUT
            self.update_captured_labels()
            self.status.setText(f"{name} capturé ({len(freqs)} pts)")
        self.redisplay()
        if not is_capture:  # les sweeps de standards ne vont pas dans les waterfalls
            self.push_waterfalls(freqs)
            self.collect_sweep(freqs)
            self.classify_sweep(freqs)
            self.map_on_sweep(freqs, params)

    def redisplay(self):
        if self.last is None:
            return
        freqs, s_raw, _ = self.last
        self.result = self.pipeline.process(freqs, s_raw)
        r = self.result
        tag = "calibré" if r.calibrated else "brut"
        z = gamma_to_z(r.s[:, 0])
        self.plot11.setTitle(f"|Z| ({tag})")
        self.plot21.setTitle(f"Phase Z déroulée ({tag})")
        self.c11.setData(freqs, _db(z))
        self.c21.setData(freqs, _phase_unwrapped_deg(z))
        self.smith.setTitle(f"Smith — S11 ({tag})")
        self.smith.set_gamma(r.s[:, 0])
        self.smith_eps.setTitle(f"Smith — S11 ({tag})")
        self.smith_eps.set_gamma(r.s[:, 0])
        show_eps = r.eps is not None
        self.plot_eps.setVisible(self.pipeline.eps_mode != "off")
        if show_eps:
            e1, e2, _ = to_eps_columns(r.eps)
            self.update_contact_label(contact_level(r.eps))
            if self.chk_norm.isChecked():
                try:
                    u = normalize_shape(freqs, r.eps, self.f_ref_mhz.value() * 1e6)
                    e1, e2 = u.real, -u.imag
                except ValueError as e:
                    r.notes.append(f"normalisation impossible : {e}")
            self.ceps1.setData(freqs, e1)
            self.ceps2.setData(freqs, e2)
        else:
            self.ceps1.setData([], [])
            self.ceps2.setData([], [])
        if r.notes:
            self.status.setText(" / ".join(r.notes))
        self.refresh_buttons()

    # -- calibration
    def update_captured_labels(self):
        got = self.pipeline.captured()
        cal = [CAL_LABELS[k] for k in CAL_LABELS if k in got]
        eps = [EPS_LABELS[k] for k in EPS_LABELS if k in got]
        self.cal_captured.setText("Capturés : " + (", ".join(cal) or "—"))
        self.eps_captured.setText("Capturées : " + (", ".join(eps) or "—"))

    def build_cal(self):
        try:
            cal = self.pipeline.build_cal()
        except CalibrationError as e:
            self.status.setText(f"Calibration : {e}")
            QMessageBox.warning(self, "Calibration", str(e))
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Enregistrer la calibration", "bench.cal", "Calibration (*.cal)"
        )
        if path:
            self.pipeline.save_cal(path)
        self.cal_info.setText(
            f"{cal.meta['type']} — {len(cal.freqs)} pts, "
            f"{cal.freqs[0] / 1e6:.0f}–{cal.freqs[-1] / 1e6:.0f} MHz"
            + ("" if path else " (non enregistrée)")
        )
        self._set_chk(True)
        self.align_to_cal()
        if path:
            self.remember_cal(path)
        self.redisplay()

    def remember_cal(self, path):
        try:
            LAST_CAL.parent.mkdir(parents=True, exist_ok=True)
            LAST_CAL.write_text(str(path))
        except OSError:
            pass

    def load_last_cal(self):
        """Charge la dernière calibration enregistrée/chargée (défaut au démarrage)."""
        try:
            path = LAST_CAL.read_text().strip()
            if not path or not Path(path).is_file():
                return False
            cal = self.pipeline.load_cal(path)
        except (OSError, CalibrationError):
            return False
        self._sync_delay()
        self.cal_info.setText(
            f"{path} — {len(cal.freqs)} pts, {cal.freqs[0] / 1e6:.0f}–{cal.freqs[-1] / 1e6:.0f} MHz"
        )
        self._set_chk(True)
        self.align_to_cal()
        return True

    def load_cal(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Charger une calibration", "", "Calibration (*.cal)"
        )
        if not path:
            return
        try:
            cal = self.pipeline.load_cal(path)
        except CalibrationError as e:
            self.status.setText(f"Calibration : {e}")
            QMessageBox.warning(self, "Calibration", str(e))
            return
        self._sync_delay()
        self.cal_info.setText(
            f"{path} — {len(cal.freqs)} pts, {cal.freqs[0] / 1e6:.0f}–{cal.freqs[-1] / 1e6:.0f} MHz"
        )
        self._set_chk(True)
        self.align_to_cal()
        self.remember_cal(path)
        self.redisplay()

    def _set_chk(self, on):
        self.chk_cal.blockSignals(True)
        self.chk_cal.setChecked(on)
        self.chk_cal.blockSignals(False)
        self.pipeline.apply_cal = on

    def on_delay(self, ns):
        self.pipeline.delay_s = ns * 1e-9
        self.delay_len.setText(f"{ns * 1e-9 * 299792458 * 0.66:.3f} m")
        if self.pipeline.cal is not None:
            self.redisplay()

    def _sync_delay(self):
        self.delay_ns.blockSignals(True)
        self.delay_ns.setValue(self.pipeline.delay_s * 1e9)
        self.delay_ns.blockSignals(False)
        self.delay_len.setText(f"{self.pipeline.delay_s * 299792458 * 0.66:.3f} m")

    def on_apply_cal(self, on):
        self.pipeline.apply_cal = on
        self.align_to_cal()
        self.redisplay()

    # -- ε*
    def on_eps_params(self, *_):
        p = self.pipeline
        p.eps_mode = self.eps_mode.currentData()
        p.c0 = self.c0_ff.value() * 1e-15
        p.ref_liquid = self.ref_liquid.currentText()
        p.smooth = self.smooth.value()
        self.clear_waterfalls()  # les valeurs précédentes ne correspondent plus à ces réglages
        self.redisplay()

    def save_eps(self):
        path, _ = QFileDialog.getSaveFileName(self, "Enregistrer ε*", "eps.csv", "CSV (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        freqs = self.last[0]
        e1, e2, tan = to_eps_columns(self.result.eps)
        np.savetxt(
            path,
            np.column_stack([freqs, e1, e2, tan]),
            delimiter=",",
            header="f_Hz,eps_real,eps_loss,tan_delta",
            comments="",
            fmt="%.9g",
        )
        self.status.setText(f"Enregistré : {path}")

    # -- export s2p
    def save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Enregistrer", "mesure.s2p", "Touchstone (*.s2p)"
        )
        if not path:
            return
        if not path.lower().endswith(".s2p"):
            path += ".s2p"
        freqs, s_raw, p = self.last
        r = self.result
        s = r.s if r is not None else s_raw
        cal_file = None
        if r is not None and r.calibrated:
            cal_file = self.pipeline.cal_path or "<en mémoire, non enregistrée>"
        write_s2p(
            path,
            freqs,
            s,
            step_hz=p["step_hz"],
            points=p["points"],
            averaging=p["avg"],
            info=self.info,
            stop_hz_requested=p["stop"],
            calibration_file=cal_file,
        )
        self.status.setText(f"Enregistré : {path}")

    def closeEvent(self, ev):
        self.hide()  # la fenêtre disparaît tout de suite, l'arrêt du sweep en cours se fait après
        self.cam_timer.stop()
        if self.cam_stream is not None:
            self.cam_stream.stop()
            self.cam_stream = None
        self.worker.stop()
        self._disconnect.emit()
        self.thread.quit()
        if not self.thread.wait(10000):  # un sweep de 1001 pts dure ~3,3 s
            self.thread.terminate()
            self.thread.wait(1000)
        super().closeEvent(ev)
