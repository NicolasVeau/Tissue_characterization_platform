#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: LiteVNA - S11/S21 module et phase
# Author: SurTouch
# Description: Affichage en direct de |S11|, |S21| (dB) et de leur phase.
# GNU Radio version: 3.8.1.0

from distutils.version import StrictVersion

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print("Warning: failed to XInitThreads()")

from PyQt5 import Qt
from gnuradio import qtgui
import sip
from gnuradio import blocks
from gnuradio import gr
from gnuradio.filter import firdes
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
import gr_litevna
from gnuradio import qtgui

class litevna_01_visu(gr.top_block, Qt.QWidget):

    def __init__(self, averaging=1, cal_file='/home/nveau/vna_app/bench8.cal', f_start=300e6, f_stop=1.3e9, points=501, port='/dev/litevna'):
        gr.top_block.__init__(self, "LiteVNA - S11/S21 module et phase")
        Qt.QWidget.__init__(self)
        self.setWindowTitle("LiteVNA - S11/S21 module et phase")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except:
            pass
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "litevna_01_visu")

        try:
            if StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
                self.restoreGeometry(self.settings.value("geometry").toByteArray())
            else:
                self.restoreGeometry(self.settings.value("geometry"))
        except:
            pass

        ##################################################
        # Parameters
        ##################################################
        self.averaging = averaging
        self.cal_file = cal_file
        self.f_start = f_start
        self.f_stop = f_stop
        self.points = points
        self.port = port

        ##################################################
        # Variables
        ##################################################
        self.f_step = f_step = (int(f_stop) - int(f_start)) // (points - 1)

        ##################################################
        # Blocks
        ##################################################
        self.vna = gr_litevna.sweep_source(port, f_start, f_stop, points, averaging, cal_file)
        self.sink_phase = qtgui.vector_sink_f(
            points,
            f_start / 1e6,
            f_step / 1e6,
            'Fréquence (MHz)',
            'rad',
            'Phase',
            2 # Number of inputs
        )
        self.sink_phase.set_update_time(0.10)
        self.sink_phase.set_y_axis(-3.2, 3.2)
        self.sink_phase.enable_autoscale(False)
        self.sink_phase.enable_grid(True)
        self.sink_phase.set_x_axis_units("")
        self.sink_phase.set_y_axis_units("")
        self.sink_phase.set_ref_level(0)

        labels = ['S11', 'S21', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(2):
            if len(labels[i]) == 0:
                self.sink_phase.set_line_label(i, "Data {0}".format(i))
            else:
                self.sink_phase.set_line_label(i, labels[i])
            self.sink_phase.set_line_width(i, widths[i])
            self.sink_phase.set_line_color(i, colors[i])
            self.sink_phase.set_line_alpha(i, alphas[i])

        self._sink_phase_win = sip.wrapinstance(self.sink_phase.pyqwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._sink_phase_win)
        self.sink_db = qtgui.vector_sink_f(
            points,
            f_start / 1e6,
            f_step / 1e6,
            'Fréquence (MHz)',
            'dB',
            'Module',
            2 # Number of inputs
        )
        self.sink_db.set_update_time(0.10)
        self.sink_db.set_y_axis(-120, 10)
        self.sink_db.enable_autoscale(False)
        self.sink_db.enable_grid(True)
        self.sink_db.set_x_axis_units("")
        self.sink_db.set_y_axis_units("")
        self.sink_db.set_ref_level(0)

        labels = ['S11', 'S21', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(2):
            if len(labels[i]) == 0:
                self.sink_db.set_line_label(i, "Data {0}".format(i))
            else:
                self.sink_db.set_line_label(i, labels[i])
            self.sink_db.set_line_width(i, widths[i])
            self.sink_db.set_line_color(i, colors[i])
            self.sink_db.set_line_alpha(i, alphas[i])

        self._sink_db_win = sip.wrapinstance(self.sink_db.pyqwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._sink_db_win)
        self.rep_sink_phase_1 = blocks.repeat(gr.sizeof_float*points, 2)
        self.rep_sink_phase_0 = blocks.repeat(gr.sizeof_float*points, 2)
        self.rep_sink_db_1 = blocks.repeat(gr.sizeof_float*points, 2)
        self.rep_sink_db_0 = blocks.repeat(gr.sizeof_float*points, 2)
        self.mag2_s21 = blocks.complex_to_mag_squared(points)
        self.mag2_s11 = blocks.complex_to_mag_squared(points)
        self.db_s21 = blocks.nlog10_ff(10, points, 0)
        self.db_s11 = blocks.nlog10_ff(10, points, 0)
        self.arg_s21 = blocks.complex_to_arg(points)
        self.arg_s11 = blocks.complex_to_arg(points)



        ##################################################
        # Connections
        ##################################################
        self.connect((self.arg_s11, 0), (self.rep_sink_phase_0, 0))
        self.connect((self.arg_s21, 0), (self.rep_sink_phase_1, 0))
        self.connect((self.db_s11, 0), (self.rep_sink_db_0, 0))
        self.connect((self.db_s21, 0), (self.rep_sink_db_1, 0))
        self.connect((self.mag2_s11, 0), (self.db_s11, 0))
        self.connect((self.mag2_s21, 0), (self.db_s21, 0))
        self.connect((self.rep_sink_db_0, 0), (self.sink_db, 0))
        self.connect((self.rep_sink_db_1, 0), (self.sink_db, 1))
        self.connect((self.rep_sink_phase_0, 0), (self.sink_phase, 0))
        self.connect((self.rep_sink_phase_1, 0), (self.sink_phase, 1))
        self.connect((self.vna, 0), (self.arg_s11, 0))
        self.connect((self.vna, 1), (self.arg_s21, 0))
        self.connect((self.vna, 0), (self.mag2_s11, 0))
        self.connect((self.vna, 1), (self.mag2_s21, 0))

    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "litevna_01_visu")
        self.settings.setValue("geometry", self.saveGeometry())
        event.accept()

    def get_averaging(self):
        return self.averaging

    def set_averaging(self, averaging):
        self.averaging = averaging
        self.vna.set_averaging(self.averaging)

    def get_cal_file(self):
        return self.cal_file

    def set_cal_file(self, cal_file):
        self.cal_file = cal_file

    def get_f_start(self):
        return self.f_start

    def set_f_start(self, f_start):
        self.f_start = f_start
        self.set_f_step((int(self.f_stop) - int(self.f_start)) // (self.points - 1))
        self.sink_db.set_x_axis(self.f_start / 1e6, self.f_step / 1e6)
        self.sink_phase.set_x_axis(self.f_start / 1e6, self.f_step / 1e6)
        self.vna.set_f_start(self.f_start)

    def get_f_stop(self):
        return self.f_stop

    def set_f_stop(self, f_stop):
        self.f_stop = f_stop
        self.set_f_step((int(self.f_stop) - int(self.f_start)) // (self.points - 1))
        self.vna.set_f_stop(self.f_stop)

    def get_points(self):
        return self.points

    def set_points(self, points):
        self.points = points
        self.set_f_step((int(self.f_stop) - int(self.f_start)) // (self.points - 1))

    def get_port(self):
        return self.port

    def set_port(self, port):
        self.port = port

    def get_f_step(self):
        return self.f_step

    def set_f_step(self, f_step):
        self.f_step = f_step
        self.sink_db.set_x_axis(self.f_start / 1e6, self.f_step / 1e6)
        self.sink_phase.set_x_axis(self.f_start / 1e6, self.f_step / 1e6)


def argument_parser():
    description = 'Affichage en direct de |S11|, |S21| (dB) et de leur phase.'
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "--averaging", dest="averaging", type=intx, default=1,
        help="Set Averaging [default=%(default)r]")
    parser.add_argument(
        "--cal-file", dest="cal_file", type=str, default='/home/nveau/vna_app/bench8.cal',
        help="Set Fichier .cal (vide = brut) [default=%(default)r]")
    parser.add_argument(
        "--f-start", dest="f_start", type=eng_float, default="300.0M",
        help="Set Start (Hz) [default=%(default)r]")
    parser.add_argument(
        "--f-stop", dest="f_stop", type=eng_float, default="1.3G",
        help="Set Stop (Hz) [default=%(default)r]")
    parser.add_argument(
        "--points", dest="points", type=intx, default=501,
        help="Set Points [default=%(default)r]")
    parser.add_argument(
        "-p", "--port", dest="port", type=str, default='/dev/litevna',
        help="Set Port (/dev/litevna ou sim) [default=%(default)r]")
    return parser


def main(top_block_cls=litevna_01_visu, options=None):
    if options is None:
        options = argument_parser().parse_args()

    if StrictVersion("4.5.0") <= StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls(averaging=options.averaging, cal_file=options.cal_file, f_start=options.f_start, f_stop=options.f_stop, points=options.points, port=options.port)
    tb.start()
    tb.show()

    def sig_handler(sig=None, frame=None):
        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    def quitting():
        tb.stop()
        tb.wait()
    qapp.aboutToQuit.connect(quitting)
    qapp.exec_()


if __name__ == '__main__':
    main()
