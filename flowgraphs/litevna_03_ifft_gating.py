#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: LiteVNA - réponse temporelle et time-gating de S11
# Author: SurTouch
# Description: IFFT de S11 sans fenêtre (réponse temporelle), porte temporelle réglable, retour en fréquence.
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
from gnuradio import fft
from gnuradio.fft import window
from gnuradio import gr
from gnuradio.filter import firdes
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio.qtgui import Range, RangeWidget
import gr_litevna
from gnuradio import qtgui

class litevna_03_ifft_gating(gr.top_block, Qt.QWidget):

    def __init__(self, averaging=1, cal_file='/home/nveau/vna_app/bench8.cal', f_start=300e6, f_stop=1.3e9, points=501, port='/dev/litevna'):
        gr.top_block.__init__(self, "LiteVNA - réponse temporelle et time-gating de S11")
        Qt.QWidget.__init__(self)
        self.setWindowTitle("LiteVNA - réponse temporelle et time-gating de S11")
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

        self.settings = Qt.QSettings("GNU Radio", "litevna_03_ifft_gating")

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
        self.t_step_ns = t_step_ns = 1e9 / (points * f_step)
        self.gate_ns = gate_ns = 5
        self.gate_bins = gate_bins = max(1, int(round(gate_ns / t_step_ns)))
        self.gate = gate = [(1.0 if (i < gate_bins or i >= points - gate_bins) else 0.0) / points for i in range(points)]

        ##################################################
        # Blocks
        ##################################################
        self.vna = gr_litevna.sweep_source(port, f_start, f_stop, points, averaging, cal_file)
        self.sink_time = qtgui.vector_sink_f(
            points,
            0,
            t_step_ns,
            'Temps (ns)',
            'dB (non normalisé)',
            'Réponse temporelle de S11',
            1 # Number of inputs
        )
        self.sink_time.set_update_time(0.10)
        self.sink_time.set_y_axis(-40, 60)
        self.sink_time.enable_autoscale(False)
        self.sink_time.enable_grid(True)
        self.sink_time.set_x_axis_units("")
        self.sink_time.set_y_axis_units("")
        self.sink_time.set_ref_level(0)

        labels = ['IFFT(S11)', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.sink_time.set_line_label(i, "Data {0}".format(i))
            else:
                self.sink_time.set_line_label(i, labels[i])
            self.sink_time.set_line_width(i, widths[i])
            self.sink_time.set_line_color(i, colors[i])
            self.sink_time.set_line_alpha(i, alphas[i])

        self._sink_time_win = sip.wrapinstance(self.sink_time.pyqwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._sink_time_win)
        self.sink_freq = qtgui.vector_sink_f(
            points,
            f_start / 1e6,
            f_step / 1e6,
            'Fréquence (MHz)',
            'dB',
            'S11 brut et après porte',
            2 # Number of inputs
        )
        self.sink_freq.set_update_time(0.10)
        self.sink_freq.set_y_axis(-80, 10)
        self.sink_freq.enable_autoscale(False)
        self.sink_freq.enable_grid(True)
        self.sink_freq.set_x_axis_units("")
        self.sink_freq.set_y_axis_units("")
        self.sink_freq.set_ref_level(0)

        labels = ['S11', 'S11 après porte', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(2):
            if len(labels[i]) == 0:
                self.sink_freq.set_line_label(i, "Data {0}".format(i))
            else:
                self.sink_freq.set_line_label(i, labels[i])
            self.sink_freq.set_line_width(i, widths[i])
            self.sink_freq.set_line_color(i, colors[i])
            self.sink_freq.set_line_alpha(i, alphas[i])

        self._sink_freq_win = sip.wrapinstance(self.sink_freq.pyqwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._sink_freq_win)
        self.rep_sink_time_0 = blocks.repeat(gr.sizeof_float*points, 2)
        self.rep_sink_freq_1 = blocks.repeat(gr.sizeof_float*points, 2)
        self.rep_sink_freq_0 = blocks.repeat(gr.sizeof_float*points, 2)
        self.null_s21 = blocks.null_sink(gr.sizeof_gr_complex*points)
        self.mag2_time = blocks.complex_to_mag_squared(points)
        self.mag2_s11 = blocks.complex_to_mag_squared(points)
        self.mag2_gated = blocks.complex_to_mag_squared(points)
        self.ifft = fft.fft_vcc(points, False, [], False, 1)
        self._gate_ns_range = Range(t_step_ns, points * t_step_ns / 2, t_step_ns, 5, 200)
        self._gate_ns_win = RangeWidget(self._gate_ns_range, self.set_gate_ns, 'Porte temporelle (ns)', "counter_slider", float)
        self.top_grid_layout.addWidget(self._gate_ns_win)
        self.gate_mult = blocks.multiply_const_vcc(gate)
        self.fft_fwd = fft.fft_vcc(points, True, [], False, 1)
        self.db_time = blocks.nlog10_ff(10, points, 0)
        self.db_s11 = blocks.nlog10_ff(10, points, 0)
        self.db_gated = blocks.nlog10_ff(10, points, 0)



        ##################################################
        # Connections
        ##################################################
        self.connect((self.db_gated, 0), (self.rep_sink_freq_1, 0))
        self.connect((self.db_s11, 0), (self.rep_sink_freq_0, 0))
        self.connect((self.db_time, 0), (self.rep_sink_time_0, 0))
        self.connect((self.fft_fwd, 0), (self.mag2_gated, 0))
        self.connect((self.gate_mult, 0), (self.fft_fwd, 0))
        self.connect((self.ifft, 0), (self.gate_mult, 0))
        self.connect((self.ifft, 0), (self.mag2_time, 0))
        self.connect((self.mag2_gated, 0), (self.db_gated, 0))
        self.connect((self.mag2_s11, 0), (self.db_s11, 0))
        self.connect((self.mag2_time, 0), (self.db_time, 0))
        self.connect((self.rep_sink_freq_0, 0), (self.sink_freq, 0))
        self.connect((self.rep_sink_freq_1, 0), (self.sink_freq, 1))
        self.connect((self.rep_sink_time_0, 0), (self.sink_time, 0))
        self.connect((self.vna, 0), (self.ifft, 0))
        self.connect((self.vna, 0), (self.mag2_s11, 0))
        self.connect((self.vna, 1), (self.null_s21, 0))

    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "litevna_03_ifft_gating")
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
        self.sink_freq.set_x_axis(self.f_start / 1e6, self.f_step / 1e6)
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
        self.set_gate([(1.0 if (i < self.gate_bins or i >= self.points - self.gate_bins) else 0.0) / self.points for i in range(self.points)])
        self.set_t_step_ns(1e9 / (self.points * self.f_step))

    def get_port(self):
        return self.port

    def set_port(self, port):
        self.port = port

    def get_f_step(self):
        return self.f_step

    def set_f_step(self, f_step):
        self.f_step = f_step
        self.set_t_step_ns(1e9 / (self.points * self.f_step))
        self.sink_freq.set_x_axis(self.f_start / 1e6, self.f_step / 1e6)

    def get_t_step_ns(self):
        return self.t_step_ns

    def set_t_step_ns(self, t_step_ns):
        self.t_step_ns = t_step_ns
        self.set_gate_bins(max(1, int(round(self.gate_ns / self.t_step_ns))))
        self.sink_time.set_x_axis(0, self.t_step_ns)

    def get_gate_ns(self):
        return self.gate_ns

    def set_gate_ns(self, gate_ns):
        self.gate_ns = gate_ns
        self.set_gate_bins(max(1, int(round(self.gate_ns / self.t_step_ns))))

    def get_gate_bins(self):
        return self.gate_bins

    def set_gate_bins(self, gate_bins):
        self.gate_bins = gate_bins
        self.set_gate([(1.0 if (i < self.gate_bins or i >= self.points - self.gate_bins) else 0.0) / self.points for i in range(self.points)])

    def get_gate(self):
        return self.gate

    def set_gate(self, gate):
        self.gate = gate
        self.gate_mult.set_k(self.gate)


def argument_parser():
    description = 'IFFT de S11 sans fenêtre (réponse temporelle), porte temporelle réglable, retour en fréquence.'
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


def main(top_block_cls=litevna_03_ifft_gating, options=None):
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
