#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: LiteVNA - fréquence fixe (CW)
# Author: SurTouch
# Description: Module et phase de S11 à fréquence fixe, en fonction du temps.
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
from gnuradio.filter import firdes
import sip
from gnuradio import blocks
from gnuradio import gr
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio.qtgui import Range, RangeWidget
import gr_litevna
from gnuradio import qtgui

class litevna_05_cw(gr.top_block, Qt.QWidget):

    def __init__(self, averaging=1, block_len=256, port='/dev/litevna'):
        gr.top_block.__init__(self, "LiteVNA - fréquence fixe (CW)")
        Qt.QWidget.__init__(self)
        self.setWindowTitle("LiteVNA - fréquence fixe (CW)")
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

        self.settings = Qt.QSettings("GNU Radio", "litevna_05_cw")

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
        self.block_len = block_len
        self.port = port

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 300.0 / averaging
        self.freq = freq = 1e9

        ##################################################
        # Blocks
        ##################################################
        self._freq_range = Range(400e6, 3e9, 1e6, 1e9, 200)
        self._freq_win = RangeWidget(self._freq_range, self.set_freq, 'Fréquence (Hz)', "counter_slider", float)
        self.top_grid_layout.addWidget(self._freq_win)
        self.vna = gr_litevna.cw_source(port, freq, block_len, averaging)
        self.sink_mag = qtgui.time_sink_f(
            600, #size
            samp_rate, #samp_rate
            '|S11|', #name
            1 #number of inputs
        )
        self.sink_mag.set_update_time(0.10)
        self.sink_mag.set_y_axis(0, 1.2)

        self.sink_mag.set_y_label('|S11| (linéaire)', "")

        self.sink_mag.enable_tags(False)
        self.sink_mag.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.sink_mag.enable_autoscale(True)
        self.sink_mag.enable_grid(False)
        self.sink_mag.enable_axis_labels(True)
        self.sink_mag.enable_control_panel(False)
        self.sink_mag.enable_stem_plot(False)


        labels = ['|S11|', 'Signal 2', 'Signal 3', 'Signal 4', 'Signal 5',
            'Signal 6', 'Signal 7', 'Signal 8', 'Signal 9', 'Signal 10']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'red', 'green', 'black', 'cyan',
            'magenta', 'yellow', 'dark red', 'dark green', 'dark blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(1):
            if len(labels[i]) == 0:
                self.sink_mag.set_line_label(i, "Data {0}".format(i))
            else:
                self.sink_mag.set_line_label(i, labels[i])
            self.sink_mag.set_line_width(i, widths[i])
            self.sink_mag.set_line_color(i, colors[i])
            self.sink_mag.set_line_style(i, styles[i])
            self.sink_mag.set_line_marker(i, markers[i])
            self.sink_mag.set_line_alpha(i, alphas[i])

        self._sink_mag_win = sip.wrapinstance(self.sink_mag.pyqwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._sink_mag_win)
        self.sink_arg = qtgui.time_sink_f(
            600, #size
            samp_rate, #samp_rate
            'arg S11', #name
            1 #number of inputs
        )
        self.sink_arg.set_update_time(0.10)
        self.sink_arg.set_y_axis(-3.2, 3.2)

        self.sink_arg.set_y_label('rad', "")

        self.sink_arg.enable_tags(False)
        self.sink_arg.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.sink_arg.enable_autoscale(False)
        self.sink_arg.enable_grid(False)
        self.sink_arg.enable_axis_labels(True)
        self.sink_arg.enable_control_panel(False)
        self.sink_arg.enable_stem_plot(False)


        labels = ['arg S11', 'Signal 2', 'Signal 3', 'Signal 4', 'Signal 5',
            'Signal 6', 'Signal 7', 'Signal 8', 'Signal 9', 'Signal 10']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'red', 'green', 'black', 'cyan',
            'magenta', 'yellow', 'dark red', 'dark green', 'dark blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(1):
            if len(labels[i]) == 0:
                self.sink_arg.set_line_label(i, "Data {0}".format(i))
            else:
                self.sink_arg.set_line_label(i, labels[i])
            self.sink_arg.set_line_width(i, widths[i])
            self.sink_arg.set_line_color(i, colors[i])
            self.sink_arg.set_line_style(i, styles[i])
            self.sink_arg.set_line_marker(i, markers[i])
            self.sink_arg.set_line_alpha(i, alphas[i])

        self._sink_arg_win = sip.wrapinstance(self.sink_arg.pyqwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._sink_arg_win)
        self.null_s21 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.mag_s11 = blocks.complex_to_mag(1)
        self.arg_s11 = blocks.complex_to_arg(1)



        ##################################################
        # Connections
        ##################################################
        self.connect((self.arg_s11, 0), (self.sink_arg, 0))
        self.connect((self.mag_s11, 0), (self.sink_mag, 0))
        self.connect((self.vna, 0), (self.arg_s11, 0))
        self.connect((self.vna, 0), (self.mag_s11, 0))
        self.connect((self.vna, 1), (self.null_s21, 0))

    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "litevna_05_cw")
        self.settings.setValue("geometry", self.saveGeometry())
        event.accept()

    def get_averaging(self):
        return self.averaging

    def set_averaging(self, averaging):
        self.averaging = averaging
        self.set_samp_rate(300.0 / self.averaging)
        self.vna.set_averaging(self.averaging)

    def get_block_len(self):
        return self.block_len

    def set_block_len(self, block_len):
        self.block_len = block_len

    def get_port(self):
        return self.port

    def set_port(self, port):
        self.port = port

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.sink_arg.set_samp_rate(self.samp_rate)
        self.sink_mag.set_samp_rate(self.samp_rate)

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self.vna.set_freq(self.freq)


def argument_parser():
    description = 'Module et phase de S11 à fréquence fixe, en fonction du temps.'
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "--averaging", dest="averaging", type=intx, default=1,
        help="Set Averaging [default=%(default)r]")
    parser.add_argument(
        "--block-len", dest="block_len", type=intx, default=256,
        help="Set Echantillons par acquisition [default=%(default)r]")
    parser.add_argument(
        "-p", "--port", dest="port", type=str, default='/dev/litevna',
        help="Set Port (/dev/litevna ou sim) [default=%(default)r]")
    return parser


def main(top_block_cls=litevna_05_cw, options=None):
    if options is None:
        options = argument_parser().parse_args()

    if StrictVersion("4.5.0") <= StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls(averaging=options.averaging, block_len=options.block_len, port=options.port)
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
