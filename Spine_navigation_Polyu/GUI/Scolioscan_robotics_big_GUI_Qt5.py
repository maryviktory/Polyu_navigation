# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Scolioscan_robotics_GUI.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

# Fix qt import error
# Include this file before import PyQt5


from PyQt5 import QtCore, QtGui, QtWidgets

try:
    _encoding = QtWidgets.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtCore.QCoreApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtCore.QCoreApplication.translate(context, text, disambig)

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(622, 565)
        Dialog.setAutoFillBackground(False)
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(190, 40, 400, 360))
        self.label_3.setFrameShape(QtWidgets.QFrame.Box)
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.label_text = QtWidgets.QLabel(Dialog)
        self.label_text.setGeometry(QtCore.QRect(370, 10, 101, 20))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setItalic(True)
        self.label_text.setFont(font)
        self.label_text.setObjectName("label_text")
        self.Button_move = QtWidgets.QPushButton(Dialog)
        self.Button_move.setGeometry(QtCore.QRect(190, 410, 401, 71))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.Button_move.setFont(font)
        self.Button_move.setMouseTracking(True)
        self.Button_move.setStyleSheet("QPushButton{ rgb(qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(0, 0, 0, 255), stop:1 rgba(255, 255, 255, 255))255, 170, 127); }\n"
"gridline-color: rgb(170, 170, 255);")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("GUI/play-button_318-42541.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Button_move.setIcon(icon)
        self.Button_move.setObjectName("Button_move")
        self.Button_stop_move = QtWidgets.QPushButton(Dialog)
        self.Button_stop_move.setGeometry(QtCore.QRect(320, 500, 111, 51))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.Button_stop_move.setFont(font)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("GUI/stop-playing-button_318-40256.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Button_stop_move.setIcon(icon1)
        self.Button_stop_move.setObjectName("Button_stop_move")
        self.toolButton = QtWidgets.QToolButton(Dialog)
        self.toolButton.setGeometry(QtCore.QRect(150, 330, 25, 19))
        self.toolButton.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.toolButton.setObjectName("toolButton")
        self.label_trajectory_name = QtWidgets.QLabel(Dialog)
        self.label_trajectory_name.setGeometry(QtCore.QRect(20, 330, 101, 21))
        self.label_trajectory_name.setText("")
        self.label_trajectory_name.setObjectName("label_trajectory_name")
        self.Button_freedrive = QtWidgets.QPushButton(Dialog)
        self.Button_freedrive.setGeometry(QtCore.QRect(20, 40, 161, 91))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.Button_freedrive.setFont(font)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("GUI/ur3-table-top-robot-small.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Button_freedrive.setIcon(icon2)
        self.Button_freedrive.setIconSize(QtCore.QSize(25, 25))
        self.Button_freedrive.setObjectName("Button_freedrive")
        self.Button_start = QtWidgets.QPushButton(Dialog)
        self.Button_start.setGeometry(QtCore.QRect(20, 160, 71, 61))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.Button_start.setFont(font)
        self.Button_start.setStyleSheet("white")
        self.Button_start.setText("")
        self.Button_start.setIcon(icon)
        self.Button_start.setIconSize(QtCore.QSize(40, 40))
        self.Button_start.setObjectName("Button_start")
        self.Button_stop = QtWidgets.QPushButton(Dialog)
        self.Button_stop.setGeometry(QtCore.QRect(110, 160, 71, 61))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.Button_stop.setFont(font)
        self.Button_stop.setText("")
        self.Button_stop.setIcon(icon1)
        self.Button_stop.setIconSize(QtCore.QSize(40, 40))
        self.Button_stop.setObjectName("Button_stop")
        self.label_UR3 = QtWidgets.QLabel(Dialog)
        self.label_UR3.setGeometry(QtCore.QRect(20, 410, 141, 141))
        self.label_UR3.setText("")
        self.label_UR3.setPixmap(QtGui.QPixmap("GUI/ur3-table-top-robot-small.png"))
        self.label_UR3.setScaledContents(True)
        self.label_UR3.setObjectName("label_UR3")
        self.label_Polyu = QtWidgets.QLabel(Dialog)
        self.label_Polyu.setGeometry(QtCore.QRect(510, 540, 81, 16))
        self.label_Polyu.setObjectName("label_Polyu")
        self.Radio_button_plot = QtWidgets.QRadioButton(Dialog)
        self.Radio_button_plot.setGeometry(QtCore.QRect(50, 360, 82, 17))
        self.Radio_button_plot.setObjectName("Radio_button_plot")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(_translate("Dialog", "Scolioscan Robotics", None))
        self.label_text.setText(_translate("Dialog", "US image", None))
        self.Button_move.setText(_translate("Dialog", "Move", None))
        self.Button_stop_move.setText(_translate("Dialog", "stop", None))
        self.toolButton.setText(_translate("Dialog", "...", None))
        self.Button_freedrive.setText(_translate("Dialog", "Freedrive", None))
        self.label_Polyu.setText(_translate("Dialog", "PolyU 07.2018", None))
        self.Radio_button_plot.setText(_translate("Dialog", "Add plots", None))


if __name__ == "__main__":
    import sys


    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

