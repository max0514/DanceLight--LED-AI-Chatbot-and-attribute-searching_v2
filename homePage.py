# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
import sys, os

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        
        # 設定視窗圖示 (如果有 icon 檔的話)
        # MainWindow.setWindowIcon(QtGui.QIcon("logo.ico"))

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setContentsMargins(30, 30, 30, 30)
        self.gridLayout.setSpacing(20)
        self.gridLayout.setObjectName("gridLayout")

        # 中間 Logo
        self.logo_label = QtWidgets.QLabel(self.centralwidget)
        self.logo_label.setMinimumSize(QtCore.QSize(400, 200))
        self.logo_label.setMaximumSize(QtCore.QSize(500, 300))
        self.logo_label.setAlignment(QtCore.Qt.AlignCenter)
        self.logo_label.setObjectName("logo_label")
        self.gridLayout.addWidget(self.logo_label, 0, 1, 1, 1)

        # 左按鈕 (AI 客服)
        self.pushButton_left = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_left.setMinimumSize(QtCore.QSize(120, 200))
        self.pushButton_left.setMaximumSize(QtCore.QSize(150, 300))
        self.pushButton_left.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_left.setObjectName("pushButton_left")
        self.gridLayout.addWidget(self.pushButton_left, 0, 0, 1, 1)

        # 右按鈕 (型號查詢)
        self.pushButton_right = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_right.setMinimumSize(QtCore.QSize(120, 200))
        self.pushButton_right.setMaximumSize(QtCore.QSize(150, 300))
        self.pushButton_right.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_right.setObjectName("pushButton_right")
        self.gridLayout.addWidget(self.pushButton_right, 0, 2, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        
        # 狀態列
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "舞光 DanceLight LED - AI 智慧客服系統"))
        
        # 載入圖片 (請確保圖片與 py 檔在同一個資料夾)
        pixmap = QPixmap("dancelight_logo.jpg")
        if not pixmap.isNull():
            self.logo_label.setPixmap(pixmap)
            self.logo_label.setScaledContents(True)
        else:
            self.logo_label.setText("DanceLight Logo")
            print("⚠️ 找不到 dancelight_logo.jpg")

        self.pushButton_left.setText(_translate("MainWindow", "AI\n客服"))
        self.pushButton_right.setText(_translate("MainWindow", "型號\n查詢"))

# 主程式邏輯
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # 載入並套用您提供的 QSS 樣式
        self.apply_style()

        # 事件連接
        self.ui.pushButton_left.clicked.connect(self.open_ai_chat)
        self.ui.pushButton_right.clicked.connect(self.open_model_search)

    def apply_style(self):
        """套用首頁專用 QSS"""
        home_qss = """
        /* 整體背景 */
        QWidget {
            background-color: white;
            font-family: ".AppleSystemUIFont", "Arial", sans-serif;
        }

        /* 共用按鈕樣式 */
        QPushButton {
            background-color: white;      /* 平常背景白色 */
            color: #185ca1;               /* 平常文字顏色 */
            border: 2px solid white;      /* 平常邊框白色 */
            border-radius: 12px;
            padding: 10px;
            font-size: 28px;
            font-weight: bold;
        }

        /* 滑鼠滑過效果 */
        QPushButton:hover {
            border: 2px solid #ccc;       /* 邊框改成灰色 */
            color: #f17039;               /* 文字變橘色 */
            background-color: qlineargradient(
                spread:pad, x1:0, y1:0, x2:0, y2:1,
                stop:0 #cccccc,           /* 上方灰色 */
                stop:1 #ffffff            /* 下方白色 */
            );
        }

        /* 按下時 */
        QPushButton:pressed {
            background-color: #d9d9d9;
            color: #0f2a4d;
        }
        """
        self.setStyleSheet(home_qss)

    def open_ai_chat(self):
        """開啟 AI 客服頁面 (ai_chat_page.py)"""
        try:
            from ai_chat_page import AIChatPage
            self.ai_chat_window = AIChatPage()
            self.ai_chat_window.show()
            self.close()  # 關閉首頁
        except ImportError:
            QtWidgets.QMessageBox.critical(self, "錯誤", "找不到 ai_chat_page.py 檔案！")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "錯誤", f"開啟失敗: {str(e)}")
            print(e)

    def open_model_search(self):
        """開啟型號查詢頁面 (model_search_page.py)"""
        try:
            from model_search_page import ModelSearchPage
            self.model_window = ModelSearchPage()
            self.model_window.show()
            self.close() # 關閉首頁
        except ImportError:
            QtWidgets.QMessageBox.critical(self, "錯誤", "找不到 model_search_page.py 檔案！")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    
    # 設定整個 App 的字體，確保簡體繁體不亂碼
    font = QtGui.QFont("Arial", 10)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())