# -*- coding: utf-8 -*-
import os
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

# ========================================================
# 1. 環境與路徑設定
# ========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))

# 解決一些電腦上可能會出現的 WinError 1114
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(current_dir)

# 設定後端資料夾路徑
backend_folder = os.path.join(current_dir, "AttributeSearch")

if os.path.exists(backend_folder):
    if backend_folder not in sys.path:
        sys.path.insert(0, backend_folder)

# ========================================================
# 2. 嘗試導入後端 (finalBackend.py)
# ========================================================
BACKEND_AVAILABLE = False
try:
    import finalBackend
    BACKEND_AVAILABLE = True
    print("成功導入 finalBackend")
except ImportError as e:
    BACKEND_AVAILABLE = False
    print("------------------------------------------------")
    print(f"導入失敗 錯誤: {e}")
    print("------------------------------------------------")

# ========================================================
# 3. 延遲導入導航 (避免循環引用)
# ========================================================
def get_home_page():
    try:
        from homePage import MainWindow
        return MainWindow
    except ImportError:
        return None

def get_ai_chat_page():
    try:
        from ai_chat_page import AIChatPage
        return AIChatPage
    except ImportError:
        return None

# ========================================================
# 4. 雙頭滑軌 (Range Slider)
# ========================================================
class RangeSlider(QtWidgets.QWidget):
    valueChanged = QtCore.pyqtSignal(int, int)

    def __init__(self, min_val=0, max_val=100, parent=None):
        super().__init__(parent)
        self.min_val = min_val
        self.max_val = max_val
        self.low = min_val
        self.high = max_val
        self.pressed_control = None 
        self.setFixedHeight(30)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        track_height = 4
        margin = 10
        w = self.width()
        h = self.height()
        track_y = h // 2 - track_height // 2
        
        # 灰色底軌
        painter.setPen(Qt.NoPen)
        painter.setBrush(QtGui.QColor("#e0e0e0"))
        painter.drawRoundedRect(margin, track_y, w - 2*margin, track_height, 2, 2)

        # 計算位置
        span = self.max_val - self.min_val
        if span == 0: span = 1
        ratio_low = (self.low - self.min_val) / span
        ratio_high = (self.high - self.min_val) / span
        
        x_low = margin + ratio_low * (w - 2*margin)
        x_high = margin + ratio_high * (w - 2*margin)

        # 藍色選取範圍
        painter.setBrush(QtGui.QColor("#185ca1"))
        rect_width = x_high - x_low
        painter.drawRoundedRect(int(x_low), track_y, int(rect_width), track_height, 2, 2)

        # 把手
        painter.setBrush(Qt.white)
        painter.setPen(QtGui.QPen(QtGui.QColor("#185ca1"), 2))
        painter.drawEllipse(QtCore.QPoint(int(x_low), h // 2), 8, 8)
        painter.drawEllipse(QtCore.QPoint(int(x_high), h // 2), 8, 8)

    def mousePressEvent(self, event):
        self.update_from_mouse(event.pos().x())

    def mouseMoveEvent(self, event):
        if self.pressed_control:
            self.update_from_mouse(event.pos().x())

    def mouseReleaseEvent(self, event):
        self.pressed_control = None

    def update_from_mouse(self, x_pos):
        w = self.width()
        margin = 10
        span = self.max_val - self.min_val
        if span == 0: span = 1
        
        val = self.min_val + ((x_pos - margin) / (w - 2*margin)) * span
        
        # 決定控制哪一個把手
        if self.pressed_control is None:
            dist_low = abs(val - self.low)
            dist_high = abs(val - self.high)
            if dist_low < dist_high:
                self.pressed_control = 'low'
            else:
                self.pressed_control = 'high'

        self.update_val(val, self.pressed_control)

    def update_val(self, val, control):
        val = max(self.min_val, min(self.max_val, val))
        if control == 'low':
            self.low = min(val, self.high)
        elif control == 'high':
            self.high = max(val, self.low)
        self.update()
        self.valueChanged.emit(int(self.low), int(self.high))

# ========================================================
# 5. UI
# ========================================================
class Ui_ModelSearchWindow(object):
    def setupUi(self, ModelSearchWindow):
        ModelSearchWindow.setObjectName("ModelSearchWindow")
        ModelSearchWindow.resize(650, 850)
        self.centralwidget = QtWidgets.QWidget(ModelSearchWindow)
        
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setContentsMargins(15, 15, 15, 15)
        self.verticalLayout.setSpacing(10)

        # --- Top Bar ---
        self.top_frame = QtWidgets.QFrame(self.centralwidget)
        self.top_layout = QtWidgets.QHBoxLayout(self.top_frame)
        self.top_layout.setContentsMargins(0, 0, 0, 0)

        self.back_home_button = QtWidgets.QPushButton("回到首頁")
        self.back_home_button.setObjectName("back_home_button")
        self.back_home_button.setFixedSize(120, 50)
        
        self.logo_label = QtWidgets.QLabel()
        logo_path = os.path.join(current_dir, "dancelight_logo.jpg")
        if os.path.exists(logo_path):
            self.logo_label.setPixmap(QtGui.QPixmap(logo_path))
        self.logo_label.setScaledContents(True)
        self.logo_label.setFixedSize(100, 50)
        
        self.go_ai_button = QtWidgets.QPushButton("AI 客服")
        self.go_ai_button.setObjectName("go_ai_button")
        self.go_ai_button.setFixedSize(120, 50)

        self.top_layout.addWidget(self.back_home_button)
        self.top_layout.addStretch()
        self.top_layout.addWidget(self.logo_label)
        self.top_layout.addStretch()
        self.top_layout.addWidget(self.go_ai_button)
        self.verticalLayout.addWidget(self.top_frame)

        # --- Search Bar ---
        self.search_frame = QtWidgets.QFrame(self.centralwidget)
        self.search_layout = QtWidgets.QHBoxLayout(self.search_frame)
        self.search_layout.setContentsMargins(0, 0, 0, 0)

        self.search_input = QtWidgets.QLineEdit()
        self.search_input.setPlaceholderText("輸入系列名稱 (例如: 米開朗)...")
        # Ensure height is sufficient
        self.search_input.setMinimumHeight(50)
        self.search_input.setObjectName("search_input")
        
        self.search_button = QtWidgets.QPushButton("查詢")
        self.search_button.setObjectName("search_button")
        self.search_button.setFixedSize(100, 50)
        
        self.search_layout.addWidget(self.search_input)
        self.search_layout.addWidget(self.search_button)
        self.verticalLayout.addWidget(self.search_frame)

        # --- Filter Group ---
        self.filter_group = QtWidgets.QGroupBox("進階篩選條件")
        self.filter_group.setObjectName("filter_group")
        self.filter_layout = QtWidgets.QVBoxLayout(self.filter_group)
        self.filter_layout.setSpacing(5)

        # Helper to create sliders
        def create_filter_row(label_text, min_v, max_v, unit=""):
            row_widget = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            
            lbl_title = QtWidgets.QLabel(label_text)
            lbl_title.setFixedWidth(80)
            
            slider = RangeSlider(min_val=min_v, max_val=max_v)
            
            lbl_value = QtWidgets.QLabel(f"{min_v} - {max_v} {unit}")
            lbl_value.setFixedWidth(120)
            lbl_value.setAlignment(Qt.AlignCenter)
            
            slider.valueChanged.connect(lambda l, h: lbl_value.setText(f"{l} - {h} {unit}"))

            row_layout.addWidget(lbl_title)
            row_layout.addWidget(slider)
            row_layout.addWidget(lbl_value)
            
            self.filter_layout.addWidget(row_widget)
            return slider, lbl_value

        self.slider_watt, self.lbl_watt = create_filter_row("功率 (W)", 0, 200, "W")
        self.slider_cct,  self.lbl_cct  = create_filter_row("色溫 (K)", 2000, 7000, "K")
        self.slider_beam, self.lbl_beam = create_filter_row("光束角 (°)", 0, 120, "°")
        self.slider_lumen,self.lbl_lumen= create_filter_row("流明 (lm)", 0, 15000, "")
        self.slider_price,self.lbl_price= create_filter_row("價格 ($)", 0, 200000, "")

        self.verticalLayout.addWidget(self.filter_group)

        # --- Results Area ---
        self.result_area = QtWidgets.QScrollArea(self.centralwidget)
        self.result_area.setObjectName("result_area")
        self.result_area.setWidgetResizable(True)
        self.result_content = QtWidgets.QWidget()
        self.result_layout = QtWidgets.QVBoxLayout(self.result_content)
        self.result_layout.setAlignment(QtCore.Qt.AlignTop)
        self.result_area.setWidget(self.result_content)
        
        self.verticalLayout.addWidget(self.result_area)
        self.verticalLayout.setStretchFactor(self.result_area, 1)

        ModelSearchWindow.setCentralWidget(self.centralwidget)

# ========================================================
# 6. 主邏輯 (ModelSearchPage)
# ========================================================
class ModelSearchPage(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_ModelSearchWindow()
        self.ui.setupUi(self)
        self.apply_style()

        # 嘗試載入資料
        if BACKEND_AVAILABLE:
            try:
                print("正在呼叫後端載入資料...")
                status = finalBackend.load_products()
                if not status['ok']:
                    QtWidgets.QMessageBox.warning(self, "資料錯誤", f"無法載入資料: {status['message']}")
                else:
                    print("資料載入完成")
            except Exception as e:
                print(f"後端 load_products 執行錯誤: {e}")
        
        # 綁定事件
        self.ui.search_button.clicked.connect(self.run_search)
        self.ui.search_input.returnPressed.connect(self.run_search)
        self.ui.back_home_button.clicked.connect(self.go_home)
        self.ui.go_ai_button.clicked.connect(self.go_ai_chat)

    def apply_style(self):
        qss = """
        /* FIX 1: Set text color to dark gray globally to fix invisible text on Mac */
        QWidget { 
            background-color: white; 
            font-family: ".AppleSystemUIFont", "Arial", sans-serif; 
            font-size: 15px; 
            color: #333333; 
        }
        
        /* FIX 2: Explicitly style labels to ensure visibility */
        QLabel {
            color: #333333;
        }

        /* FIX 3: Style the input box with border and height */
        QLineEdit { 
            border: 2px solid #cccccc; 
            border-radius: 8px; 
            padding: 8px; 
            min-height: 30px; 
            color: #333333;
        }
        QLineEdit:focus { border: 2px solid #185ca1; }
        
        QPushButton { 
            background-color: #185ca1; color: white; border-radius: 8px; font-weight: bold; 
        }
        QPushButton:hover { background-color: #f17039; }
        
        #back_home_button, #go_ai_button {
            background-color: white; color: #185ca1; border: 2px solid #ccc;
        }
        #back_home_button:hover, #go_ai_button:hover {
            color: #f17039; border-color: #f17039;
        }

        QGroupBox { 
            border: 1px solid #ddd; 
            border-radius: 8px; 
            margin-top: 10px; 
            padding-top: 15px; 
            font-weight: bold; 
            color: #333333; /* Ensure GroupBox title is visible */
        }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }

        #result_card { 
            background-color: white; border-radius: 10px; border: 1px solid #e0e0e0; margin-bottom: 5px;
        }
        #result_card:hover { 
            border: 2px solid #185ca1; background-color: #f0f8ff; 
        }
        #result_title { font-size: 16px; font-weight: bold; color: #185ca1; }
        #result_detail { font-size: 13px; color: #666; }
        #result_price { font-size: 15px; font-weight: bold; color: #f17039; }
        """
        self.setStyleSheet(qss)

    def run_search(self):
        keyword = self.ui.search_input.text().strip()
        
        # 取得參數
        params = {
            'series_keyword': keyword,
            'watt_lo': self.ui.slider_watt.low, 'watt_hi': self.ui.slider_watt.high,
            'cct_lo': self.ui.slider_cct.low,   'cct_hi': self.ui.slider_cct.high,
            'beam_lo': self.ui.slider_beam.low, 'beam_hi': self.ui.slider_beam.high,
            'lumen_lo': self.ui.slider_lumen.low, 'lumen_hi': self.ui.slider_lumen.high,
            'price_lo': self.ui.slider_price.low, 'price_hi': self.ui.slider_price.high,
            'topk': 50
        }
        
        print(f"搜尋參數: {params}")

        results = []
        
        # 呼叫後端
        if BACKEND_AVAILABLE:
            try:
                resp = finalBackend.filter_products(**params)
                if resp['ok']:
                    results = resp['items']
                    if not results:
                        self.show_message("提示", "查無符合條件的產品")
                else:
                    self.show_message("後端錯誤", resp['message'])
            except Exception as e:
                print(f"執行 filter_products 時發生例外: {e}")
                self.show_message("系統錯誤", str(e))
        else:
            print("後端無法使用或找不到後端(finalBackend.py)")
            

        # 清除並顯示
        self.clear_results()
        for item in results:
            card = self.create_result_card(item)
            self.ui.result_layout.addWidget(card)

    def clear_results(self):
        while self.ui.result_layout.count():
            item = self.ui.result_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def create_result_card(self, item):
        frame = QtWidgets.QFrame()
        frame.setObjectName("result_card")
        layout = QtWidgets.QHBoxLayout(frame)
        
        # 圖片
        img_label = QtWidgets.QLabel()
        logo_path = os.path.join(current_dir, "dancelight_logo.jpg")
        pixmap = QtGui.QPixmap(logo_path) if os.path.exists(logo_path) else QtGui.QPixmap(80, 80)
        if not os.path.exists(logo_path): pixmap.fill(QtGui.QColor("#eee"))
            
        img_label.setPixmap(pixmap.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        img_label.setFixedSize(80, 80)

        # 文字資訊
        text_layout = QtWidgets.QVBoxLayout()
        title = f"{item.get('series', '')} {item.get('model', 'Unknown')}"
        lbl_title = QtWidgets.QLabel(title)
        lbl_title.setObjectName("result_title")
        # ----------------------------------------------------
        # FIX: Enable text selection
        # ----------------------------------------------------
        lbl_title.setTextInteractionFlags(Qt.TextSelectableByMouse)
        
        details = (f"瓦數: {item.get('watt')}W | 色溫: {item.get('cct')}K | "
                   f"光通量: {item.get('lumen')}lm | 光束角: {item.get('beam')}°")
        lbl_desc = QtWidgets.QLabel(details)
        lbl_desc.setObjectName("result_detail")
        lbl_desc.setWordWrap(True)
        # ----------------------------------------------------
        # FIX: Enable text selection
        # ----------------------------------------------------
        lbl_desc.setTextInteractionFlags(Qt.TextSelectableByMouse)

        text_layout.addWidget(lbl_title)
        text_layout.addWidget(lbl_desc)

        # 價格
        p_val = item.get('price_from', item.get('price', 0))
        lbl_price = QtWidgets.QLabel(f"${p_val}")
        lbl_price.setObjectName("result_price")
        lbl_price.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        # ----------------------------------------------------
        # FIX: Enable text selection
        # ----------------------------------------------------
        lbl_price.setTextInteractionFlags(Qt.TextSelectableByMouse)

        layout.addWidget(img_label)
        layout.addLayout(text_layout)
        layout.addWidget(lbl_price)
        
        return frame

    def show_message(self, title, msg):
        QtWidgets.QMessageBox.information(self, title, msg)

    def go_home(self):
        Home = get_home_page()
        if Home:
            self.home_window = Home()
            self.home_window.show()
            self.close()

    def go_ai_chat(self):
        Chat = get_ai_chat_page()
        if Chat:
            self.chat_window = Chat()
            self.chat_window.show()
            self.close()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ModelSearchPage()
    window.show()
    sys.exit(app.exec_())