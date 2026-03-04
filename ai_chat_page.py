# -*- coding: utf-8 -*-
import sys
import os
import torch
from PyQt5 import QtCore, QtGui, QtWidgets

# FIX 1: Corrected imports to match docling_rag_v5.py
from docling_rag_v5 import RAGSystem, Config

# ---------- Background Worker Thread ----------
class RAGWorker(QtCore.QThread):
    """Handles AI retrieval and generation in the background to prevent UI freezing."""
    answer_ready = QtCore.pyqtSignal(dict)

    def __init__(self, rag_system, question):
        super().__init__()
        self.rag_system = rag_system
        self.question = question

    def run(self):
        result = self.rag_system.query(self.question)
        self.answer_ready.emit(result)

# ---------- UI Layout Class ----------
class Ui_AIChatWindow(object):
    def setupUi(self, AIChatWindow):
        AIChatWindow.setObjectName("AIChatWindow")
        AIChatWindow.resize(600, 700)
        
        self.centralwidget = QtWidgets.QWidget(AIChatWindow)
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setContentsMargins(15, 15, 15, 15)
        self.verticalLayout.setSpacing(10)

        # Top Navigation Bar
        self.top_button_frame = QtWidgets.QFrame(self.centralwidget)
        self.top_button_layout = QtWidgets.QHBoxLayout(self.top_button_frame)
        
        self.back_home_button = QtWidgets.QPushButton("回到首頁")
        self.back_home_button.setObjectName("back_home_button")
        self.back_home_button.setFixedSize(160, 60)
        
        self.logo_label = QtWidgets.QLabel()
        # Ensure 'dancelight_logo.jpg' exists in the same folder
        if os.path.exists("dancelight_logo.jpg"):
            self.logo_label.setPixmap(QtGui.QPixmap("dancelight_logo.jpg"))
        else:
            self.logo_label.setText("DanceLight")
            self.logo_label.setAlignment(QtCore.Qt.AlignCenter)
            
        self.logo_label.setScaledContents(True)
        self.logo_label.setFixedSize(120, 60)
        
        self.go_model_button = QtWidgets.QPushButton("型號查詢")
        self.go_model_button.setObjectName("go_model_button")
        self.go_model_button.setFixedSize(160, 60)

        self.top_button_layout.addWidget(self.back_home_button)
        self.top_button_layout.addStretch()
        self.top_button_layout.addWidget(self.logo_label)
        self.top_button_layout.addStretch()
        self.top_button_layout.addWidget(self.go_model_button)
        self.verticalLayout.addWidget(self.top_button_frame)

        # Chat Display Area
        self.chat_display = QtWidgets.QScrollArea(self.centralwidget)
        self.chat_display.setObjectName("chat_display")
        self.chat_display.setWidgetResizable(True)
        self.chat_display_content = QtWidgets.QWidget()
        self.chat_display_layout = QtWidgets.QVBoxLayout(self.chat_display_content)
        self.chat_display_layout.setAlignment(QtCore.Qt.AlignTop)
        self.chat_display.setWidget(self.chat_display_content)
        self.verticalLayout.addWidget(self.chat_display)

        # Input Area
        self.input_frame = QtWidgets.QFrame(self.centralwidget)
        self.input_frame.setObjectName("input_frame")
        self.input_layout = QtWidgets.QHBoxLayout(self.input_frame)
        self.input_layout.setContentsMargins(5, 5, 5, 5) # Padding for the frame
        
        self.input_text = QtWidgets.QLineEdit()
        self.input_text.setPlaceholderText("系統初始化中...")
        self.input_layout.addWidget(self.input_text)

        self.send_button = QtWidgets.QPushButton("發送")
        self.send_button.setObjectName("send_button")
        self.input_layout.addWidget(self.send_button)
        
        self.verticalLayout.addWidget(self.input_frame)
        AIChatWindow.setCentralWidget(self.centralwidget)

# ---------- Main Logic ----------
class AIChatPage(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_AIChatWindow()
        self.ui.setupUi(self)
        
        self.load_qss()

        # Initial State
        self.ui.input_text.setEnabled(False)
        self.ui.send_button.setEnabled(False)

        # Events
        self.ui.send_button.clicked.connect(self.send_message)
        self.ui.input_text.returnPressed.connect(self.send_message)
        self.ui.back_home_button.clicked.connect(self.go_home)

        # Delay RAG loading so window appears first
        QtCore.QTimer.singleShot(100, self.init_rag_after_show)

    def load_qss(self):
        # FIX 2: Added min-height and color to QLineEdit to fix Mac display issue
        qss = """
        QWidget { 
            background-color: white; 
            font-family: ".AppleSystemUIFont", "Arial", sans-serif; 
            font-size: 16px; 
        }
        
        #chat_display { 
            background-color: #f7f7f7; 
            border-radius: 12px; 
            padding: 10px; 
            border: 1px solid #e0e0e0; 
        }
        
        QWidget#chat_display_content { 
            background-color: #f7f7f7; 
        }
        
        .user_message { 
            background-color: #185ca1; 
            color: white; 
            border-radius: 12px; 
            padding: 8px 12px; 
            margin: 5px; 
        }
        
        .ai_message { 
            background-color: #f17039; 
            color: white; 
            border-radius: 12px; 
            padding: 8px 12px; 
            margin: 5px; 
        }
        
        #input_frame { 
            background-color: #ffffff; 
            border-radius: 25px; 
            border: 2px solid #cccccc; 
            padding: 5px; 
        }
        
        QLineEdit { 
            border: none; 
            font-size: 16px; 
            padding: 6px; 
            color: black;         /* Fix: Make text visible */
            min-height: 30px;     /* Fix: Prevent collapse on Mac */
            background-color: transparent;
        }
        
        #send_button { 
            background-color: #185ca1; 
            color: white; 
            border-radius: 20px; 
            padding: 8px 20px; 
            font-weight: bold; 
        }
        #send_button:hover { 
            background-color: #f17039; 
        }
        
        #back_home_button, #go_model_button { 
            background-color: white; 
            color: #185ca1; 
            border: 2px solid #185ca1; 
            border-radius: 8px; 
            font-size: 18px; 
            font-weight: bold; 
        }
        #back_home_button:hover, #go_model_button:hover { 
            color: #f17039; 
            border-color: #f17039; 
        }
        """
        self.setStyleSheet(qss)

    def init_rag_after_show(self):
        """Initialize RAG System"""
        print("正在載入 RAG 系統與模型...")
        try:
            # FIX 1: Using correct Config class name
            config = Config(
                pdf_path="2025舞光LED21st(單頁水印可搜尋).pdf",
                enable_ocr=True,
                enable_query_expansion=False
            )
            # FIX 1: Using correct RAGSystem class name
            self.rag_system = RAGSystem(config)
            self.rag_system.initialize()
            
            self.ui.input_text.setEnabled(True)
            self.ui.send_button.setEnabled(True)
            self.ui.input_text.setPlaceholderText("請輸入訊息...")
            self.ai_reply("您好！我是舞光 LED 客服 AI。已經為您載入最新 2025 型錄，請問想找什麼燈具嗎？")
            print("系統就緒")
            
        except Exception as e:
            print(f"Error initializing RAG: {e}")
            self.ai_reply(f"系統錯誤: {str(e)}")

    def send_message(self):
        msg = self.ui.input_text.text().strip()
        if msg:
            self.display_message(msg, is_user=True)
            self.ui.input_text.clear()
            self.ui.send_button.setEnabled(False)
            self.ui.input_text.setPlaceholderText("AI 正在檢索 388 頁型錄中...")

            self.worker = RAGWorker(self.rag_system, msg)
            self.worker.answer_ready.connect(self.handle_ai_response)
            self.worker.start()

    def go_home(self):
        try:
            from homePage import MainWindow
            self.home_window = MainWindow()
            self.home_window.show()
            self.close()
        except ImportError:
            print("Cannot find homePage.py")
    
    def handle_ai_response(self, result):
        self.ui.send_button.setEnabled(True)
        self.ui.input_text.setPlaceholderText("請輸入訊息...")
        answer = result.get("answer", "抱歉，目前連線不穩定。")
        self.ai_reply(answer)

    def display_message(self, text, is_user=True):
        h_layout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(text)
        label.setWordWrap(True)
        label.setMaximumWidth(400)
        
        if is_user:
            h_layout.addStretch()
            label.setProperty("class", "user_message")
        else:
            label.setProperty("class", "ai_message")
            h_layout.addStretch()
            
        h_layout.addWidget(label)
        self.ui.chat_display_layout.addLayout(h_layout)
        
        QtCore.QTimer.singleShot(50, lambda: self.ui.chat_display.verticalScrollBar().setValue(
            self.ui.chat_display.verticalScrollBar().maximum()
        ))

    def ai_reply(self, text):
        h_layout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel("")
        label.setWordWrap(True)
        label.setMaximumWidth(400)
        label.setProperty("class", "ai_message")
        h_layout.addWidget(label)
        h_layout.addStretch()
        self.ui.chat_display_layout.addLayout(h_layout)

        # Typewriter effect
        self.typing_index = 0
        self.typing_text = text
        self.typing_label = label
        self.typing_timer = QtCore.QTimer()
        self.typing_timer.setInterval(25)
        self.typing_timer.timeout.connect(self.type_next_character)
        self.typing_timer.start()

    def type_next_character(self):
        if self.typing_index < len(self.typing_text):
            self.typing_label.setText(self.typing_label.text() + self.typing_text[self.typing_index])
            self.typing_index += 1
            self.ui.chat_display.verticalScrollBar().setValue(self.ui.chat_display.verticalScrollBar().maximum())
        else:
            self.typing_timer.stop()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = AIChatPage()
    window.show()
    sys.exit(app.exec_())