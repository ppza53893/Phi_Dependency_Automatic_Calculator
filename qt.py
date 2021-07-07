import PyQt5.QtWidgets as Widget

app = Widget.QApplication([])
button = Widget.QPushButton('click')

def button_click():
    aleart = Widget.QMessageBox()
    aleart.setText('You Clicked the button!')
    aleart.exec_()

button.clicked.connect(button_click)
button.show()
app.exec_()