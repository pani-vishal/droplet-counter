import math
import sys
import os
import typing

import PyQt5
import PyQt5.QtCore
from PyQt5 import QtGui, QtCore, QtWidgets
import importlib

import numpy as np


import src.droplet_classification.evaluate_model

from PyQt5.QtCore import QLibraryInfo
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
    QLibraryInfo.PluginsPath
)

class Droplet:
    def __init__(self, radius, position, state):
        self.radius = radius
        self.state = state
        self.position = position
        self.listeners = []

    def get_state(self):
        return self.state

    def get_radius(self):
        return self.radius

    def get_position(self):
        return self.position

    def set_state(self, state):
        self.state = state
        self.update_listeners()

    def set_radius(self, radius):
        self.radius = radius
        self.update_listeners()

    def set_position(self, position):
        self.position = position
        self.update_listeners()

    def add_listener(self, listener):
        self.listeners.append(listener)
        self.update_listeners()

    def del_listener(self, listener):
        self.listeners.remove(listener)
        self.update_listeners()

    def update_listeners(self):
        for listener in self.listeners:
            listener.update_droplet()

class DropletDatabase:
    def __init__(self):
        self.droplets = []

    def add_droplet(self, droplet):
        self.droplets.append(droplet)

    def delete_droplet(self, droplet):
        self.droplets.remove(droplet)

    def clear(self):
        self.droplets = []

    def export(self):
        export = ""
        for droplet in self.droplets:
            x, y = droplet.get_position()
            export += f"{x}, {y}, {droplet.get_radius()}, {droplet.get_state()}\n"
        return export

class DropletWidgetColorGenerator:
    @staticmethod
    def get_color_from_state(state):
        if state == "full":
            return QtCore.Qt.green
        elif state == "empty":
            return QtCore.Qt.red
        elif state == "not_a_droplet":
            return QtCore.Qt.blue
        else:
            return QtCore.Qt.yellow

class DropletCircleWidget(QtWidgets.QGraphicsItem):

    def __init__(self, droplet, droplet_color_generator, parent=None):
        # call constructor of QGraphicsItem
        super(DropletCircleWidget, self).__init__(None)

        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsFocusable, True)

        self.parent = parent

        self.setAcceptHoverEvents(True)

        self.droplet = droplet
        self.droplet_color_generator = droplet_color_generator

        self.boundRect = QtCore.QRectF(-1.1 * droplet.get_radius(), -1.1 * droplet.get_radius(),
                                       2.2 * droplet.get_radius(), 2.2 * droplet.get_radius())
        self.focusRect = QtCore.QRectF(-droplet.get_radius(), -droplet.get_radius(), 2 * droplet.get_radius(),
                                       2 * droplet.get_radius())

        x, y = self.droplet.get_position()
        self.setPos(x, y)

    def mouseMoveEvent(self, event):
        self.droplet.set_position((event.scenePos().toPoint().x(), event.scenePos().toPoint().y()))
        QtWidgets.QGraphicsItem.mouseMoveEvent(self, event)

    def mousePressEvent(self, event):
        self.setSelected(True)
        QtWidgets.QGraphicsItem.mousePressEvent(self, event)

    def boundingRect(self):
        return self.boundRect

    def paint(self, painter, option, widget=None):
        painter.setBrush(QtGui.QBrush(QtCore.Qt.NoBrush))
        pen = QtGui.QPen(QtCore.Qt.SolidLine)
        pen.setWidth(1)
        pen.setColor(self.droplet_color_generator.get_color_from_state(self.droplet.get_state()))
        painter.setPen(pen)
        painter.drawEllipse(-int(self.droplet.get_radius()), -int(self.droplet.get_radius()),
                            2 * int(self.droplet.get_radius()), 2 * int(self.droplet.get_radius()))
        if self.isSelected():
            self.drawFocusRect(painter)

    def drawFocusRect(self, painter):
        self.focusBrush = QtGui.QBrush()
        self.focusPen = QtGui.QPen(QtCore.Qt.DotLine)
        self.focusPen.setColor(QtCore.Qt.black)
        self.focusPen.setWidth(2)
        painter.setBrush(self.focusBrush)
        painter.setPen(self.focusPen)
        painter.drawRect(self.focusRect)

    def hoverEnterEvent(self, event):
        QtWidgets.QGraphicsItem.hoverEnterEvent(self, event)

    def hoverLeaveEvent(self, event):
        QtWidgets.QGraphicsItem.hoverLeaveEvent(self, event)

    def wheelEvent(self, event):
        wheel_dif = event.delta() / 120

        if self.isSelected():
            self.droplet.set_radius(self.droplet.get_radius() + wheel_dif)

            event.accept()
        else:
            super().wheelEvent(event)

    def keyPressEvent(self, event):
        if self.isSelected():
            if event.key() == QtCore.Qt.Key_1:
                self.droplet.set_state("full")
                event.accept()
            elif event.key() == QtCore.Qt.Key_2:
                self.droplet.set_state("empty")
                event.accept()
            elif event.key() == QtCore.Qt.Key_3:
                self.droplet.set_state("not_a_droplet")
                event.accept()
        else:
            super().keyPressEvent(event)

    def update_droplet(self):
        x_cur, y_cur = self.droplet.get_position()

        self.setPos(QtCore.QPoint(x_cur, y_cur))

        self.boundRect.setRect(- 1.1 * self.droplet.get_radius(),
                               - 1.1 * self.droplet.get_radius(),
                               2.2 * self.droplet.get_radius(),
                               2.2 * self.droplet.get_radius())

        self.focusRect.setRect(-self.droplet.get_radius(),
                               -self.droplet.get_radius(),
                               2 * self.droplet.get_radius(),
                               2 * self.droplet.get_radius())

        self.update(-2 * self.droplet.get_radius(),
                    -2 * self.droplet.get_radius(),
                    4 * self.droplet.get_radius(),
                    4 * self.droplet.get_radius())


class DropletGraphicsView(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super(DropletGraphicsView, self).__init__(parent)

        self.parent = parent

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Plus:
            zoom_in_factor = 1.1

            self.scale(zoom_in_factor, zoom_in_factor)

            event.accept()
        elif event.key() == QtCore.Qt.Key_Minus:
            zoom_out_factor = 1 / 1.1

            self.scale(zoom_out_factor, zoom_out_factor)

            event.accept()

        super().keyPressEvent(event)

class QHSeperationLine(QtWidgets.QFrame):
  '''
  a horizontal seperation line\n
  '''
  def __init__(self):
    super().__init__()
    self.setMinimumWidth(1)
    self.setFixedHeight(20)
    self.setFrameShape(QtWidgets.QFrame.HLine)
    self.setFrameShadow(QtWidgets.QFrame.Sunken)
    self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
    return

class DropletWindow(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.droplet_database = DropletDatabase()

        # set window name
        self.setWindowTitle("Droplet classifier v0.1")

        # resize window
        self.resize(1200, 800)

        self._createActions()

        # create central layout widget
        self.central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.central_widget)

        # horizontal layout, 1 - droplet image, 2 - options
        self.main_layout = QtWidgets.QHBoxLayout()

        self.droplet_scene = QtWidgets.QGraphicsScene()

        self.droplet_image_path = None
        self.droplet_image = None
        self.droplet_image_item = None

        self.droplet_widget_color_generator = DropletWidgetColorGenerator()

        self.droplet_view = DropletGraphicsView(self.droplet_scene)

        self.main_layout.addWidget(self.droplet_view, 1)

        self._createToolbox()
        self.main_layout.addWidget(self.toolbox_widget)

        self.central_widget.setLayout(self.main_layout)


        self._connectActions()
        self._createMenuBar()

    def _createMenuBar(self):
        self.menu_bar = QtWidgets.QMenuBar(self)
        self.setMenuBar(self.menu_bar)

        self.menu_bar.addAction(self.open_action)
        self.menu_bar.addAction(self.create_droplet_action)
        self.menu_bar.addAction(self.delete_droplet_action)
        self.menu_bar.addAction(self.delete_all_droplets_action)

    def _createActions(self):
        self.open_action = QtWidgets.QAction("&Open Image", self)
        self.open_action.setShortcut(QtCore.Qt.Key_O)

        self.create_droplet_action = QtWidgets.QAction("&New Droplet", self)
        self.create_droplet_action.setShortcut(QtCore.Qt.Key_N)

        self.delete_droplet_action = QtWidgets.QAction("&Remove Droplet", self)
        self.delete_droplet_action.setShortcut(QtCore.Qt.Key_R)

        self.delete_all_droplets_action = QtWidgets.QAction("&Clear all Droplets", self)
        self.delete_all_droplets_action.setShortcut(QtCore.Qt.Key_C)

        self.mark_droplet_circles_action = QtWidgets.QAction("&Mark Droplet Circles", self)

    def _createToolbox(self):
        self.toolbox_widget = QtWidgets.QWidget()
        self.toolbox_widget.setFixedWidth(400)
        self.toolbox_layout = QtWidgets.QVBoxLayout(self.toolbox_widget)

        self.circle_detector_groupbox = QtWidgets.QGroupBox("Circle detection")

        self.circle_detector_groupbox_layout = QtWidgets.QVBoxLayout()
        self.circle_detector_algorithms_combobox = QtWidgets.QComboBox()

        circle_detector_algorithms = [file.replace(".py", "") for file in os.listdir("src/circle_detection")]
        circle_detector_algorithms.remove("__pycache__")

        for algorithm in circle_detector_algorithms:
            self.circle_detector_algorithms_combobox.addItem(algorithm)

        self.circle_detector_algorithms_combobox.currentIndexChanged.connect(self.changeCircleDetectorConfigurator)
        self.circle_detector_groupbox_layout.addWidget(self.circle_detector_algorithms_combobox)

        self.circle_detector_configurator = None
        self.circle_detector_configurator_form_widget = None
        self.changeCircleDetectorConfigurator()

        self.circle_detector_groupbox.setLayout(self.circle_detector_groupbox_layout)

        self.toolbox_layout.addWidget(self.circle_detector_groupbox)

        self.circle_detector_run_button = QtWidgets.QPushButton("Run circle detection")
        self.circle_detector_run_button.clicked.connect(self.markDropletCircles)
        self.circle_detector_run_button.setEnabled(False)

        self.toolbox_layout.addWidget(self.circle_detector_run_button)


        ####
        self.droplet_classification_groupbox = QtWidgets.QGroupBox("Droplet classification")
        self.droplet_classification_groupbox_layout = QtWidgets.QVBoxLayout()
        self.droplet_classification_groupbox.setLayout(self.droplet_classification_groupbox_layout)

        self.droplet_classification_form_widget = QtWidgets.QWidget()
        self.droplet_classification_form_layout = QtWidgets.QFormLayout(self.droplet_classification_form_widget)

        self.droplet_classification_model = QtWidgets.QLineEdit('"models/vgg16_classifier"')
        self.droplet_classification_form_layout.addRow("Model", self.droplet_classification_model)

        self.droplet_classification_groupbox_layout.addWidget(self.droplet_classification_form_widget)
        self.toolbox_layout.addWidget(self.droplet_classification_groupbox)

        self.droplet_classification_button = QtWidgets.QPushButton("Run droplet classification")
        self.droplet_classification_button.clicked.connect(self.classifyDroplets)
        self.droplet_classification_button.setEnabled(False)

        self.toolbox_layout.addWidget(self.droplet_classification_button)

        ####
        self.toolbox_layout.addWidget(QHSeperationLine())

        self.export_data_button = QtWidgets.QPushButton("Export data")
        self.export_data_button.clicked.connect(self.exportData)
        self.toolbox_layout.addWidget(self.export_data_button)

        ####
        self.droplet_statistics_groupbox = QtWidgets.QGroupBox("Droplet statistics")
        self.droplet_statistics_groupbox_layout = QtWidgets.QVBoxLayout()
        self.droplet_statistics_groupbox.setLayout(self.droplet_statistics_groupbox_layout)


        self.toolbox_layout.addWidget(self.droplet_statistics_groupbox)



    def _connectActions(self):
        self.open_action.triggered.connect(self.openFile)
        self.create_droplet_action.triggered.connect(self.createDroplet)
        self.delete_droplet_action.triggered.connect(self.deleteDroplet)
        self.mark_droplet_circles_action.triggered.connect(self.markDropletCircles)
        self.delete_all_droplets_action.triggered.connect(self.deleteAllDroplets)

    def openFile(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                            "Image Files (*.png *.jpg *.bmp)", options=options)
        if filename:
            print(f"LOADING: {filename}")
            if self.droplet_image_item is not None:
                self.droplet_scene.clear()

            self.droplet_image_path = filename
            self.droplet_image = QtGui.QPixmap(filename)
            self.droplet_image_item = QtWidgets.QGraphicsPixmapItem(self.droplet_image)
            self.droplet_image_item.setZValue(-10000)

            self.droplet_scene.addItem(self.droplet_image_item)

            self.droplet_view.setMaximumSize(self.droplet_image.size().width(), self.droplet_image.size().height())
            self.droplet_view.setSceneRect(
                QtCore.QRectF(0, 0, self.droplet_image.size().width(), self.droplet_image.size().height()))

            self.droplet_view.fitInView(
                QtCore.QRectF(0, 0, self.droplet_image.size().width(), self.droplet_image.size().height()),
                QtCore.Qt.KeepAspectRatio)

            self.circle_detector_run_button.setEnabled(True)
            self.droplet_classification_button.setEnabled(True)

    def createDroplet(self):
        origin = self.droplet_view.mapFromGlobal(QtGui.QCursor.pos())
        relative_origin = self.droplet_view.mapToScene(origin)

        mouse_position_x = relative_origin.toPoint().x()
        mouse_position_y = relative_origin.toPoint().y()

        if self.droplet_image_item is None:
            return

        if mouse_position_x < 0 \
                or mouse_position_y < 0 \
                or mouse_position_x > self.droplet_image.size().width() \
                or mouse_position_y > self.droplet_image.size().height():
            mouse_position_x = self.droplet_image.size().width() / 2.0
            mouse_position_y = self.droplet_image.size().height() / 2.0

        new_droplet = Droplet(20, (int(mouse_position_x), int(mouse_position_y)), "full")

        droplet_widget_new = DropletCircleWidget(new_droplet, self.droplet_widget_color_generator)
        new_droplet.add_listener(droplet_widget_new)

        self.droplet_scene.addItem(droplet_widget_new)
        self.droplet_database.add_droplet(new_droplet)

    def deleteDroplet(self):
        selected_items = self.droplet_scene.selectedItems()
        for selected_item in selected_items:
            if isinstance(selected_item, DropletCircleWidget):
                self.droplet_database.delete_droplet(selected_item.droplet)
                self.droplet_scene.removeItem(selected_item)


    def deleteAllDroplets(self):
        self.droplet_database.clear()
        items_in_scene = self.droplet_scene.items()
        for item in items_in_scene:
            if isinstance(item, DropletCircleWidget):
                self.droplet_scene.removeItem(item)

    def markDropletCircles(self):
        self.deleteAllDroplets()
        algorithm_name = self.circle_detector_algorithms_combobox.currentText()
        module = importlib.import_module(f"src.circle_detection.{algorithm_name}")

        print(algorithm_name)
        circle_detector_func = getattr(module, f"{algorithm_name}")

        config = self.circle_detector_configurator.get_config()
        print(config)
        eval_config = {}

        for key in config:
            eval_config[key] = eval(config[key])

        circles = circle_detector_func(self.droplet_image_path, **eval_config)[0]

        for x, y, r in circles:
            new_droplet = Droplet(math.floor(r), (int(x), int(y)), "not_a_droplet")
            droplet_widget_new = DropletCircleWidget(new_droplet, self.droplet_widget_color_generator)
            new_droplet.add_listener(droplet_widget_new)
            self.droplet_scene.addItem(droplet_widget_new)

            self.droplet_database.add_droplet(new_droplet)


    def changeCircleDetectorConfigurator(self):
        algorithm_name = self.circle_detector_algorithms_combobox.currentText()

        module = importlib.import_module(f"src.circle_detection.{algorithm_name}")

        self.circle_detector_configurator = getattr(module, f"{algorithm_name}_gui_config")()

        if self.circle_detector_configurator_form_widget is not None:
            self.circle_detector_groupbox_layout.removeWidget(self.circle_detector_configurator_form_widget)
            self.circle_detector_configurator_form_widget.setVisible(False)

        self.circle_detector_configurator_form_widget = self.circle_detector_configurator.form_widget
        self.circle_detector_groupbox_layout.addWidget(self.circle_detector_configurator_form_widget)

        self.toolbox_layout.activate()

    def classifyDroplets(self):

        model_path = eval(self.droplet_classification_model.text())

        result = src.droplet_classification.evaluate_model.evaluate_model(self.droplet_image_path, self.droplet_database.droplets, model_path)

        predictions = np.argmax(result, axis=1)

        for droplet, prediction in zip(self.droplet_database.droplets, predictions):
            if prediction == 0:
                droplet.set_state("empty")
            elif prediction == 1:
                droplet.set_state("full")
            elif prediction == 2:
                droplet.set_state("not_a_droplet")


    def exportData(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export droplet data to file", os.path.splitext(self.droplet_image_path)[0] + ".txt",
                                                            "Text Files (*.txt)", options=options)

        if filename:
            outfile = open(filename, "w")
            outfile.write(f"{self.droplet_image_path}\n")
            outfile.write(self.droplet_database.export())
            outfile.close()



if __name__ == '__main__':
    # Create the Qt Application
    app = QtWidgets.QApplication(sys.argv)

    win = DropletWindow()
    win.show()

    # Create and show the form
    scene = QtWidgets.QGraphicsScene()

    view = DropletGraphicsView(scene)

    # Run the main Qt loop
    sys.exit(app.exec_())
