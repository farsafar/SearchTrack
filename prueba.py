from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.label import Label
from kivy.core.window import Window

from ultralytics import YOLO
import cv2

class CameraApp(App):
    def build(self):
        self.model = YOLO('yolov8n.pt')
        self.cap = cv2.VideoCapture(0)
        self.niveles = 0
        self.puntaje = 0

        layout = BoxLayout(orientation='vertical')

        self.image = Image(size_hint=(1, 0.8))
        layout.add_widget(self.image)

        self.message_label = Label(text='', size_hint=(1, 0.1), font_size=24, color=(1, 1, 1, 1))
        layout.add_widget(self.message_label)

        self.score_label = Label(text='Puntaje: 0', size_hint=(1, 0.1), font_size=24, color=(1, 1, 1, 1))
        layout.add_widget(self.score_label)

        Clock.schedule_interval(self.update, 1.0 / 30.0)

        return layout

    def update(self, dt):
        ret, frame = self.cap.read()

        if not ret:
            return

        results = self.model.track(frame, persist=True)

        for result in results:
            if self.niveles == 0 and result.boxes.cls[0] == 0:
                self.message_label.text = "Persona detectada. Busca un perro"
                self.niveles = 1
                self.puntaje += 10
            elif self.niveles == 1 and result.boxes.cls[0] == 16:
                self.message_label.text = "Perro detectado. Busca un carro"
                self.niveles = 2
                self.puntaje += 10
            elif self.niveles == 2 and result.boxes.cls[0] == 2:
                self.message_label.text = "Carro detectado. Busca un gato"
                self.niveles = 3
                self.puntaje += 10
            elif self.niveles == 3 and result.boxes.cls[0] == 17:
                self.message_label.text = "Gato detectado. Busca un árbol"
                self.niveles = 4
                self.puntaje += 10
            elif self.niveles == 4 and result.boxes.cls[0] == 5:
                self.message_label.text = "Árbol detectado. ¡Has completado el juego!"
                self.niveles = 5
                self.puntaje += 10
                break

        # Mostrar el puntaje actualizado
        self.score_label.text = f"Puntaje: {self.puntaje}"

        # Mostrar la imagen con los resultados
        annotated_frame = results[0].plot()
        buf1 = cv2.flip(annotated_frame, 0)
        buf = buf1.tostring()
        texture = Texture.create(size=(annotated_frame.shape[1], annotated_frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture

    def on_stop(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    Window.size = (800, 600)
    Window.clearcolor = (0.1, 0.1, 0.1, 1)
    CameraApp().run()