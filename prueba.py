from ultralytics import YOLO
import cv2

# Cargar el modelo YOLOv8
model = YOLO('yolov8n.pt')

# Cargar el video
cap = cv2.VideoCapture(0)  

# Inicializar la variable de niveles y puntaje
niveles = 0
puntaje = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detectar y rastrear objetos
    results = model.track(frame, persist=True)

    # Verificar si se detecta un objeto y actualizar el puntaje
    for result in results:
        if niveles == 0 and result.boxes.cls[0] == 0:  
            cv2.putText(frame, "Persona detectada. Busca un perro", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            niveles = 1
            puntaje += 10
        elif niveles == 1 and result.boxes.cls[0] == 16:  
            cv2.putText(frame, "Perro detectado. Busca un carro", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            niveles = 2
            puntaje += 10
        elif niveles == 2 and result.boxes.cls[0] == 2:  
            cv2.putText(frame, "Carro detectado. Busca un gato", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            niveles = 3
            puntaje += 10
        elif niveles == 3 and result.boxes.cls[0] == 17:  
            cv2.putText(frame, "Gato detectado. Busca un árbol", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            niveles = 4
            puntaje += 10
        elif niveles == 4 and result.boxes.cls[0] == 5:  
            cv2.putText(frame, "Árbol detectado. ¡Has completado el juego!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            niveles = 5
            puntaje += 10
            break

    # Mostrar el puntaje en la pantalla
    cv2.putText(frame, f"Puntaje: {puntaje}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar los resultados en el video
    annotated_frame = results[0].plot()
    cv2.imshow('Detected Objects', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()