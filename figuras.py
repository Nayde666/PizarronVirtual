'''
El proposito que tengo para este programa es:
    para dibujar sobre un lienzo, distintas figuras y que cuando 
    el dedo indice y el dedo pulgar sea el tamaño más pequeño que pueda
    tener la figura, (el temaño se puede establecer) y conforme
    separemos ambos dedos, la figura aumente en cuestion de tamaño
'''

import cv2
import mediapipe as mp
import numpy as np

# --------------------------------------------------------------
# Inicialización de MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# --------------------------------------------------------------
# Configuración de la cámara
captura = cv2.VideoCapture(1)
captura.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# --------------------------------------------------------------
# funciones para las figuras
# calcula la distancia euclidiana entre dos puntos
def calculate_distance(p1, p2):
    return int(np.linalg.norm(np.array(p1) - np.array(p2)))
# dibujar un triangulo
def draw_triangle(img, center, size, color):
    half_size = size // 2
    points = np.array([
        [center[0], center[1] - half_size],
        [center[0] - half_size, center[1] + half_size],
        [center[0] + half_size, center[1] + half_size]
    ], np.int32)
    cv2.fillPoly(img, [points], color)
# dibujar un pentagono
def draw_pentagon(img, center, size, color):
    angle = np.deg2rad(360 / 5)
    points = [
        (
            int(center[0] + size * np.cos(i * angle - np.pi / 2)),
            int(center[1] + size * np.sin(i * angle - np.pi / 2))
        ) for i in range(5)
    ]
    points = np.array(points, np.int32)
    cv2.fillPoly(img, [points], color)
# dibujar figuras basicsa 
def draw_saved_shapes(frame, posiciones):
    for pos in posiciones:
        if pos['figura'] == 'square':
            top_left = (pos['x'] - pos['tamaño'] // 2, pos['y'] - pos['tamaño'] // 2)
            bottom_right = (pos['x'] + pos['tamaño'] // 2, pos['y'] + pos['tamaño'] // 2)
            cv2.rectangle(frame, top_left, bottom_right, pos['color'], -1)
        elif pos['figura'] == 'circle':
            cv2.circle(frame, (pos['x'], pos['y']), pos['tamaño'] // 2, pos['color'], -1)
        elif pos['figura'] == 'triangle':
            draw_triangle(frame, (pos['x'], pos['y']), pos['tamaño'], pos['color'])
        elif pos['figura'] == 'pentagon':
            draw_pentagon(frame, (pos['x'], pos['y']), pos['tamaño'], pos['color'])
# --------------------------------------------------------------
# variables globales
figura_geometrica = None
tamaño_figura = 50
color_figura = (255, 0, 255)
posiciones_dibujadas = []
gesture_captured = False  # Bandera para evitar redibujar

# --------------------------------------------------------------
# Procesamiento principal con MediaPipe
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while captura.isOpened():
        ret, frame = captura.read()
        if not ret:
            break

        # preparar imagen
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = hands.process(img_rgb)

        # dibujar figuras guardadas
        draw_saved_shapes(frame, posiciones_dibujadas)
        # --------------------------------------------------------------
        # mostrar botones de seleccion de figuras
        cv2.rectangle(frame, (60, 380), (100, 420), (255, 0, 255), -1)  # cuadrado
        cv2.circle(frame, (180, 400), 25, (0, 255, 255), -1)  # circul
        draw_triangle(frame, (300, 400), 40, (0, 165, 255))  # triangulo
        draw_pentagon(frame, (420, 400), 30, (255, 0, 0))  # pentagon o
        # --------------------------------------------------------------
        # procesar ambos manos detectadas
        if resultados.multi_hand_landmarks:
            right_hand, left_hand = None, None

            for idx, hand_landmarks in enumerate(resultados.multi_hand_landmarks):
                handedness = resultados.multi_handedness[idx].classification[0].label
                if handedness == 'Right':
                    right_hand = hand_landmarks
                elif handedness == 'Left':
                    left_hand = hand_landmarks

            # mano derecha para seleccionar figura
            if right_hand:
                landmarks = right_hand.landmark
                x_indice = int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width)
                y_indice = int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)
                x_pulgar = int(landmarks[mp_hands.HandLandmark.THUMB_TIP].x * width)
                y_pulgar = int(landmarks[mp_hands.HandLandmark.THUMB_TIP].y * height)

                # detección de selección
                if 60 < x_indice < 100 and 380 < y_indice < 420:
                    figura_geometrica, color_figura = 'square', (255, 0, 255)
                elif (x_indice - 180) ** 2 + (y_indice - 400) ** 2 < 25 ** 2:
                    figura_geometrica, color_figura = 'circle', (0, 255, 255)
                elif 280 < x_indice < 320 and 380 < y_indice < 420:
                    figura_geometrica, color_figura = 'triangle', (0, 165, 255)
                elif 390 < x_indice < 450 and 380 < y_indice < 420:
                    figura_geometrica, color_figura = 'pentagon', (255, 0, 0)

                # ajustar tamaño de las fuguras con distancia índice-pulgar
                distancia = calculate_distance((x_indice, y_indice), (x_pulgar, y_pulgar))
                tamaño_figura = max(20, min(200, distancia)) # a elección

                if figura_geometrica == 'square':
                    cv2.rectangle(frame, (x_indice - tamaño_figura // 2, y_indice - tamaño_figura // 2),
                                  (x_indice + tamaño_figura // 2, y_indice + tamaño_figura // 2), color_figura, -1)
                elif figura_geometrica == 'circle':
                    cv2.circle(frame, (x_indice, y_indice), tamaño_figura // 2, color_figura, -1)
                elif figura_geometrica == 'triangle':
                    draw_triangle(frame, (x_indice, y_indice), tamaño_figura, color_figura)
                elif figura_geometrica == 'pentagon':
                    draw_pentagon(frame, (x_indice, y_indice), tamaño_figura, color_figura)

            # mano izquierda para gesto de confirmacion para dibujar figura
            if left_hand and figura_geometrica:
                y_indice_izq = left_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                y_medio_izq = left_hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y

                if y_indice_izq < left_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and \
                   y_medio_izq < left_hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and not gesture_captured:
                    posiciones_dibujadas.append({
                        'figura': figura_geometrica,
                        'x': x_indice,
                        'y': y_indice,
                        'tamaño': tamaño_figura,
                        'color': color_figura
                    })
                    gesture_captured = True  #evita múltiples registros del mismo gesto

                if y_indice_izq > left_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y:
                    gesture_captured = False  # reiniciar bandera

        # Mostrar ventana
        cv2.imshow("Dibujar con gestos de manos", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
            break

captura.release()
cv2.destroyAllWindows()

