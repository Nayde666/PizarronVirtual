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
mp_drawing = mp.solutions.drawing_utils # 21 puntos
mp_hands = mp.solutions.hands # Implementación de mediapipe handaa
#----------------------------------------------------------------
# capturar video con cv2
captura = cv2.VideoCapture(1) # 0 camara de la lap, 1 camara del celular
captura.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
#----------------------------------------------------------------
# configuracion medipipe
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Detectar ambas manos
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5  # Estos últimos 2, son valores recomendados por defecto
) as hands:
    figura_geometrica = None  # seleccionará 'circle', 'square' o etc.
    tamaño_inicial_fig = 50 
    color_figura = (175, 255, 10)

    # Calcular la distancia entre dos puntos
    def calculate_distance(p1, p2):
        return int(np.linalg.norm(np.array(p1) - np.array(p2)))

    while captura.isOpened():
        ret, frame = captura.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # espejo
        height, width, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = hands.process(img_rgb)

        # botones dibujados en la pantalla
        cv2.rectangle(frame, (50, 50), (150, 150), (255, 100, 0), -1)
        cv2.circle(frame, (250, 100), 50, (238, 198, 0), -1)
        
        if resultados.multi_hand_landmarks:
            for hand_landmarks in resultados.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS
                )
                landmarks = hand_landmarks.landmark

                # index_tip = (int(landmarks[8].x * width), int(landmarks[8].y * height)) #indice 8 corresponde a la punta deldedo
                # thumb_tip = (int(landmarks[4].x * width), int(landmarks[4].y * height)) #pulgar
                # coordenadas de la punta de los dedos
                x_indice = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width)
                y_indice = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)
                x_menique = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * width)
                y_menique = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * height)
                x_pulgar = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * width)
                y_pulgar = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * height)


                # deteccion de seleccion
                if 50 < x_indice < 150 and 50 < y_indice < 150:
                    figura_geometrica = 'square'
                elif (x_indice - 250) ** 2 + (y_indice - 100) ** 2 < 50 ** 2:
                    figura_geometrica = 'circle'

                # ajustr el tamaño de la figura (usando la distancia entre el índice y el pulgar)
                distancia = calculate_distance((x_indice, y_indice), (x_pulgar, y_pulgar))
                tamaño_figura = max(20, min(200, distancia))

                # mostrar la figura selecciondad
                if figura_geometrica == 'square':
                    top_left = (x_indice - tamaño_figura // 2, y_indice - tamaño_figura // 2)
                    bottom_right = (x_indice + tamaño_figura // 2, y_indice + tamaño_figura // 2)
                    cv2.rectangle(frame, top_left, bottom_right, color_figura, -1)
                elif figura_geometrica == 'circle':
                    cv2.circle(frame, (x_indice, y_indice), tamaño_figura // 2, color_figura, -1)

        cv2.imshow("Dibujar con gestos de manos", frame)
        # esc
        k = cv2.waitKey(1)
        if k == 27:
            break

captura.release()
cv2.destroyAllWindows()
