import cv2
import numpy as np
import mediapipe as mp
# --------------------------------------------------------------
# inicialización de MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
# --------------------------------------------------------------
# configuracion de la cmara
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
# Variables globales para las figuras
figura_geometrica = None
tamaño_figura = 50
color_figura = (255, 0, 255)
posiciones_dibujadas = []
gesto_capturado = False  
#------------------------------------------------
# asignacion de colores en BGR para el pincel
color_Blanco = (255, 255, 255)
color_Rojo = (2, 2, 140)
color_Naranja = (0, 120, 255)
color_Amarillo = (11, 203, 241)
color_Verde = (132, 255, 182)
color_Azul = (255, 229, 43)
color_Violeta = (178, 83, 142)
color_Rosa = (212, 0, 255)
color_inicial = color_Blanco
# --------------------------------------------------------------
# posiciones en la pantalla de los colores
seleccion_colores = {
    "blanco": {"color": color_Blanco, "posicion": (10, 6, 40, 40), "relleno": -1},
    "rojo": {"color": color_Rojo, "posicion": (50, 6, 80, 40), "relleno": -1},
    "naranja": {"color": color_Naranja, "posicion": (90, 6, 120, 40), "relleno": -1},
    "amarillo": {"color": color_Amarillo, "posicion": (130, 6, 160, 40), "relleno": -1},
    "verde": {"color": color_Verde, "posicion": (170, 6, 200, 40), "relleno": -1},
    "azul": {"color": color_Azul, "posicion": (210, 6, 240, 40), "relleno": -1},
    "violeta": {"color": color_Violeta, "posicion": (250, 6, 280, 40), "relleno": -1},
    "rosa": {"color": color_Rosa, "posicion": (290, 6, 320, 40), "relleno": -1},
}
#--------------------------------------------------------------------
limpiar_pantalla = {"posicion": (340, 6, 380, 40), "label": "limpiar", "color": (177, 77, 255)}
borrador = {"posicion": (430, 6, 470, 40), "label": "borrador", "color": (255, 255, 255)}
seleccion_del_grosor = {"posicion": (490, 6, 600, 50), "niveles": 20}

x1 = None
y1 = None
imAux = None
dibujando = False
grosor_inicial = 3
modo_borrador = False
#--------------------------------------------------------------------
#pamtalla completa
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#--------------------------------------------------------------------
# Configuracion mediapope
with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,#(detectarambas manos)
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5 # Estos ultimos 2, son valores recomendados por defecto
) as hands:
    while True:
        ret, frame = captura.read()
        if ret == False: break
        # efecto espejo
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        if imAux is None: imAux = np.zeros(frame.shape, dtype=np.uint8)# Matriz de 0s del mismo tamaño que gframe
        # ---------------------------------------------------SECCION SUPERIOR--------------------------------------------------------------
        # mostrar colores disponibles
        for nombre, color_informacion in seleccion_colores.items():
            x1_minicaja, y1_minicaja, x2_minicaja, y2_minicaja = color_informacion["posicion"]
            cv2.rectangle(frame, (x1_minicaja, y1_minicaja), (x2_minicaja, y2_minicaja), color_informacion["color"], color_informacion["relleno"])
        # mostrar botn de borrador
        icono_limpiar = cv2.imread('limpiartodo.png')
        icono_limpiar = cv2.resize(icono_limpiar, (40, 40))

        icono_borrador = cv2.imread('borrador.png')
        icono_borrador = cv2.resize(icono_borrador, (40, 40))

        x1_borrador, y1_borrador, x2_borrador, y2_borrador = borrador["posicion"]
        cv2.rectangle(frame, (x1_borrador, y1_borrador), (x2_borrador, y2_borrador), 2)
        #cv2.putText(frame, borrador["label"], (x1_borrador + 5, y1_borrador + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, borrador["color"], 2)
        frame[6:46, 430:470] = icono_borrador  # Icono de borrador

        # mostrar boton limpiar pantalla
        x1_clear, y1_clear, x2_clear, y2_clear = limpiar_pantalla["posicion"]
        cv2.rectangle(frame, (x1_clear, y1_clear), (x2_clear, y2_clear), 2)
        #cv2.putText(frame, limpiar_pantalla["label"], (x1_clear + 10, y1_clear + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, limpiar_pantalla["color"], 2)
        frame[6:46, 340:380] = icono_limpiar  # Icono de limpiar

        #mostrar barra para seleccionar el grosor del pincel
        x1_bar, y1_bar, x2_bar, y2_bar = seleccion_del_grosor["posicion"]
        cv2.rectangle(frame, (x1_bar, y1_bar), (x2_bar, y2_bar), (200, 200, 200), 2)
        cv2.putText(frame, f'Grosor: {grosor_inicial}', (x1_bar, y2_bar + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        #--------------------------------------------------------------------
        # Procesar manos y cambio d rbg a rgb 
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = hands.process(frame_rgb)

        mano_izquierda_abierta = False
        if resultados.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(resultados.multi_hand_landmarks, resultados.multi_handedness):
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS
                )
                landmarks = hand_landmarks.landmark
                etiqueta_mano = handedness.classification[0].label
                x_indice = int(hand_landmarks.landmark[8].x * width)
                y_indice = int(hand_landmarks.landmark[8].y * height)
                x_medio = int(hand_landmarks.landmark[12].x * width)
                y_medio = int(hand_landmarks.landmark[12].y * height)
                
                #modo borrador
                if x1_borrador <= x_indice <= x2_borrador and y1_borrador <= y_indice <= y2_borrador:
                    modo_borrador = True
                    color_inicial = (0, 0, 0)  # color negro para el borrador
                else:
                    modo_borrador = False

                # Detectar si la mano izquierda está abierta (se detiene el dibujo)
                dedos_abiertos = sum(
                    hand_landmarks.landmark[i].y < hand_landmarks.landmark[i - 2].y
                    for i in [8, 12, 16, 20]
                )
                if etiqueta_mano == "Left" and dedos_abiertos == 4:
                    mano_izquierda_abierta = True

                # cambiar color si el dedo indice esta sobre uno de los cuadros de color
                for nombre, color_informacion in seleccion_colores.items():
                    x1_minicaja, y1_minicaja, x2_minicaja, y2_minicaja = color_informacion["posicion"]
                    if x1_minicaja <= x_indice <= x2_minicaja and y1_minicaja <= y_indice <= y2_minicaja:
                        color_inicial = color_informacion["color"]
                        # Resaltar el cuadro seleccionado
                        for key in seleccion_colores.keys():
                            seleccion_colores[key]["relleno"] = 2
                        seleccion_colores[nombre]["relleno"] = -1
                # limpiar toda la pantalla al seleccionar el boton
                if x1_clear <= x_indice <= x2_clear and y1_clear <= y_indice <= y2_clear:
                    imAux = np.zeros(frame.shape, dtype=np.uint8)
                    posiciones_dibujadas.clear()  

                # seleccionar el grosor del pincel al mover el dedo índice sobre la barra
                if x1_bar <= x_indice <= x2_bar and y1_bar <= y_indice <= y2_bar:
                    grosor_inicial = int((x_indice - x1_bar) * seleccion_del_grosor["niveles"] / (x2_bar - x1_bar))
                    grosor_inicial = max(1, min(grosor_inicial, seleccion_del_grosor["niveles"]))

                # Dibujar solo con la mano derecha cuando índice y medio están levantados
                if etiqueta_mano == "Right" and hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y \
                        and hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y:
                    if not dibujando:
                        dibujando = True
                        x1, y1 = x_indice, y_indice
                    else:
                        color_a_usar = color_inicial if not modo_borrador else (0, 0, 0)
                        cv2.line(imAux, (x1, y1), (x_indice, y_indice), color_a_usar, grosor_inicial)
                        x1, y1 = x_indice, y_indice

        if mano_izquierda_abierta:
            dibujando = False
            x1, y1 = None, None
        # ---------------------------------------------------SECCION INFERIOR--------------------------------------------------------------
        # Dibujar figuras guardadas y tambn en auxiliar
        draw_saved_shapes(frame, posiciones_dibujadas)
        draw_saved_shapes(imAux, posiciones_dibujadas)

        # Dibujar botones de selección
        cv2.rectangle(frame, (60, 380), (100, 420), (255, 0, 255), -1)  # Cuadrado
        cv2.circle(frame, (180, 400), 25, (0, 255, 255), -1)  # Círculo
        draw_triangle(frame, (300, 400), 40, (0, 165, 255))  # Triángulo
        draw_pentagon(frame, (420, 400), 30, (255, 0, 0))  # Pentágono
        #--------------------------------------------------------------------
        # Procesar manos detectadas
        if resultados.multi_hand_landmarks:
            right_hand, left_hand = None, None

            for idx, hand_landmarks in enumerate(resultados.multi_hand_landmarks):
                handedness = resultados.multi_handedness[idx].classification[0].label
                if handedness == 'Right':
                    right_hand = hand_landmarks
                elif handedness == 'Left':
                    left_hand = hand_landmarks
            #--------------------------------------------------------------------
            # procesar mano derecha para seleccionar figura
            if right_hand:
                landmarks = right_hand.landmark
                x_indice = int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width)
                y_indice = int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)
                x_pulgar = int(landmarks[mp_hands.HandLandmark.THUMB_TIP].x * width)
                y_pulgar = int(landmarks[mp_hands.HandLandmark.THUMB_TIP].y * height)
                #--------------------------------------------------------------------
                # detectar figura seleccionada
                if 60 < x_indice < 100 and 380 < y_indice < 420:
                    figura_geometrica, color_figura = 'square', (255, 0, 255)
                elif (x_indice - 180) ** 2 + (y_indice - 400) ** 2 < 25 ** 2:
                    figura_geometrica, color_figura = 'circle', (0, 255, 255)
                elif 280 < x_indice < 320 and 380 < y_indice < 420:
                    figura_geometrica, color_figura = 'triangle', (0, 165, 255)
                elif 390 < x_indice < 450 and 380 < y_indice < 420:
                    figura_geometrica, color_figura = 'pentagon', (255, 0, 0)
                #--------------------------------------------------------------------
                # ajustar tamaño con distancia índice-pulgar
                distancia = calculate_distance((x_indice, y_indice), (x_pulgar, y_pulgar))
                tamaño_figura = max(20, min(200, distancia))
                #--------------------------------------------------------------------
                # mostrar la figura seleccionada en tiempo real
                if figura_geometrica == 'square':
                    cv2.rectangle(frame, (x_indice - tamaño_figura // 2, y_indice - tamaño_figura // 2),
                                  (x_indice + tamaño_figura // 2, y_indice + tamaño_figura // 2), color_figura, -1)
                elif figura_geometrica == 'circle':
                    cv2.circle(frame, (x_indice, y_indice), tamaño_figura // 2, color_figura, -1)
                elif figura_geometrica == 'triangle':
                    draw_triangle(frame, (x_indice, y_indice), tamaño_figura, color_figura)
                elif figura_geometrica == 'pentagon':
                    draw_pentagon(frame, (x_indice, y_indice), tamaño_figura, color_figura)
            #--------------------------------------------------------------------
            # procesar mano izquierda para gesto de confirmacion de la figura a dibujar
            if left_hand and figura_geometrica:
                y_indice_izq = left_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                y_medio_izq = left_hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y

                if y_indice_izq < left_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and \
                   y_medio_izq < left_hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and not gesto_capturado:
                    posiciones_dibujadas.append({
                        'figura': figura_geometrica,
                        'x': x_indice,
                        'y': y_indice,
                        'tamaño': tamaño_figura,
                        'color': color_figura
                    })
                    gesto_capturado = True  # evita varios registros del mismo gesto
                    figura_geometrica = None  # reinicia la selección de figura
                    tamaño_figura = 0  

                if y_indice_izq > left_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y:
                    gesto_capturado = False  # Reinicia bandera

        # Fusionar dibujo auxiliar con la imagen
        imAuxGray = cv2.cvtColor(imAux, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(imAuxGray, 10, 255, cv2.THRESH_BINARY)
        thInv = cv2.bitwise_not(th)
        frame = cv2.bitwise_and(frame, frame, mask=thInv)
        frame = cv2.add(frame, imAux)

        cv2.imshow('frame', frame)
        cv2.imshow('imAux', imAux)

        if cv2.waitKey(1) & 0xFF == 27:
            break

captura.release()
cv2.destroyAllWindows()
