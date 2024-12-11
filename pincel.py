'''
    El proposito que tengo para este programa es:
    Tener opciones de colores, en forma de botones de colores, y cuando
    se seleconen con el dedo indice se podra dibujar con ese mismo color, 
    existira la opcion de borrador y ptra de limpiar pantalla, asi como el cambiar el
    tamaño del pincel 
'''
import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Capturar video con cv2
captura = cv2.VideoCapture(1)
captura.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Asignación de colores en BGR
color_Blanco = (255, 255, 255)
color_Rojo = (2, 2, 140)
color_Naranja = (0, 120, 255)
color_Amarillo = (11, 203, 241)
color_Verde = (132, 255, 182)
color_Azul = (255, 229, 43)
color_Violeta = (178, 83, 142)
color_Rosa = (212, 0, 255)

color_inicial = color_Blanco

# Diccionario con las posiciones en la pantalla de los colores
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

limpiar_pantalla = {"posicion": (340, 6, 380, 40), "label": "limpiar", "color": (177, 77, 255)}
borrador = {"posicion": (430, 6, 470, 40), "label": "borrador", "color": (255, 255, 255)}
seleccion_del_grosor = {"posicion": (490, 6, 600, 50), "niveles": 20}

x1 = None
y1 = None
imAux = None
dibujando = False
grosor_inicial = 3
modo_borrador = False

# Pantalla completa
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as hands:
    while True:
        ret, frame = captura.read()
        if ret == False: break
        # espejo
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        if imAux is None: imAux = np.zeros(frame.shape, dtype=np.uint8)
        
        # mostrar colores disponibles
        for nombre, color_informacion in seleccion_colores.items():
            x1_minicaja, y1_minicaja, x2_minicaja, y2_minicaja = color_informacion["posicion"]
            cv2.rectangle(frame, (x1_minicaja, y1_minicaja), (x2_minicaja, y2_minicaja), color_informacion["color"], color_informacion["relleno"])
        # mostrar botn de borrador
        x1_borrador, y1_borrador, x2_borrador, y2_borrador = borrador["posicion"]
        cv2.rectangle(frame, (x1_borrador, y1_borrador), (x2_borrador, y2_borrador), borrador["color"], 2)
        cv2.putText(frame, borrador["label"], (x1_borrador + 5, y1_borrador + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, borrador["color"], 2)
        # mostrar boton limpiar pantalla
        x1_clear, y1_clear, x2_clear, y2_clear = limpiar_pantalla["posicion"]
        cv2.rectangle(frame, (x1_clear, y1_clear), (x2_clear, y2_clear), limpiar_pantalla["color"], 2)
        cv2.putText(frame, limpiar_pantalla["label"], (x1_clear + 10, y1_clear + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, limpiar_pantalla["color"], 2)
        #mostrar barra para seleccionar el grosor del pincel
        x1_bar, y1_bar, x2_bar, y2_bar = seleccion_del_grosor["posicion"]
        cv2.rectangle(frame, (x1_bar, y1_bar), (x2_bar, y2_bar), (200, 200, 200), 2)
        cv2.putText(frame, f'Grosor: {grosor_inicial}', (x1_bar, y2_bar + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # Procesar manos
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
                    color_inicial = (0, 0, 0)  # Color negro para el borrador
                else:
                    modo_borrador = False

                # Detectar si la mano izquierda está abierta (fin de dibujo)
                dedos_abiertos = sum(
                    hand_landmarks.landmark[i].y < hand_landmarks.landmark[i - 2].y
                    for i in [8, 12, 16, 20]
                )
                if etiqueta_mano == "Left" and dedos_abiertos == 4:
                    mano_izquierda_abierta = True

                # cambiar color si el dedo índice está sobre uno de los cuadros de color
                for nombre, color_informacion in seleccion_colores.items():
                    x1_minicaja, y1_minicaja, x2_minicaja, y2_minicaja = color_informacion["posicion"]
                    if x1_minicaja <= x_indice <= x2_minicaja and y1_minicaja <= y_indice <= y2_minicaja:
                        color_inicial = color_informacion["color"]
                        # Resaltar el cuadro seleccionado
                        for key in seleccion_colores.keys():
                            seleccion_colores[key]["relleno"] = 2
                        seleccion_colores[nombre]["relleno"] = -1
                # limpiar toda la pantalla al seleccionar el botón
                if x1_clear <= x_indice <= x2_clear and y1_clear <= y_indice <= y2_clear:
                    imAux = np.zeros(frame.shape, dtype=np.uint8)

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

