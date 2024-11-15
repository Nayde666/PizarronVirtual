'''
El proposito que tengo para este programa es:
    Tener opciones de colores, en forma de botones de colores, y cuando
    se seleconen con el dedo pulgar se podra dibujar con ese mismo color, 
    existira la opcion de borrador y ptra de limpiar pantalla, asi como el cambiar el
    tamaño del pincel 
'''
import cv2
import numpy as np 
import mediapipe as mp 
# -------------------------------------------------------------------------------------------------------------
mp_drawing = mp.solutions.drawing_utils # 21 puntos
mp_hands = mp.solutions.hands # Implementación de mediapipe handaa

# capturar video con cv2
captura = cv2.VideoCapture(1) # 0 camara de la lap, 1 camara del celular
captura.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# ASignacion de colores en BGR recordar que ese ofrmato lo manjea opencv
color_Blanco = (255,255,255)
color_Rojo = (2,2,140)
color_Naranja = (0,120,255)
color_Amarillo = (11,203,241)
color_Verde = (132,255,182)
color_Azul = (255,229,43)
color_Violeta = (178,83,142)
color_Rosa = (212,0,255)

color_inicial = color_Blanco
# -------------------------------------------------------------------------------------------------------------------
# diccionario con las posiciones en la pantalla de los colores(x,y,w,h)
seleccion_colores = {
    "blanco" : {"color": color_Blanco, "posicion":(10, 6, 40, 40), "relleno": -1},
    "rojo" : {"color": color_Rojo, "posicion":(50, 6, 80, 40), "relleno": -1},
    "naranja" : {"color": color_Naranja, "posicion":(90, 6, 120, 40), "relleno": -1},
    "amarillo" : {"color": color_Amarillo, "posicion":(130, 6, 160, 40), "relleno": -1},
    "verde" : {"color": color_Verde, "posicion":(170, 6, 200, 40), "relleno": -1},
    "azul" : {"color": color_Azul, "posicion":(210, 6, 240, 40), "relleno": -1},
    "violeta" : {"color": color_Violeta, "posicion":(250, 6, 280, 40), "relleno": -1},
    "rosa" : {"color": color_Rosa, "posicion":(290, 6, 320, 40), "relleno": -1},
}
limpiar_pantalla = {"posicion": (340,6,380,40), "label": "limpiar", "color": (177,77,255)} #boton limpiar pantalla completa
seleccion_del_grosor ={"posicion": (490,6,600,50), "niveles":20} # seleccionar grosor apara lapiz
borrador = {"posicion": (430, 6, 470, 40), "label": "borrador", "color": (255, 255, 255)}

#--------------------------------------------------------------------------------------------------------------------
x1 = None
y1 = None
imAux = None 

dibujando = False
grosor_inicial = 3
#---------------------------------------------------------------------------------------------------------------------
#pamtalla completa
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#---------------------------------------------------------------------------------------------------------------------
# Configuracion mediapope
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2, #(detectarambas manos)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5 # Estos ultimos 2, son valores recomendados por defecto
) as hands:
    while True:
        ret, frame = captura.read()
        if ret == False: break
        # efecto espejo
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        if imAux is None: imAux = np.zeros(frame.shape, dtype=np.uint8) # Matriz de 0s del mismo tamaño que grame
        # ---------------------------------------------------SECCIÓN SUPERIOR--------------------------------------------------------------
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
        # -----------------------------------------------------------------------------------------------------------------------------------
        # Cambiar de gbr a rgb porque las detecciones se hacen en rgb
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = hands.process(frame_rgb)

        if resultados.multi_hand_landmarks is not None:
            #------------------------------------------
            # dibujando los puntos y sus conexiones con mediapipe
            for hand_landmarks in resultados.multi_hand_landmarks:
                #print(hand_landmarks)
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    # cambiar colores
                    #mp_drawing.DrawingSpec(color=(255,255,0), thickness=4, circle_radius=5),
                    #mp_drawing.DrawingSpec(color=(255,255,0), thickness=4)
                )
                # definir coordenadas del dedo indice y meñique (estos están a prueba)
                # Coordenadas del dedo índice y meñique
                x_indice = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width)
                y_indice = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)
                x_menique = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * width)
                y_menique = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * height)

                #modo borrador
                if x1_borrador <= x_indice <= x2_borrador and y1_borrador <= y_indice <= y2_borrador:
                    modo_borrador = True
                    color_inicial = (0, 0, 0)  # Color negro para el borrador
                else:
                    modo_borrador = False


                # cambiar color si el dedo índice esta sobre uno de los cuadros de color
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
                # seleccionar el grosor del pincel al mover el dedo indice sobre la barra
                if x1_bar <= x_indice <= x2_bar and y1_bar <= y_indice <= y2_bar:
                    grosor_inicial = int((x_indice - x1_bar) * seleccion_del_grosor["niveles"] / (x2_bar - x1_bar))
                    grosor_inicial = max(1, min(grosor_inicial, seleccion_del_grosor["niveles"]))
                #-----------------------------------------------------------------------------------------------------------
                # si el indice está levantado (activar dibujo) y en elcaso en el que el meñique  esté levantado(detener dibujo)
                if not dibujando and y_indice < y_menique:
                    dibujando = True
                    x1, y1 = x_indice, y_indice
                elif dibujando and y_menique < y_indice:
                    dibujando = False
                    x1, y1 = None, None  # Reiniciar coordenadas

                 # dibujar o borrar
                if dibujando and x1 is not None:
                    if modo_borrador:
                        cv2.line(imAux, (x1, y1), (x_indice, y_indice), (0, 0, 0), grosor_inicial)
                    else:
                        cv2.line(imAux, (x1, y1), (x_indice, y_indice), color_inicial, grosor_inicial)
                    x1, y1 = x_indice, y_indice
                #---------------------------------------------------------------------------------------------------------------
                # Dibujar si el dedo indice está levantado
                if dibujando and x1 is not None:
                    imAux = cv2.line(imAux, (x1, y1), (x_indice, y_indice), color_inicial, grosor_inicial)
                    x1, y1 = x_indice, y_indice
        
        # se fusiona el el dibujo auxiliar que es la matriz de 0s con el frame
        imAuxGray = cv2.cvtColor(imAux, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(imAuxGray, 10, 255, cv2.THRESH_BINARY)
        thInv = cv2.bitwise_not(th)
        frame = cv2.bitwise_and(frame, frame, mask=thInv)
        frame = cv2.add(frame, imAux)
        
        # mostrar ambos resulatdos
        cv2.imshow('frame', frame)
        cv2.imshow('imAux',imAux)
        # esc
        k = cv2.waitKey(1)
        if k == 27:
            break
captura.release()
cv2.destroyAllWindows()
