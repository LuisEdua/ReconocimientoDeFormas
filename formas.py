import cv2

color = (0, 0, 0)
color_texto = (255, 255, 255)
# Cargamos la figura
#image = cv2.imread('./FIGURAS1.jpg')
image = cv2.imread('./FIGURAS1.jpg')
# Cambiamos la escala de color a imagen en escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Dibujo de los bordes detectando los cambios en lospg bordes
# ((image,limiteMin, limiteMax)
canny = cv2.Canny(image, 50, 100)
# Suavizar los bordes dilantándolos
# Se dilata la imagen 12 veces para evitar curvas o lineas no deseadas
canny2 = cv2.dilate(canny, None, iterations=12)
# Suavizar los bordes erosionándolos
# Se erosionan los bordes 10 veces para evitar errores en la detección de los contornos
canny3 = cv2.erode(canny2, None, iterations=10)
# Se detectan los contornos haciendo conteo de los lados usando el borde externo de la imagen
# RETR_EXTERNAL, RETR_LIST, RETR_COMP y RETR_TREE
cnts, _ = cv2.findContours(canny3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # OpenCV 4
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.imshow('image', canny)
#cv2.imshow('Canny2', canny2)
#cv2.imshow('Canny3', canny3)
cv2.waitKey(0)
# Aplicamos un for para detectar el número de lados de cada contorno
for c in cnts:
    # Se calcula un porcentaje de error en la longitud de los arcos detectados en cada grupo de contornos
    # Calculando el error sobre el perímetro de la figura
    # Se pone el error en 0.0089 para que detecte correctamente las lineas
    epsilon = 0.0089 * cv2.arcLength(c, True)
    # Aproximación del contorno a un polígono de contorno cerrado
    approx = cv2.approxPolyDP(c, epsilon, True)
    # Se obtiene un rectángulo que encierra al contorno detectado
    x, y, w, h = cv2.boundingRect(approx)
    # Comparación para el número de lados detectado
    if len(approx) == 1:
        cv2.putText(image, 'Linea', (x, y - 5), 1, 1, color, 1)
    if len(approx) == 3:
        cv2.putText(image, 'Triangulo', (x, y - 5), 1, 1, color, 1)
    if len(approx) == 4:
        proporcion = float(w) / h
        print('aspect_ratio= ', proporcion)
        if proporcion > 0.85 and proporcion < 1.25:
            cv2.putText(image, 'Cuadrado', (x, y - 5), 1, 1, color, 1)
        else:
            cv2.putText(image, 'Rectangulo', (x, y - 5), 1, 1, color, 1)
    if len(approx) == 5:
        cv2.putText(image, 'Pentagono', (x, y - 5), 1, 1, color, 1)
    if len(approx) == 6:
        cv2.putText(image, 'Hexagono', (x, y - 5), 1, 1, color, 1)
    # Para detección de círculos se realiza mediante la definición de un circulo
    # como un polígono de muchos lados
    if len(approx) > 10:
        cv2.putText(image, 'Circulo', (x, y - 5), 1, 1, color, 1)
    cv2.drawContours(image, [approx], 0, color, 2)
# Se muestra la imagen con los contornos detectados
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
