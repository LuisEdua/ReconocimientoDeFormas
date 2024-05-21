import cv2

color = (201, 55, 100)
# Cargamos la figura
camara = cv2.VideoCapture(0)
camara.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)


# Cambiamos la escala de color a imagen en escala de grises
def procesar(image):
    original = image
    # Dibujo de los bordes detectando los cambios en los bordes
    # ((image,limiteMin, limiteMax)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Aplicamos un filtro Gaussiano
    image = cv2.GaussianBlur(image, (51, 51), cv2.BORDER_DEFAULT)
    # image = cv2.medianBlur(image,5)
    image = cv2.bilateralFilter(image, 5, 5, 7)
    canny = cv2.Canny(image, 150, 200)
    # Suavizar los bordes dilantándolos
    canny = cv2.dilate(canny, None, iterations=3)
    # Suavizar los bordes erosionándolos
    # canny = cv2.erode(canny, None, iterations=1)
    # Se detectan los contornos haciendo conteo de los lados usando el borde externo de la imagen
    # RETR_EXTERNAL, RETR_LIST, RETR_COMP y RETR_TREE
    cnts, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # OpenCV 4
    # Aplicamos un for para detectar el número de lados de cada contorno
    for c in cnts:
        # Se calcula un porcentaje de error en la longitud de los arcos detectados en cada grupo de contornos
        # Calculando el error sobre el perímetro de la figura
        epsilon = 0.5 * cv2.arcLength(c, True)
        # Aproximación del contorno a un polígono de contorno cerrado
        approx = cv2.approxPolyDP(c, epsilon, True)
        # Se obtiene un rectángulo que encierra al contorno detectado
        x, y, w, h = cv2.boundingRect(approx)
        proporcion = float(w) / h
        if (w > 150 and h > 150) and (w < 600 and h < 600):
            # Comparación para el número de lados detectado
            if len(approx) == 4:
                print('Relación de aspecto= ', proporcion)
                if proporcion > 0.85 and proporcion < 1.25:
                    cv2.putText(original, 'Cuadrado', (x, y - 5), 1, 1, color, 1)
                    cv2.drawContours(original, [approx], 0, color, 2)
                else:
                    cv2.putText(original, 'Rectangulo', (x, y - 5), 1, 1, color, 1)
                    cv2.drawContours(original, [approx], 0, color, 2)
            # Para detección de círculos se realiza mediante la definición de un circulo
            # como un polígono de muchos lados
            if len(approx) > 6:
                cv2.putText(image, 'Circulo', (x, y - 5), 1, 1, color, 1)
                # cv2.drawContours(image, [approx], 0, color,2)
                cv2.drawContours(original, [approx], 0, color, 2)
            image = cv2.rectangle(original, (x, y), (x + w, y + h), (0, 0, 255), 2)
        else:
            image = original
        cv2.imshow('image', image)


while (camara.isOpened()):
    ret, image = camara.read()
    procesar(image)
    if cv2.waitKey(10) & 0XFF == 27:
        break
cv2.destroyAllWindows()
