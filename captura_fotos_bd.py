import cv2
import numpy as np
import keyboard
import os

capturar = False

keyboard.on_press_key("q", lambda _: podeCapturar(True))

def podeCapturar(valor):
    global capturar
    capturar = valor

def capturaFotosBD():
    global capturar

    if not os.path.exists('DeteccaoFaceExpressao/fotos'):
        os.makedirs('DeteccaoFaceExpressao/fotos')

    classificador = cv2.CascadeClassifier("haarcascade-frontalface-default.xml")
    classificadorOlho = cv2.CascadeClassifier("haarcascade-eye.xml")
    camera = cv2.VideoCapture(1)
    amostra = 1
    numeroAmostras = 25
    id = input('Digite seu identificador: ')
    largura, altura = 220, 220

    while (True):
        conectado, imagem = camera.read()
        imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        facesDetectadas = classificador.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(150,150))

        for (x, y, l, a) in facesDetectadas:
            cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
            regiao = imagem[y:y + a, x:x + l]
            regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
            olhosDetectados = classificadorOlho.detectMultiScale(regiaoCinzaOlho)
            for (ox, oy, ol, oa) in olhosDetectados:
                cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (0, 255, 0), 2 )
                cv2.waitKey(1)
                if  capturar == True:
                    if np.average(imagemCinza) > 110:
                        imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
                        cv2.imwrite("DeteccaoFaceExpressao/fotos/pessoa." + str(id) + "." + str(amostra) + ".jpg", imagemFace)
                        print("[foto " + str(amostra) + " capturada com sucesso]")
                        amostra += 1
                    capturar = False

        cv2.imshow("Face", imagem)
        if (amostra >= numeroAmostras + 1):
            break

    print("Faces capturadas com sucesso")

    camera.release()
    cv2.destroyAllWindows()

#capturaFotosBD()