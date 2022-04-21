import cv2
import numpy as np
import keyboard
import os
import alunos_repositorio

capturar = False
fechar = False


def podeCapturar(valor):
    global capturar
    capturar = valor


def podeFechar(valor):
    global fechar
    fechar = valor

def capturaFotosBD():
    global capturar
    global fechar

    fechar = False

    keyboard.on_press_key("q", lambda _: podeCapturar(True))
    keyboard.on_press_key("x", lambda _: podeFechar(True))

    if not os.path.exists('DeteccaoFaceExpressao/fotos'):
        os.makedirs('DeteccaoFaceExpressao/fotos')

    classificador = cv2.CascadeClassifier("haarcascade-frontalface-default.xml")
    classificadorOlho = cv2.CascadeClassifier("haarcascade-eye.xml")
    camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    amostra = 1
    numeroAmostras = 25

    alunos = alunos_repositorio.list()
    print('Id' + '\t' + 'Nome')
    for aluno in alunos:
        print(str(aluno[0])+'\t'+aluno[1])

    id = input('Digite seu identificador: ')
    largura, altura = 220, 220

    existe = False

    for aluno in alunos:
        if str(aluno[0]) == id:
            existe = True
    if existe == False:
        print('Id invalido. ')

    while (existe):
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
                        cv2.putText(imagem, "Imagem " + str(amostra) + "/" + str(numeroAmostras) + " capturada", (x, y - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 0, 255), thickness=2)
                        print("[foto " + str(amostra) + " capturada com sucesso]")
                        cv2.imshow("Face", imagem)
                        cv2.waitKey(2000)
                        amostra += 1
                    capturar = False

        cv2.putText(imagem, "Para capturar foto aperte Q", org=(30, 410), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255, 0, 0), thickness=2)
        cv2.putText(imagem, "Para fechar aperte X", org=(30, 440), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255, 0, 0), thickness=2)

        cv2.imshow("Face", imagem)

        if (amostra >= numeroAmostras + 1):
            break

        if fechar == True:
            cv2.destroyAllWindows()
            keyboard.unhook_all();
            break

    camera.release()
    cv2.destroyAllWindows()

#capturaFotosBD()