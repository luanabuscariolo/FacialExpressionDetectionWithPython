import captura_fotos_bd
import treinamento_fotos_bd
import reconhecedor_face_expressao

def print_menu():
    print('1 -- Capturar imagem')
    print('2 -- Treinar algoritmo')
    print('3 -- Reconhecer face e expressão')
    print('4 -- Sair')

while(True):
    print_menu()
    try:
        option = int(input('Digite sua escolha: '))
    except:
        print('Caracter inválido, tente novamente.')
    if option == 1:
        captura_fotos_bd.capturaFotosBD()
    elif option == 2:
        treinamento_fotos_bd.treinamentoFotosBD()
    elif option == 3:
        reconhecedor_face_expressao.reconhecedorFaceExpressao()
    elif option == 4:
        quit(0)
    else:
        print('Válido somente números de 1 a 4, tente novamente.')
