import captura_fotos_bd
import treinamento_fotos_bd
import reconhecedor_face_expressao
import alunos_repositorio
import cadastro
import os

def print_menu():
    print('1 -- Cadastrar aluno')
    print('2 -- Capturar imagem')
    print('3 -- Treinar algoritmo')
    print('4 -- Reconhecer face e expressão')
    print('5 -- Sair')

alunos_repositorio.createTable()

while(True):
    print_menu()
    try:
        option = int(input('Digite sua escolha: '))
    except:
        print('Caracter inválido, tente novamente.')
    if option == 1:
        cadastro.cadastrarAluno()
    elif option == 2:
        captura_fotos_bd.capturaFotosBD()
    elif option == 3:
        treinamento_fotos_bd.treinamentoFotosBD()
    elif option == 4:
        reconhecedor_face_expressao.reconhecedorFaceExpressao()
    elif option == 5:
        quit(0)
    else:
        print('Válido somente números de 1 a 5, tente novamente.')

    input("Pressione ENTER para continuar")
    os.system('cls' if os.name == 'nt' else 'clear')

