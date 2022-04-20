import alunos_repositorio

def cadastrarAluno():
    nomeAluno = input('Qual no nome do aluno?')
    idAluno = alunos_repositorio.insert(nomeAluno)
    print('O Id do(a) ' + nomeAluno + ' é ' + str(idAluno))
