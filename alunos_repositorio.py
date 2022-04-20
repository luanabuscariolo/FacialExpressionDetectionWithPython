import sqlite3

banco = sqlite3.connect('bd_alunos.db')
cursor = banco.cursor()

def createTable():
    try: cursor.execute("CREATE TABLE alunos ("
                   "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                   "nome TEXT)")
    except:
        print("Tabela já criada.")

def insert(nome):
    cursor.execute("INSERT INTO alunos (nome) VALUES ('"  + nome + "')")
    banco.commit()

def get(id):
    cursor.execute('''SELECT * FROM alunos where id = ''' + id)
    return cursor.fetchone()

def list():
    cursor.execute('''SELECT * FROM alunos''')
    return cursor.fetchall()
