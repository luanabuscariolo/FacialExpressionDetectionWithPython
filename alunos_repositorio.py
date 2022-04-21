import sqlite3

banco = None
cursor = None

def createTable():
    global banco
    global cursor
    banco = sqlite3.connect('bd_alunos.db')
    cursor = banco.cursor()
    try: cursor.execute("CREATE TABLE alunos ("
                   "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                   "nome TEXT)")
    except:
        print("Tabela já criada.")

def insert(nome):
    cursor.execute("INSERT INTO alunos (nome) VALUES ('"  + nome + "')")
    banco.commit()
    return cursor.lastrowid

def get(id):
    cursor.execute('''SELECT * FROM alunos where id = ''' + str(id))
    return cursor.fetchone()

def list():
    cursor.execute('''SELECT * FROM alunos''')
    return cursor.fetchall()

#print(get(1))