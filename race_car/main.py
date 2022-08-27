from ast import While
from playground import Playground

playground = Playground(10, 15, 4)
for i in range(10):
    array = playground.runaway
    print(array)
    print('------------------------')
    playground.generate_new_runaway()