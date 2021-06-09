# Aprendizaje por refuerzo: Fundamentos teóricos del algoritmo AlphaZero e implementación
Repositorio con los códigos de mi TFG de Informática del Doble Grado en Ingeniería Informática - Matemáticas: <i>Aprendizaje por refuerzo: Fundamentos teóricos del algoritmo AlphaZero e implementación</i>

Se incluyen los códigos donde se implementa el algoritmo para el tres en raya (AlphaZero) y para el Conecta 4 (AlphaZero y AlphaZero Q)

Entre el tres en raya y el Conecta 4, la mayor diferencia está en el archivo `board` (dado que el juego que implementan es diferente) y `MCTS`, donde se adapta el número de acciones posibles. El resto de archivos son muy similares.

Entre AlphaZero y AlphaZero Q la única diferencia reside en el cálculo del valor con el que se entrena la red neuronal, para lo cual se modifica el archivo `MCTS`. Además, en el archivo `driver` se modifica ligeramente la forma de entrenar, de acuerdo a lo indicado en la memoria. 
