Pregunta central: ¿cómo podemos identificar y cuantificar los estereotipos culturales de un documento o corpus dado?

Usamos como ejemplo el siguiente estereotipo: la ocupación de chef está más asociada a hombres que a mujeres -- en este caso _chef_ es el término **contexto** y _mujer_ y _varón_ son los términos **target** A y B.

### Cuantificar estereotipos dados

Los pasos a seguir son:

1. Identificar los términos que representan al estereotipo -- es decir, debemos definir grupos de palabras para cada componente del estereotipo (contexto y targets). Ejemplo:
  - Contexto: singular, plural y sinónimos (chef, chefs, etc.)
  - Targets: pronombres y nombres propios típicos (él, ella, etc.)

Dificultades:

a) La ausencia de un criterio taxativo para definir los grupos. No está claro si debemos incluir sinónimos o nombres propios, ni cuáles ni cuántos. Además no hay forma de definir el concepto exacto si tiene homónimos.

b) En el caso de análisis intertemporales, es probable que un mismo estereotipo se defina por distintos grupos de palabras a lo largo del tiempo porque el uso del lenguaje e incluso el significado de las palabras cambia. Construir los grupos en este contexto no es trivial.

2. Definir una métrica de sesgo con nivel de medida de razón que use como input los grupos de palabras. El cero indica la ausencia de sesgo, y los valores positivos (negativos) indican la magnitud de sesgo hacia el target A (B).

#### PPMI

Hiperparámetros:

- Tamaño de la ventana
- Ponderación de la co-coocucrrencia según la distancia en la ventana (probablemente tiene impacto menor)

#### Métricas basadas en _embeddings_

Los hiperparámetros dependen de la metodología usada para computar los embeddings.

Ejemplo: Garg et al (2018)

_Nota: Garg se puede extender a más de dos grupos target (revisar)._

### Identificar estereotipos significativos

Alternativamente podemos querer descubrir estereotipos significativos en lugar de medir estereotipos prefijados. En este caso es necesario:

1. Predefinir contextos y targets potenciales, con sus grupos de palabras correspondientes.

Dificultades:

No se puede definir todos los contextos y targets posibles. Es necesario usar algún criterio arbitrario para acotar la búsqueda -- estereotipos reseñados previamente, por ejemplo.

2. Aplicar una métrica de sesgo a todas las combinaciones posibles de contexto-targets. Los valores "significativos" se catalogan como estereotipos.

Dificultades:

Aumenta la probabilidad de falsos positivos por comparaciones múltiples si usamos tests de hipótesis (?) -- se puede contrarrestar con alguna correción (?)

### Comparación de métricas

Dificultad:

Un ejercicio de interés comparar los resultados de métricas distintas para determinar cuál halla mayor o menor nivel de estereotipo. Sin embargo, no es posible comparar el nivel de métricas construidas con diferentes metodologías. Las unidades de medida en general no son comparables.

"Soluciones":
- En general las métricas determinan si el término contexto se halla más cerca de un target o del otro con respecto a un punto de equidistancia (el 0 tanto para el PPMI como para Garg). Esto por lo menos habilita las comparaciones en términos de la "dirección" y la presencia de estereotipo.
- Por otra parte, la comparación se puede facilitar con análisis longitudinales del sesgo -- métricas distintas se pueden comparar según las variaciones que presentan a lo largo del tiempo, en lugar de los niveles.

### Evaluación de métricas

Para evaluar la bondad de la métrica de sesgo se puede:

1. Usar fuentes externas para medir el estereotipo en "el mundo real".
Por ejemplo, podemos medir la proporción de chefs que son varones y mujeres y comparar los resultados con el sesgo medido en el corpus.

Dificultades:

a) Un problema de este enfoque es que el estereotipo presente en el corpus no es necesariamente un reflejo fiel del estereotipo en otros ámbitos (laboral, social, etc.). Los textos pueden sobre o subrepresentar estas realidades pero esto no indica si la métrica es buena o no para cuantificar el sesgo.

b) Otro problema es que las magnitudes de las métricas externas no siempre se pueden comparar con los niveles de las métricas de sesgo. Esto se puede resolver haciendo análisis a lo largo del tiempo o entre corpus distintos.

2. Analizar el grado en que cada documento del corpus contribuye a la métrica de sesgo global. Definiendo puntos de corte en el grado, se pueden determinar al menos tres tipos de documentos: de alto impacto positivo, de alto impacto negativo y de impacto neutral sobre la métrica de sesgo. Si el sesgo detectado en los documentos de estos tres grupos coincide con el nivel de estereotipo que le asignan humanos, entonces hay evidencia a favor de que la métrica mide correctamente el estereotipo.

### Observaciones / Hipótesis

1. No necesariamente todos los documentos tienen el mismo peso social "real" -- hay documentos que se leen significativamente más que otros. De esta manera, si bien el texto de dos documentos puede presentar un estereotipo de magnitud similar, el impacto verdadero en la representación cultural no es el mismo. Esto no es un defecto de las métricas pero es un aspecto a tener en cuenta.

2. La dependencia de los conjuntos de palabras prefijados es un problema imposible de sortear -- afecta a todas las métricas por igual.

3. Las métricas basadas en word embeddings:
  a. Son más susceptibles al tamaño del corpus que PPMI (si el corpus es pequeño los resultados son más """ruidosos""" -- ¿cuán pequeño? ¿qué es ruidoso?)
  b. Son más susceptibles a la configuración de los hiperparámetros que PPMI
  c. Son más susceptibles a las palabras poco frecuentes que PPMI
  d. No captan fielmente el nivel de estereotipo de los documentos porque capturan coocurrencias de orden superior que son díficiles de medir analíticamente (los modelos son una "caja negra"). Esto se puede acentuar si en el estereotipo el contexto combina múltiples dimensiones relevantes -- por ejemplo, en _chef_ tanto el concepto de _cocina_ como el de _jefe_ son relevantes.

### Experimentos

3d. Medir estereotipo que "combine dimensiones" a lo largo del tiempo con PPMI y Garg. Comparar los resultados entre sí y con métricas externas.

3d. Medir estereotipo que "combine dimensiones" con PPMI y Garg. Evaluar el impacto de los documentos en las métricas. Evaluar el estereotipo "real" de documentos que se detectaron como del alto sesgo negativo y positivo.

#### Estereotipo por documento

##### PPMI

  - El odds ratio no está definido si alguno de los dos targets no está en el documento
    - Posible solución:
      - Si está A y no B: return PPMI(A,C)
      - Si está B y no A: return -PPMI(B,C)
      - Si están ambos: return PPMI(A,C) - PPMI(B,C)
  - Aun si el OR está definido, el valor no refleja el aporte al estereotipo global del corpus (la frecuencia de las palabras puede ser pequeña)
    - ¿Poner un umbral de frecuencia? ¿Ponderar por frecuencia?





    - Dump wiki ingles
      - subcorpus

    - Experimento 1
      - Hipotesis: las buenas metricas de bias son insensibles a documentos que no tienen
      target

      - medir sesgo de género para "todas las palabras" (como table 2 de garg)
        - medir frecuencia de los terminos con mas sesgo (Hipotesis: garg trae palabras poco frecuentes)
        (nota: usar cutoff de frecuencia 20 para vocab --> de esta manera los PMI infinitos son "validos")
        - hipotesis: palabras poco frecuentes con pmi alto --> pero con CI se vería la no significancia

      - sacar documentos irrelevantes (no tienen targets) y evaluar impacto en metricas

    - ver que % de docs no tienen targets (he, she)
      - si son pocos, undersamplear
