Pregunta central: ¿cómo podemos identificar y cuantificar los estereotipos culturales presentes en un documento o corpus dado?

Usamos como ejemplo el estereotipo según el cual la ocupación de chef está más asociada a hombres que a mujeres. En este caso _chef_ es el término contexto y _mujer_ y _varón_ son los término target.

Si buscamos cuantificar estereotipos dados en un corps dado:

En primer lugar es necesario identificar los términos que representan al estereotipo -- es decir, debemos definir grupos de palabras cada componente del estereotipo (contexto y targets). Ejemplo:
  - Chef: singular, plural y sinónimos
  - Mujer: pronombres femeninos y nombres propios típicamente femeninos
  - Varón: pronombres masculinos y nombres propios típicamente masculinos
Una primera dificultad es la ausencia de un criterio taxativo para definir los grupos. No está claro si debemos incluir sinónimos o nombres propios, ni cuáles ni cuántos. Además no hay forma de definir el concepto exacto si el mismo tiene homónimos.
En el caso de análisis intertemporales, es probable que un mismo estereotipo se defina por distintos grupos de palabras a lo largo del tiempo porque el uso del lenguaje e incluso el significado de las palabras cambia. Construir los grupos en este contexto no es trivial.

En segundo lugar debemos definir una métrica de sesgo que tome como input los grupos de palabras definidos.

PPMI

(Hiperparámetros)

Métodos basados en _word embeddings_

(Hiperparámetros)

La métrica presentada por Garg se puede extender a más de dos grupos target.

Alternativamente podemos querer descubrir estereotipos significativos en lugar de medir estereotipos prefijados. En este caso: ...

### Comparación de métricas

No es posible comparar el nivel de métricas construidas con diferentes metodologías para determinar cuál halla mayor o menor nivel de estereotipo. Las unidades de medida no son comparables.

Sin embargo, en general sí es posible comparar a las métricas en términos del estereotipo que encuentran, porque en general determinan si el término contexto se halla más cerca de un target o del otro con respecto a un punto de equidistancia (el 0 tanto para el PPMI como para Garg).

Por otra parte, la comparación se puede facilitar con análisis longitudinales del sesgo -- métricas distintas se pueden comparar según las variaciones que presentan a lo largo del tiempo en lugar de los niveles.

### Evaluación de métricas

Para evaluar la bondad de la métrica de sesgo se puede:

1. Usar fuentes externas para medir el estereotipo en "el mundo real".
Por ejemplo, podemos medir la proporción de chefs que son varones y mujeres y comparar los resultados con el sesgo medido en el corpus. Un problema de este enfoque es que el estereotipo presente en el corpus no es necesariamente un reflejo fiel del estereotipo en otros ámbitos (laboral, social, etc.). Los textos pueden sobre o subrepresentar estas realidades pero esto no indica si la métrica es buena o no para cuantificar el sesgo. Otro problema es que los niveles de las métricas externas no siempre se pueden comparar con los niveles de las métricas de sesgo.
2. Analizar el grado en que cada documento del corpus contribuye a la métrica de sesgo global. Definiendo puntos de corte en el grado, se pueden definir al menos tres tipos de documentos: de alto impacto positivo, de alto impacto negativo y de impacto neutral sobre la métrica de sesgo. Si el sesgo detectado en los documentos de estos tres grupos coincide con el nivel de estereotipo que le asignan humanos, entonces hay evidencia a favor de que la métrica mide correctamente el estereotipo.

### Hipótesis

- En primer lugar definir un corpus. No necesariamente todos los documentos tienen el mismo peso "real" o "social" -- hay documentos que se leen significativamente más que otros. De esta manera, si bien el texto de dos documentos puede presentar un estereotipo de magnitud similar, el impacto verdadero en la representación cultural no es el mismo. Esto no es un defecto de las métricas de sesgo pero es un aspecto a tener en cuenta.

- La dependencia de los conjuntos de palabras prefijados es un problema imposible de sortear -- afecta a todas las métricas por igual.

Las métricas basadas en word embeddings:

- Son más susceptibles al tamaño del corpus (si el corpus es pequeño --¿cuánto?-- los resultados son más """ruidosos""").
- Son más susceptibles a la configuración de los hiperparámetros que PPMI
- Son más susceptibles a las palabras poco frecuentes que PPMI
- No captan fielmente el nivel de estereotipo de los documentos porque capturan coocurrencias de orden superior que son díficiles de medir analíticamente (los modelos son una "caja negra").

### Experimentos

...
