# ML Experiment Tracker (Local)

## Descripción del Proyecto

Este proyecto tiene como objetivo principal proporcionar una herramienta sencilla y eficiente para el seguimiento de experimentos de Machine Learning (ML) de forma **local**, sin la necesidad de configurar y mantener servidores adicionales. La idea es ofrecer a los investigadores y desarrolladores de ML una solución ligera para registrar métricas, parámetros, artefactos y resultados de sus experimentos directamente en su entorno de trabajo.

## Objetivo

El objetivo fundamental es crear un sistema de seguimiento de experimentos de ML que sea:

- **Local**: No requiere conexión a internet ni despliegue de servidores. Todos los datos se almacenan en el sistema de archivos local del usuario.
- **Fácil de usar**: Interfaz intuitiva y API sencilla para registrar y consultar experimentos.
- **Ligero**: Mínimas dependencias y bajo consumo de recursos.
- **Flexible**: Permite registrar una variedad de datos de experimentos (parámetros, métricas, modelos, conjuntos de datos, etc.).
- **Reproducible**: Facilita la recreación de experimentos al almacenar toda la información relevante.

## Características (Propuestas)

- **Registro de Parámetros**: Guarda los hiperparámetros utilizados en cada experimento.
- **Registro de Métricas**: Almacena métricas de rendimiento (accuracy, loss, F1-score, etc.) a lo largo del entrenamiento o al final del experimento.
- **Almacenamiento de Artefactos**: Permite guardar modelos entrenados, preprocesadores, visualizaciones, etc.
- **Etiquetado de Experimentos**: Organiza experimentos con etiquetas personalizadas.
- **Comparación de Experimentos**: Herramientas para comparar fácilmente los resultados de diferentes ejecuciones.
- **Interfaz de Usuario (Opcional)**: Una interfaz web local o de línea de comandos para visualizar y gestionar experimentos.
- **Integración con Librerías ML**: Posible integración con frameworks populares como Scikit-learn, TensorFlow, PyTorch.

## ¿Por qué un tracker local?

En muchos escenarios, especialmente durante las fases iniciales de investigación y desarrollo, la sobrecarga de configurar y mantener soluciones de seguimiento de experimentos basadas en la nube puede ser un obstáculo. Este proyecto busca eliminar esa barrera, permitiendo a los usuarios centrarse en sus experimentos sin preocuparse por la infraestructura. Es ideal para:
- Proyectos personales y de investigación.
- Desarrollo rápido de prototipos.
- Entornos con restricciones de red o seguridad.