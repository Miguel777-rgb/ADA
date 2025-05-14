import random

def generar_ubicaciones(num_ubicaciones, nombre_archivo="1_million_location.txt"):
    """
    Genera un archivo de texto con coordenadas de latitud y longitud aleatorias.

    Args:
        num_ubicaciones (int): El número de ubicaciones a generar.
        nombre_archivo (str, opcional): El nombre del archivo a crear.
            Por defecto es "Xnumber_location.txt".
    """
    try:
        with open(nombre_archivo, 'w') as archivo:
            for _ in range(num_ubicaciones):
                # Genera latitud y longitud aleatorias dentro de rangos razonables
                latitud = random.uniform(-84, -82)  # Valores de latitud entre -84 y -82
                longitud = random.uniform(134, 136) # Valores de longitud 134 y 136
                # Escribe la coordenada en el archivo con el formato especificado
                archivo.write(f"{latitud},{longitud}\n")
        print(f"Se ha generado el archivo '{nombre_archivo}' con {num_ubicaciones} ubicaciones.")
    except Exception as e:
        print(f"Ocurrió un error al generar el archivo: {e}")

def generar_conexiones(num_conexiones, nombre_archivo="1_million_user.txt"):
    """
    Genera un archivo de texto con conexiones de aristas aleatorias, simulando usuarios
    y sus interacciones con ubicaciones.

    Args:
        num_conexiones (int): El número de conexiones de aristas a generar (filas en el archivo).
        nombre_archivo (str, opcional): El nombre del archivo a crear.
            Por defecto es "Xnumber_user.txt".
    """
    try:
        with open(nombre_archivo, 'w') as archivo:
            for i in range(num_conexiones):
                # Genera un número aleatorio de ubicaciones visitadas por el usuario
                num_ubicaciones_visitadas = random.randint(0, 100)  # Cada usuario visita entre 1 y 20 ubicaciones
                # Genera una lista de ubicaciones visitadas (índices)
                ubicaciones_visitadas = [random.randint(1, 1000700) for _ in range(num_ubicaciones_visitadas)]  # Asume que tienes 10,000 ubicaciones
                # Escribe las conexiones en el archivo con el formato especificado
                archivo.write(f"{i},{','.join(map(str, ubicaciones_visitadas))}\n")
        print(f"Se ha generado el archivo '{nombre_archivo}' con {num_conexiones} conexiones.")
    except Exception as e:
        print(f"Ocurrió un error al generar el archivo: {e}")

if __name__ == "__main__":
    # Genera 10,000 ubicaciones en el archivo "10_thousand_location.txt"
    generar_ubicaciones(1000000)
    generar_conexiones(1000000) 
    #generar_ubicaciones(10, "test_location.txt") #para pruebas
