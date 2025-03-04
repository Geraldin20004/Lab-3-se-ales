# LABORATORIO NÚMERO 3 DE SEÑALES
Para este laboratorio, nuestro objetivo fue simular un escenario tipo "cóctel", en el que varios micrófonos capturan el sonido ambiente mientras diferentes personas conversan. La meta principal era aislar la voz de una sola persona a partir de las grabaciones. 


## Configuración del sistema

Para la grabación de los audios, nos dirigimos al laboratorio insonorizado de la universidad con el fin de minimizar interferencias y asegurar condiciones controladas. Dentro del laboratorio, colocamos una silla en el centro de la sala como punto de referencia. A cada extremo de la silla, posicionamos dos dispositivos móviles (utilizados como micrófonos) con una distancia equidistante respecto al centro, asegurando una simetría en la captura del sonido.

Cada una de nosotras se ubicó a una distancia de nueve baldosas de la silla, enfrentándonos directamente. Esta disposición permitió una captura uniforme del sonido desde ambas posiciones, garantizando que las fuentes sonoras estuvieran alineadas y equidistantes respecto a los dispositivos de grabación.

Para sincronizar la captura del audio y minimizar la variabilidad en el inicio de la grabación, realizamos una cuenta regresiva hasta cinco antes de comenzar a hablar. Nos aseguramos de que la grabación estuviera activa antes de iniciar el experimento para evitar pérdidas de datos y garantizar la consistencia en la adquisición del sonido ambiente y la voz.

Esta configuración permitió un registro claro y estructurado de la propagación del sonido, con una distancia definida entre fuentes y micrófonos, asegurando la validez de los datos obtenidos.

```python
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.fftpack import fft
from scipy.signal import welch, correlate
from sklearn.decomposition import FastICA
import os
```
Despues de colocar las respectivas librerias se continuo cargando los audios y despues calculando su SNR:
```python
from google.colab import drive
drive.mount('/content/drive')

# Ruta de la carpeta en Google Drive
base_path = "/content/drive/My Drive/lab3"

# Nombres de archivos en la carpeta "lab3"
files = ["sofia.mp3", "geral.mp3", "ruido.mp3"]
file_paths = [os.path.join(base_path, f) for f in files]

def calculate_snr(signal, noise):
    signal_power = np.mean(signal ** 2)  # Potencia de la señal
    noise_power = np.mean(noise ** 2)  # Potencia del ruido
    return 10 * np.log10(signal_power / noise_power)  # SNR en dB
```

  ```python
signals = []
sr = None

for file in file_paths:
    signal, sr = librosa.load(file, sr=None)  # Carga el audio
    signals.append(signal)
```
El objetivo de las anteriores lineas de codigos es que se puedan cargar multiples archivos de audio y almacenarlo,sr es la frecuencia de muestreo (cuántas veces por segundo se midió el audio). Despues de esto se ajustan las señales al mismo tamaño ya que hay algunas que tienen algunos segundos de mas.
```python
max_length = max(len(sig) for sig in signals)  # Encuentra la señal más larga
signals = np.array([np.pad(sig, (0, max_length - len(sig))) for sig in signals])  # Rellena con ceros las señales más cortas
```
Si hay una señal que este corta se va rellenar con ceros para asegurar el mismo tiempo.Despues de esto se identifica el ruido y se calculo el SNR de la señales.

```python
ruido = signals[-1]  # Última señal es el ruido

snr_values = {}
for i in range(len(signals) - 1):  # Excluir el ruido
    snr_values[files[i]] = calculate_snr(signals[i], ruido)
    print(f"SNR de {files[i]}: {snr_values[files[i]]:.2f} dB")
```
  Por medio del codigo anterior se pudo calcular el SNR de cada audio donde nos dio los siguientes valores:
  
  ![image](https://github.com/user-attachments/assets/26243aac-a000-47bd-aaef-f0342531bce6)

Dado que el SNR obtenido fue adecuado, se procedió con la siguiente etapa de la guía, la cual se detalla a continuación. Se llevó a cabo un análisis temporal y espectral de las señales captadas por cada micrófono, permitiendo identificar las características principales de cada fuente sonora. Para el análisis espectral, se empleó la Transformada Rápida de Fourier (FFT), facilitando la exploración de las frecuencias predominantes en cada señal.
A continuacion se va explicar el codigo realizado para que la guia fuera posible y se va explicar para que sirve cada linea de codigo:
```python
for i, signal in enumerate(signals):
    plt.figure(figsize=(8, 4))
    librosa.display.waveshow(signal, sr=sr)
    plt.title(f"Forma de onda de {files[i]}")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.show()
```
Esta parte del codigo nos permite graficar la onda de cada señal, permitiendo observar como varia en el tiempo.
![image](https://github.com/user-attachments/assets/3bcf451d-990b-422f-8d20-b2c63392e2af)
![image](https://github.com/user-attachments/assets/72b59d8b-ca77-4b5e-9e70-d161cfae8647)
![image](https://github.com/user-attachments/assets/32c6e2bc-1f9e-4622-b631-9aea4814e0d6)
```python
for i, signal in enumerate(signals):
    N = len(signal)
    T = 1.0 / sr
    freqs = np.fft.fftfreq(N, T)[:N//2]
    fft_vals = np.abs(fft(signal))[:N//2]
    
    plt.figure(figsize=(8, 4))
    plt.plot(freqs, fft_vals)
    plt.title(f"Espectro de {files[i]}")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud FFT")
    plt.show()
```
Con la anteriores lineas de codigo se va graficar el espectro de frecuencia:
![image](https://github.com/user-attachments/assets/3e16bfb0-ea19-4bf1-8165-1a26854437a3)
![image](https://github.com/user-attachments/assets/f8f9d596-9cfe-4f03-b3a0-084c5bd93890)
![image](https://github.com/user-attachments/assets/95608ce1-fc49-4c0b-b6e4-3a7e3c7f24c2)

Ahora, procedemos a graficar la forma de onda de cada señal
```python
for i, signal in enumerate(signals):
    plt.figure(figsize=(8, 4))
    librosa.display.waveshow(signal, sr=sr)
    plt.title(f"Forma de onda de {files[i]}")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.show()
```


-	Recorre todas las señales de audio en signals con for i, signal in enumerate(signals): i es el índice del archivo (0 = sofia.mp3, 1 = geral.mp3, 2 = ruido.mp3). signal es el contenido de audio del archivo.

-	plt.figure(figsize=(8, 4)): Crea una nueva figura con tamaño de 8 pulgadas de ancho y 4 de alto.  Si no usáramos esta línea, las gráficas se verían muy pequeñas.

-	librosa.display.waveshow(signal, sr=sr): Dibuja la forma de onda de la señal con la frecuencia de muestreo (sr). Esto genera una curva que representa los cambios de amplitud del sonido en el tiempo.


-	plt.title(f"Forma de onda de {files[i]}"): Agrega un título a la gráfica con el nombre del archivo (sofia.mp3, geral.mp3, etc.).

-	plt.xlabel("Tiempo (s)") y plt.ylabel("Amplitud"): Se añaden etiquetas a los ejes para entender mejor la gráfica. El eje X representa el tiempo (segundos). El eje Y representa la amplitud del sonido.


-	plt.show(): Muestra la gráfica en pantalla.


Luego, graficamos el espectro de frecuencia (FFT)


```python
for i, signal in enumerate(signals):
    N = len(signal)
    T = 1.0 / sr
    freqs = np.fft.fftfreq(N, T)[:N//2]
    fft_vals = np.abs(fft(signal))[:N//2]    
    plt.figure(figsize=(8, 4))
    plt.plot(freqs, fft_vals)
    plt.title(f"Espectro de {files[i]}")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud FFT")
    plt.show()
```

-	 len(signal): Se obtiene el tamaño de la señal con N donde, N indica cuántas muestras tiene el audio y cuanto mayor es N, más preciso será el análisis de frecuencia.


-	1.0 / sr: Se calcula el tiempo entre muestras con T, esto nos dice cada cuánto se tomó una muestra de la señal.

-	fft(signal): Se calcula la FFT de la señal, esto convierte la señal a su representación en frecuencia. Luego, se obtiene la magnitud absoluta de la FFT con np.abs(fft(signal)).

-	 [:N//2]: Se filtran solo las frecuencias positivas , La FFT produce frecuencias positivas y negativas. Como el audio es una señal real, la mitad negativa es un reflejo de la positiva.

-	plt.plot(freqs, fft_vals): Se grafica el espectro de frecuencia , Eje X (Frecuencia en Hz): Muestra qué frecuencias están en la señal. Eje Y (Magnitud de la FFT): Muestra cuánto contribuye cada frecuencia.


Procedemos a aplicar FastICA para separar las voces


```python
ica = FastICA(n_components=len(signals)-1, max_iter=500)   # Excluye el ruido
separated_sources = ica.fit_transform(signals[:-1].T).T  # Excluye la fila de ruido
```

-	FastICA(n_components=len(signals)-1, max_iter=500): Se define que queremos extraer n-1 componentes (porque el ruido no nos interesa). Se usa 500 iteraciones máximas para lograr una mejor separación.
  
-	ica.fit_transform(signals[:-1].T).T: 

                      •	signals[:-1] → Excluye la última señal (ruido).
                      •	.T → Transpone la matriz para que cada columna sea una fuente de audio.
                      •	ica.fit_transform(...) → Aplica ICA para separar las fuentes.
                      •	.T → Transpone de nuevo para obtener las señales separadas correctamente.

 
  Después graficamos  las señales separadas

  
```python
for i, source in enumerate(separated_sources):
    plt.figure(figsize=(8, 4))
    librosa.display.waveshow(source, sr=sr)
    plt.title(f"Forma de onda de la voz separada {i+1}")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.show()
```

-	for i, source in enumerate(separated_sources): i representa el índice de la señal separada. source es la señal separada correspondiente. Recorre todas las señales separadas para graficarlas una por una.

-	plt.figure(figsize=(8, 4)) : Crea una nueva figura con tamaño de 8x4 pulgadas para que la gráfica sea legible.


-	librosa.display.waveshow(source, sr=sr): Dibuja la forma de onda de la señal separada. sr=sr indica la frecuencia de muestreo para que el eje X esté en segundos.

-	plt.title(f"Forma de onda de la voz separada {i+1}"): Agrega un título con el número de la señal separada (Voz separada 1, Voz separada 2, etc.).
-	plt.xlabel("Tiempo (s)") y plt.ylabel("Amplitud"): Se agregan etiquetas en los ejes para entender la gráfica. Eje X: Representa el tiempo en segundos. Eje Y: Representa la amplitud de la señal (intensidad del sonido).

-	plt.show(): Muestra la gráfica en pantalla.

Por último, guardamos las voces separadas en Google Drive
```python
for i, source in enumerate(separated_sources):
    output_file = os.path.join(base_path, f"voz_separada_{i+1}.mp3")
    sf.write(output_file, source, sr)
    print(f"Se guardó la voz separada {i+1} en: {output_file}")
```
-	for i, source in enumerate(separated_sources): Recorrer las señales separadas 
                    •	i → Índice de la voz separada (0 para la primera voz, 1 para la segunda voz, etc.).
                    •	source → Contiene la señal de audio separada por ICA.
                    •	Con este bucle, guardamos cada voz en un archivo diferente.

-	os.path.join(base_path, f"voz_separada_{i+1}.mp3"): Definir la ruta del archivo
                    •	 base_path → Ruta de la carpeta en Google Drive (lab3)
                    •	voz_separada_{i+1}.mp3 → Nombre del archivo ("voz_separada_1.mp3", "voz_separada_2.mp3", etc.).

-	sf.write(output_file, source, sr): Guardar el archivo 
                    •	sf.write() → Función de soundfile para guardar archivos de audio.
                    •	output_file → Ruta donde se guardará el archivo (en Drive).
                    •	source → Señal de audio separada.
                    •	sr → Frecuencia de muestreo, para que el audio se reproduzca a la velocidad correcta.
                    •	El resultado es que la voz separada se guarda en formato MP3 en Drive.
-	print(f"Se guardó la voz separada {i+1} en: {output_file}"): Mostrar un mensaje de confirmación, esto imprime en pantalla la ruta del archivo guardado. Así puedes verificar que el archivo se guardó correctamente en Google Drive > lab3.

### Principales Métodos de Separación de Fuentes de Audio

Para aislar una señal de interés a partir de múltiples señales capturadas por micrófonos, existen diversas técnicas de separación de fuentes. Cada método tiene enfoques específicos según el contexto y la complejidad del entorno acústico.  

Análisis de Componentes Independientes (ICA)**  
Basado en estadísticas, este método busca separar señales mezcladas en fuentes independientes asumiendo que son estadísticamente distintas. Utiliza descomposición matemática para maximizar la independencia entre las señales, siendo útil en la separación de voz en entornos ruidosos y el análisis de EEG.  

-Beamforming  
Esta técnica emplea una matriz de micrófonos para enfocar la captación de sonido en una dirección específica, reduciendo el ruido de otras fuentes. Es ampliamente utilizado en asistentes de voz, videoconferencias y audífonos inteligentes.  

-Redes Neuronales (Deep Learning) 
Métodos avanzados como *Deep Clustering* y *TasNet* permiten a las redes neuronales aprender representaciones de audio y separarlas con precisión, incluso en entornos ruidosos. Se aplican en asistentes virtuales y restauración de grabaciones antiguas.  

-Factorización Matricial No Negativa (NMF)
Descompone las señales en componentes base y pesos asociados, facilitando la identificación de patrones recurrentes. Se usa en la separación de señales musicales y en la mejora del habla en ambientes ruidosos.  

-Transformada de Fourier de Tiempo Corto (STFT) 
Convierte las señales al dominio de la frecuencia para aplicar enmascaramiento espectral, filtrando el ruido y resaltando la señal deseada. Es común en la reducción de ruido y la mejora de calidad en señales de audio.  




