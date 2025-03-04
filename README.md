# LABORATORIO NÚMERO 3 DE SEÑALES
Para este laboratorio, nuestro objetivo fue simular un escenario tipo "cóctel", en el que varios micrófonos capturan el sonido ambiente mientras diferentes personas conversan. La meta principal era aislar la voz de una sola persona a partir de las grabaciones.  

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



## Configuración del sistema

Para la grabación de los audios, nos dirigimos al laboratorio insonorizado de la universidad con el fin de minimizar interferencias y asegurar condiciones controladas. Dentro del laboratorio, colocamos una silla en el centro de la sala como punto de referencia. A cada extremo de la silla, posicionamos dos dispositivos móviles (utilizados como micrófonos) con una distancia equidistante respecto al centro, asegurando una simetría en la captura del sonido.

Cada una de nosotras se ubicó a una distancia de nueve baldosas de la silla, enfrentándonos directamente. Esta disposición permitió una captura uniforme del sonido desde ambas posiciones, garantizando que las fuentes sonoras estuvieran alineadas y equidistantes respecto a los dispositivos de grabación.

Para sincronizar la captura del audio y minimizar la variabilidad en el inicio de la grabación, realizamos una cuenta regresiva hasta cinco antes de comenzar a hablar. Nos aseguramos de que la grabación estuviera activa antes de iniciar el experimento para evitar pérdidas de datos y garantizar la consistencia en la adquisición del sonido ambiente y la voz.

Esta configuración permitió un registro claro y estructurado de la propagación del sonido, con una distancia definida entre fuentes y micrófonos, asegurando la validez de los datos obtenidos.
