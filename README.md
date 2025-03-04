# LABORATORIO NÚMERO 3 DE SEÑALES
Para este laboratorio, nuestro objetivo fue simular un escenario tipo "cóctel", en el que varios micrófonos capturan el sonido ambiente mientras diferentes personas conversan. La meta principal era aislar la voz de una sola persona a partir de las grabaciones.  

Para lograrlo, primero realizamos grabaciones de aproximadamente 30 segundos con nuestro compañero. Luego, extrajimos la sección inicial de ruido y la utilizamos como un audio independiente de referencia. A continuación, calculamos la relación señal/ruido (SNR) de cada grabación. Al confirmar que el SNR era adecuado, procedimos con el procesamiento y filtrado de las señales para obtener la voz deseada, como sabemos para programar se utlizan librerias que son de gran ayuda para que el codigo funcione que en este caso fueron las siguientes:

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
    
  Por medio del codigo anterior se pudo calcular el SNR de cada audio donde nos dio los siguientes valores:
![image](https://github.com/user-attachments/assets/ebc1fd66-c55b-40d2-bec5-329535ff2a30)
![image](https://github.com/user-attachments/assets/c8f10ea4-a8db-4c9d-95c5-45efbc193324)

Dado que el SNR obtenido fue adecuado, se procedió con la siguiente etapa de la guía, la cual se detalla a continuación. Se llevó a cabo un análisis temporal y espectral de las señales captadas por cada micrófono, permitiendo identificar las características principales de cada fuente sonora. Para el análisis espectral, se empleó la Transformada Rápida de Fourier (FFT), facilitando la exploración de las frecuencias predominantes en cada señal.
A continuacion se va explicar el codigo realizado para que la guia fuera posible y se va explicar para que sirve cada linea de codigo:

