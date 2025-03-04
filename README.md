# LABORATORIO NÚMERO 3 DE SEÑALES
Para este laboratorio, nuestro objetivo fue simular un escenario tipo "cóctel", en el que varios micrófonos capturan el sonido ambiente mientras diferentes personas conversan. La meta principal era aislar la voz de una sola persona a partir de las grabaciones.  

Para lograrlo, primero realizamos grabaciones de aproximadamente 30 segundos con nuestro compañero. Luego, extrajimos la sección inicial de ruido y la utilizamos como un audio independiente de referencia. A continuación, calculamos la relación señal/ruido (SNR) de cada grabación. Al confirmar que el SNR era adecuado, procedimos con el procesamiento y filtrado de las señales para obtener la voz deseada.

´´´python 
if len(file_names) < 2:
    print("⚠️ Debes subir al menos dos archivos: uno limpio y otro con ruido.")
else:
    # Cargar los dos audios
    audio_limpio, sr1 = librosa.load(file_names[0], sr=None)  # Audio original
    audio_ruidoso, sr2 = librosa.load(file_names[1], sr=None)  # Audio con ruido

    # Verificar que tengan la misma tasa de muestreo
    if sr1 != sr2:
        print("⚠️ Los audios deben tener la misma tasa de muestreo.")
    else:
        # Asegurar que tengan la misma longitud
        min_len = min(len(audio_limpio), len(audio_ruidoso))
        audio_limpio = audio_limpio[:min_len]
        audio_ruidoso = audio_ruidoso[:min_len]

        # Calcular el ruido (diferencia entre el audio ruidoso y el limpio)
        ruido = audio_ruidoso - audio_limpio

        # Calcular la potencia de la señal y del ruido
        potencia_señal = np.mean(audio_limpio**2)
        potencia_ruido = np.mean(ruido**2)

        # Evitar divisiones por cero
        if potencia_ruido == 0:
            snr = np.inf  # SNR infinito si no hay ruido
        else:
            # Calcular SNR en dB
            snr = 10 * np.log10(potencia_señal / potencia_ruido)

        # Mostrar resultados
        print(f"🔊 SNR calculado: {snr:.2f} dB")
    
    
