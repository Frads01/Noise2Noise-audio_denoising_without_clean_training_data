import torch
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from matplotlib import colors, pyplot as plt
from numpy.fft import rfft, irfft

from scipy.io import wavfile

import torchaudio


# Imposta il backend audio come Soundfile per Windows e Sox per Linux
torchaudio.set_audio_backend("soundfile")


def ms(x):
    """
    Calcola il valore quadratico medio (Mean Square) di un array.

    Args:
        x (np.ndarray): Array di input.

    Returns:
        float: Valore quadratico medio dell'array.
    """
    return (np.abs(x)**2).mean()


def rms(x):
    """
    Calcola la radice del valore quadratico medio (Root Mean Square) di un array.

    Args:
        x (np.ndarray): Array di input.

    Returns:
        float: Radice del valore quadratico medio dell'array.
    """
    return np.sqrt(ms(x))


def normalise(y, power):
    """
    Normalizza la potenza di un segnale audio.

    Args:
        y (np.ndarray): Segnale audio da normalizzare.
        power (float): Potenza desiderata.

    Returns:
        np.ndarray: Segnale audio normalizzato.
    """
    # La potenza media di una Gaussiana con `mu=0` e `sigma=x` è x^2.
    return y * np.sqrt(power / ms(y))


def noise(N, color, power):
    """
    Generatore di rumore colorato.

    Args:
        N (int): Numero di campioni.
        color (str): Colore del rumore ('white', 'pink', 'blue', 'brown', 'violet').
        power (float): Potenza del rumore (power = std_dev^2).
        # https://en.wikipedia.org/wiki/Colors_of_noise

    Returns:
        np.ndarray: Array contenente il rumore generato.
    """
    noise_generators = {
        'white': white,
        'pink': pink,
        'blue': blue,
        'brown': brown,
        'violet': violet
    }
    return noise_generators[color](N, power)


def white(N, power):
    """
    Genera rumore bianco.

    Args:
        N (int): Numero di campioni.
        power (float): Potenza del rumore.

    Returns:
        np.ndarray: Array contenente il rumore bianco generato.
    """
    y = np.random.randn(N).astype(np.float32)
    return normalise(y, power)


def pink(N, power):
    """
    Genera rumore rosa.

    Args:
        N (int): Numero di campioni.
        power (float): Potenza del rumore.

    Returns:
        np.ndarray: Array contenente il rumore rosa generato.
    """
    orig_N = N
    # Poiché rfft->ifft produce output di lunghezza diversa a seconda che gli input siano di lunghezza pari o dispari
    N+=1
    x = np.random.randn(N).astype(np.float32)
    X = rfft(x) / N
    S = np.sqrt(np.arange(X.size)+1.)  # +1 per evitare la divisione per zero
    y = irfft(X/S).real[:orig_N]
    return normalise(y, power)


def blue(N, power):
    """
    Genera rumore blu.

    Args:
        N (int): Numero di campioni.
        power (float): Potenza del rumore.

    Returns:
        np.ndarray: Array contenente il rumore blu generato.
    """
    orig_N = N
    # Poiché rfft->ifft produce output di lunghezza diversa a seconda che gli input siano di lunghezza pari o dispari
    N+=1
    x = np.random.randn(N).astype(np.float32)
    X = rfft(x) / N
    S = np.sqrt(np.arange(X.size))  # Filtro
    y = irfft(X*S).real[:orig_N]
    return normalise(y, power)


def brown(N, power):
    """
    Genera rumore marrone (Browniano).

    Args:
        N (int): Numero di campioni.
        power (float): Potenza del rumore.

    Returns:
        np.ndarray: Array contenente il rumore marrone generato.
    """
    orig_N = N
    # Poiché rfft->ifft produce output di lunghezza diversa a seconda che gli input siano di lunghezza pari o dispari
    N+=1
    x = np.random.randn(N).astype(np.float32)
    X = rfft(x) / N
    S = np.arange(X.size)+1  # Filtro
    y = irfft(X/S).real[:orig_N]
    return normalise(y, power)


def violet(N, power):
    """
    Genera rumore viola.

    Args:
        N (int): Numero di campioni.
        power (float): Potenza del rumore.

    Returns:
        np.ndarray: Array contenente il rumore viola generato.
    """
    orig_N = N
    # Poiché rfft->ifft produce output di lunghezza diversa a seconda che gli input siano di lunghezza pari o dispari
    N+=1
    x = np.random.randn(N).astype(np.float32)
    X = rfft(x) / N
    S = np.arange(X.size)  # Filtro
    y = irfft(X*S).real[0:orig_N]
    return normalise(y, power)


def generate_colored_gaussian_noise(file_path='./sample_audio.wav', snr=10, color='white'):
    """
    Genera un segnale audio con rumore gaussiano colorato aggiunto. (Versione con torchaudio)

    Args:
        file_path (str): Percorso del file audio da caricare.
        snr (float): Rapporto segnale-rumore (SNR) desiderato in dB.
        color (str): Colore del rumore ('white', 'pink', 'blue', 'brown', 'violet').

    Returns:
        np.ndarray: Segnale audio con rumore aggiunto.
    """

    # Carica i dati audio in un array numpy 1D
    un_noised_file, _ = torchaudio.load(file_path)
    un_noised_file = un_noised_file.numpy()
    un_noised_file = np.reshape(un_noised_file, -1)

    # Crea un array di potenza audio
    un_noised_file_watts = un_noised_file ** 2

    # Crea un array audio in decibel
    un_noised_file_db = 10 * np.log10(un_noised_file_watts)

    # Calcola la potenza del segnale e converti in dB
    un_noised_file_avg_watts = np.mean(un_noised_file_watts)
    un_noised_file_avg_db = 10 * np.log10(un_noised_file_avg_watts)

    # Calcola la potenza del rumore
    added_noise_avg_db = un_noised_file_avg_db - snr
    added_noise_avg_watts = 10 ** (added_noise_avg_db / 10)

    # Genera un campione casuale di rumore gaussiano additivo
    added_noise = noise(len(un_noised_file), color, added_noise_avg_watts)

    # Aggiungi il rumore al segnale originale
    noised_audio = un_noised_file + added_noise

    return noised_audio


def mynoise(original,snr):
    """
    Genera un segnale audio con rumore gaussiano bianco aggiunto. (Versione custom)

    Args:
        original (np.ndarray): segnale audio originale
        snr (float): Rapporto segnale-rumore (SNR) desiderato in dB.

    Returns:
        np.ndarray: Segnale audio con rumore aggiunto.
    """
    N = np.random.randn(len(original)).astype(np.float32)
    numerator = sum(np.square(original.astype(np.float32)))
    denominator = sum(np.square(N))
    factor = 10**(snr/10.0)
    K = (numerator/(factor*denominator))**0.5
    noise = original + K*N
    return noise


def check_snr(reference, test):
    """
        Calcola l'SNR tra due segnali

        Args:
            reference(np.ndarray): segnale originale
            test(np.ndarray): segnale a cui è stato aggiunto rumore

        Returns:
            float: valore di SNR in dB
    """
    eps = 0.00001
    numerator = 0.0
    denominator = 0.0
    for i in range(len(reference)):
        numerator += reference[i]**2
        denominator += (reference[i] - test[i])**2
    numerator += eps
    denominator += eps
    return 10*np.log10(numerator/denominator)


def gen_colored_gaussian_noise(file_path='./sample_audio.wav', snr=10, color='white'):
    """
    Genera un segnale audio con rumore gaussiano colorato aggiunto (Versione con scipy.io.wavfile).

    Args:
        file_path (str): Percorso del file audio da caricare.
        snr (float): Rapporto segnale-rumore (SNR) desiderato in dB.
        color (str): Colore del rumore ('white', 'pink', 'blue', 'brown', 'violet').  Questa versione usa mynoise, quindi il colore è ignorato

    Returns:
        np.ndarray: Segnale audio con rumore aggiunto.
    """

    # Carica i dati audio in un array numpy 1D
    fs, un_noised_file = wavfile.read(file_path)
    noised_audio = mynoise(un_noised_file,snr)
    return noised_audio


def load_audio_file(file_path='./sample_audio.wav'):
    """
    Carica un file audio e restituisce la forma d'onda come array NumPy.

    Args:
        file_path (str): Percorso del file audio da caricare.

    Returns:
        np.ndarray: Forma d'onda del file audio.
    """
    fs, waveform = wavfile.read(file_path)
    return waveform


def save_audio_file(np_array=np.array([0.5]*1000),file_path='./sample_audio.wav', sample_rate=48000, bit_precision=16):
    """
    Salva un array NumPy come file audio WAV.

    Args:
        np_array (np.ndarray): Array NumPy contenente la forma d'onda.
        file_path (str): Percorso del file audio da salvare.
        sample_rate (int): Frequenza di campionamento (default: 48000).
        bit_precision (int):  Profondità di bit (default: 16). Il valore viene forzato a 16 in quanto wavfile accetta solo array a 16 bit

    """
    np_array = np_array.flatten()
    np_array = np_array.astype('int16')
    wavfile.write(file_path,sample_rate,np_array)