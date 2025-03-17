import torch
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from matplotlib import colors, pyplot as plt
from numpy.fft import rfft, irfft

from scipy.io.wavfile import read

import torchaudio


# Set Audio backend as Soundfile for windows and Sox for Linux
torchaudio.set_audio_backend("soundfile")


def ms(x):
    """
    Calcola la potenza media quadratica (mean square) di un segnale.

    Args:
        x (array-like): Il segnale di input.

    Returns:
        float: La potenza media quadratica del segnale.
    """
    return (np.abs(x)**2).mean()


def rms(x):
    """
    Calcola la radice della potenza media quadratica (root mean square) di un segnale.

    Args:
        x (array-like): Il segnale di input.

    Returns:
        float: La radice della potenza media quadratica del segnale.
    """
    return np.sqrt(ms(x))


def normalise(y, power):
    """
    Normalizza un segnale in base alla potenza specificata.

    Args:
        y (array-like): Il segnale da normalizzare.
        power (float): La potenza desiderata.

    Returns:
        array-like: Il segnale normalizzato.
    """
    return y * np.sqrt(power / ms(y))


def noise(N, color, power):
    """
    Generatore di rumore colorato.

    Args:
        N (int): Il numero di campioni da generare.
        color (str): Il colore del rumore ('white', 'pink', 'blue', 'brown', 'violet').
        power (float): La potenza del rumore (varianza).

    Returns:
        array-like: Il segnale di rumore generato.

    Raises:
        KeyError: Se il colore specificato non è supportato.
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
        N (int): Il numero di campioni da generare.
        power (float): La potenza del rumore (varianza).

    Returns:
        array-like: Il segnale di rumore bianco generato.
    """
    y = np.random.randn(N).astype(np.float32)
    return normalise(y, power)


def pink(N, power):
    """
    Genera rumore rosa.

    Args:
        N (int): Il numero di campioni da generare.
        power (float): La potenza del rumore (varianza).

    Returns:
        array-like: Il segnale di rumore rosa generato.
    """
    orig_N = N
    # Aggiusta la lunghezza per la trasformata di Fourier reale
    N += 1
    x = np.random.randn(N).astype(np.float32)
    X = rfft(x) / N
    S = np.sqrt(np.arange(X.size) + 1.)  # +1 per evitare divisione per zero
    y = irfft(X / S).real[:orig_N]
    return normalise(y, power)


def blue(N, power):
    """
    Genera rumore blu.

    Args:
        N (int): Il numero di campioni da generare.
        power (float): La potenza del rumore (varianza).

    Returns:
        array-like: Il segnale di rumore blu generato.
    """
    orig_N = N
    # Aggiusta la lunghezza per la trasformata di Fourier reale
    N += 1
    x = np.random.randn(N).astype(np.float32)
    X = rfft(x) / N
    S = np.sqrt(np.arange(X.size))  # Filtro
    y = irfft(X * S).real[:orig_N]
    return normalise(y, power)


def brown(N, power):
    """
    Genera rumore browniano (rosso).

    Args:
        N (int): Il numero di campioni da generare.
        power (float): La potenza del rumore (varianza).

    Returns:
        array-like: Il segnale di rumore browniano generato.
    """
    orig_N = N
    # Aggiusta la lunghezza per la trasformata di Fourier reale
    N += 1
    x = np.random.randn(N).astype(np.float32)
    X = rfft(x) / N
    S = np.arange(X.size) + 1  # Filtro
    y = irfft(X / S).real[:orig_N]
    return normalise(y, power)


def violet(N, power):
    """
    Genera rumore viola.

    Args:
        N (int): Il numero di campioni da generare.
        power (float): La potenza del rumore (varianza).

    Returns:
        array-like: Il segnale di rumore viola generato.
    """
    orig_N = N
    # Aggiusta la lunghezza per la trasformata di Fourier reale
    N += 1
    x = np.random.randn(N).astype(np.float32)
    X = rfft(x) / N
    S = np.arange(X.size)  # Filtro
    y = irfft(X * S).real[0:orig_N]
    return normalise(y, power)


def generate_colored_gaussian_noise(file_path='./sample_audio.wav', snr=10, color='white'):
    """
    Genera un segnale audio con rumore gaussiano colorato aggiunto.

    Args:
        file_path (str): Il percorso del file audio da caricare.
        snr (float): Il rapporto segnale-rumore (SNR) desiderato in dB.
        color (str): Il colore del rumore da aggiungere ('white', 'pink', 'blue', 'brown', 'violet').

    Returns:
        array-like: Il segnale audio con rumore aggiunto.
    """

    # Carica il file audio e lo converte in un array numpy monodimensionale
    un_noised_file, _ = torchaudio.load(file_path)
    un_noised_file = un_noised_file.numpy()
    un_noised_file = np.reshape(un_noised_file, -1)

    # Calcola la potenza del segnale
    un_noised_file_watts = un_noised_file ** 2

    # Calcola il livello del segnale in dB
    un_noised_file_db = 10 * np.log10(un_noised_file_watts)

    # Calcola la potenza media del segnale e il livello medio in dB
    un_noised_file_avg_watts = np.mean(un_noised_file_watts)
    un_noised_file_avg_db = 10 * np.log10(un_noised_file_avg_watts)

    # Calcola la potenza del rumore da aggiungere in base all'SNR
    added_noise_avg_db = un_noised_file_avg_db - snr
    added_noise_avg_watts = 10 ** (added_noise_avg_db / 10)

    # Genera il rumore colorato
    added_noise = noise(len(un_noised_file), color, added_noise_avg_watts)

    # Aggiunge il rumore al segnale originale
    noised_audio = un_noised_file + added_noise

    return noised_audio


def load_audio_file(file_path='./sample_audio.wav'):
    """
    Carica un file audio e lo restituisce come array NumPy.

    Args:
        file_path (str): Il percorso del file audio da caricare.

    Returns:
        array-like: L'array NumPy rappresentante la forma d'onda audio.
    """
    waveform, _ = torchaudio.load(file_path)
    waveform = waveform.numpy()
    waveform = np.reshape(waveform, -1)
    return waveform


def save_audio_file(np_array=np.array([0.5]*1000), file_path='./sample_audio.wav', sample_rate=48000, bit_precision=16):
    """
    Salva un array NumPy come file audio.

    Args:
        np_array (array-like): L'array NumPy contenente i dati audio. Default: array di 0.5 di 1000 elementi.
        file_path (str): Il percorso in cui salvare il file audio. Default: './sample_audio.wav'.
        sample_rate (int): La frequenza di campionamento dell'audio. Default: 48000.
        bit_precision (int): La profondità di bit (bit per sample). Default: 16.
    """
    np_array = np.reshape(np_array, (1, -1))
    torch_tensor = torch.from_numpy(np_array)
    torchaudio.save(file_path, torch_tensor, sample_rate, bits_per_sample=bit_precision)