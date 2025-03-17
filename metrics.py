from metrics_utils import *
import numpy as np
from scipy.io import wavfile
from scipy import interpolate
from scipy.linalg import solve_toeplitz, toeplitz
import pesq as pypesq
from pystoi import stoi
import random

# Input atteso: 2 array numpy, uno per il segnale pulito di riferimento, l'altro per il segnale degradato, e la frequenza di campionamento (dovrebbe essere la stessa)
# Il modo in cui useremmo queste metriche sarebbe quello di calcolare i valori sul segnale pulito confrontato con quello rumoroso e poi sul segnale pulito confrontato con i nostri risultati di denoising.


class AudioMetrics():
    """
    Classe per calcolare le metriche audio tra due segnali.

    Args:
        target_speech (np.ndarray): Il segnale audio pulito di riferimento.
        input_speech (np.ndarray): Il segnale audio degradato o processato.
        fs (int): La frequenza di campionamento dei segnali audio.

    Raises:
        AudioMetricsException: Se le lunghezze dei segnali non corrispondono.

    Attributes:
        min_cutoff (float): Valore minimo di clipping per evitare problemi con il silenzio.
        clip_values (tuple): Tupla contenente i valori di clipping minimo e massimo.
        SNR (float): Rapporto segnale-rumore (SNR).
        SSNR (float): Rapporto segnale-rumore segmentato (SSNR).
        PESQ (float): Stima percettiva della qualità del parlato (PESQ).
        STOI (float): Intelligibilità oggettiva a breve termine del parlato (STOI).
        CSIG (float): Qualità del segnale vocale.
        CBAK (float): Qualità dell'intrusività dello sfondo.
        COVL (float): Misura della qualità complessiva.
    """

    def __init__(self, target_speech, input_speech, fs):
        if len(target_speech) != len(input_speech):
            raise AudioMetricsException("Signal lengths don't match!")

        self.min_cutoff = 0.01
        self.clip_values = (-self.min_cutoff, self.min_cutoff)

        # Le metriche SSNR e composite falliscono quando si confronta il silenzio
        # Il valore minimo del segnale viene troncato a 0.001 o -0.001 per ovviare a questo problema. Per riferimento, in un caso non di silenzio, il valore minimo era intorno a 40 (???? Trovare il valore corretto)
        # Per PESQ e STOI, i risultati sono identici indipendentemente dal fatto che 0 sia presente o meno

        # Le metriche sono le seguenti:
        # SSNR : Segmented Signal to noise ratio - Limitato da [-10,35] (più alto è meglio)
        # PESQ : Perceptable Estimation of Speech Quality - Limitato da [-0.5, 4.5]
        # STOI : Short Term Objective Intelligibilty of Speech - Da 0 a 1
        # CSIG : Quality of Speech Signal. Varia da 1 a 5 (più alto è meglio)
        # CBAK : Quality of Background intrusiveness. Varia da 1 a 5 (più alto è meglio - meno intrusivo)
        # COVL : Overall Quality measure. Varia da 1 a 5 (più alto è meglio)
        # CSIG, CBAK e COVL sono calcolati usando PESQ e alcune altre metriche come LLR e WSS

        clean_speech = np.zeros(shape=target_speech.shape)
        processed_speech = np.zeros(shape=input_speech.shape)

        for index, data in np.ndenumerate(target_speech):
            # Se il valore è inferiore alla differenza min_cutoff da 0, allora tronca
            if data == 0:
                clean_speech[index] = 0.01
            else:
                clean_speech[index] = data

        for index, data in np.ndenumerate(input_speech):
            # Se il valore è inferiore alla differenza min_cutoff da 0, allora tronca
            if data == 0:
                processed_speech[index] = 0.01
            else:
                processed_speech[index] = data

        # print('clean speech: ', clean_speech)
        # print('processed speech : ', processed_speech)
        self.SNR = snr(target_speech, input_speech)
        self.SSNR = SNRseg(target_speech, input_speech, fs)
        self.PESQ = pesq_score(clean_speech, processed_speech, fs, force_resample=True)
        self.STOI = stoi_score(clean_speech, processed_speech, fs)
        self.CSIG, self.CBAK, self.COVL = composite(
            clean_speech, processed_speech, fs)

    def display(self):
        """
        Stampa le metriche calcolate.
        """
        fstring = "{} : {:.3f}"
        metric_names = ["CSIG", "CBAK", "COVL", "PESQ", "SSNR", "STOI", "SNR"]
        for name in metric_names:
            metric_value = eval("self." + name)
            print(fstring.format(name, metric_value))


class AudioMetrics2():
    """
    Classe per calcolare le metriche audio tra due segnali (versione ridotta).
    Questa versione calcola solo SNR, SSNR e STOI.
    
    Args:
        target_speech (np.ndarray): Il segnale audio pulito di riferimento.
        input_speech (np.ndarray): Il segnale audio degradato o processato.
        fs (int): La frequenza di campionamento dei segnali audio.

    Raises:
        AudioMetricsException: Se le lunghezze dei segnali non corrispondono.

    Attributes:
        min_cutoff (float): Valore minimo di clipping per evitare problemi con il silenzio.
        clip_values (tuple): Tupla contenente i valori di clipping minimo e massimo.
        SNR (float): Rapporto segnale-rumore (SNR).
        SSNR (float): Rapporto segnale-rumore segmentato (SSNR).
        STOI (float): Intelligibilità oggettiva a breve termine del parlato (STOI).
    """
    def __init__(self, target_speech, input_speech, fs):
        if len(target_speech) != len(input_speech):
            raise AudioMetricsException("Signal lengths don't match!")

        self.min_cutoff = 0.01
        self.clip_values = (-self.min_cutoff, self.min_cutoff)

        # Le metriche SSNR e composite falliscono quando si confronta il silenzio
        # Il valore minimo del segnale viene troncato a 0.001 o -0.001 per ovviare a questo problema.
        # Per PESQ e STOI, i risultati sono identici indipendentemente dal fatto che 0 sia presente o meno


        # Le metriche sono le seguenti:
        # SSNR : Segmented Signal to noise ratio - Limitato da [-10,35] (più alto è meglio)
        # PESQ : Perceptable Estimation of Speech Quality - Limitato da [-0.5, 4.5]
        # STOI : Short Term Objective Intelligibilty of Speech - Da 0 a 1
        # CSIG : Quality of Speech Signal. Varia da 1 a 5 (più alto è meglio)
        # CBAK : Quality of Background intrusiveness. Varia da 1 a 5 (più alto è meglio - meno intrusivo)
        # COVL : Overall Quality measure. Varia da 1 a 5 (più alto è meglio)
        # CSIG,CBAK e COVL sono calcolati usando PESQ e alcune altre metriche come LLR e WSS

        clean_speech = np.zeros(shape=target_speech.shape)
        processed_speech = np.zeros(shape=input_speech.shape)

        for index, data in np.ndenumerate(target_speech):
            # Se il valore è inferiore alla differenza min_cutoff da 0, allora tronca
            if data == 0:
                clean_speech[index] = 0.01
            else:
                clean_speech[index] = data

        for index, data in np.ndenumerate(input_speech):
            # Se il valore è inferiore alla differenza min_cutoff da 0, allora tronca
            if data == 0:
                processed_speech[index] = 0.01
            else:
                processed_speech[index] = data

        # print('clean speech: ', clean_speech)
        # print('processed speech : ', processed_speech)
        self.SNR = snr(target_speech, input_speech)
        self.SSNR = SNRseg(target_speech, input_speech, fs)
        self.STOI = stoi_score(clean_speech, processed_speech, fs)

# Riferimento formula: http://www.irisa.fr/armor/lesmembres/Mohamed/Thesis/node94.html


def snr(reference, test):
    """
    Calcola il rapporto segnale-rumore (SNR) tra due segnali.

    Args:
        reference (np.ndarray): Il segnale di riferimento.
        test (np.ndarray): Il segnale di test.

    Returns:
        float: Il valore SNR in dB.
    """
    numerator = 0.0
    denominator = 0.0
    for i in range(len(reference)):
        numerator += reference[i]**2
        denominator += (reference[i] - test[i])**2
    return 10 * np.log10(numerator / denominator)


# Riferimento: https://github.com/schmiph2/pysepm

def SNRseg(clean_speech, processed_speech, fs, frameLen=0.03, overlap=0.75):
    """
    Calcola il rapporto segnale-rumore segmentato (SSNR) tra due segnali.

    Args:
        clean_speech (np.ndarray): Il segnale pulito.
        processed_speech (np.ndarray): Il segnale processato.
        fs (int): La frequenza di campionamento.
        frameLen (float): La lunghezza della finestra in secondi (default: 0.03).
        overlap (float): La sovrapposizione tra le finestre (default: 0.75).

    Returns:
        float: Il valore SSNR medio in dB.
    """
    eps = np.finfo(np.float64).eps

    winlength = round(frameLen * fs)  # lunghezza della finestra in campioni
    skiprate = int(np.floor((1 - overlap) * frameLen * fs))  # skip della finestra in campioni
    MIN_SNR = -10  # SNR minimo in dB
    MAX_SNR = 35  # SNR massimo in dB

    hannWin = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, winlength + 1) / (winlength + 1)))
    clean_speech_framed = extract_overlapped_windows(
        clean_speech, winlength, winlength - skiprate, hannWin)
    processed_speech_framed = extract_overlapped_windows(
        processed_speech, winlength, winlength - skiprate, hannWin)

    signal_energy = np.power(clean_speech_framed, 2).sum(-1)
    noise_energy = np.power(clean_speech_framed - processed_speech_framed, 2).sum(-1)

    segmental_snr = 10 * np.log10(signal_energy / (noise_energy + eps) + eps)
    segmental_snr[segmental_snr < MIN_SNR] = MIN_SNR
    segmental_snr[segmental_snr > MAX_SNR] = MAX_SNR
    segmental_snr = segmental_snr[:-1]  # rimuovi l'ultimo frame -> non valido
    return np.mean(segmental_snr)


def composite(clean_speech, processed_speech, fs):
    """
    Calcola le metriche composite CSIG, CBAK e COVL.

    Args:
        clean_speech (np.ndarray): Il segnale pulito.
        processed_speech (np.ndarray): Il segnale processato.
        fs (int): La frequenza di campionamento.

    Returns:
        tuple: Una tupla contenente i valori CSIG, CBAK e COVL.
    """
    wss_dist = wss(clean_speech, processed_speech, fs)
    llr_mean = llr(clean_speech, processed_speech, fs, used_for_composite=True)
    segSNR = SNRseg(clean_speech, processed_speech, fs)
    pesq_mos, mos_lqo = pesq(clean_speech, processed_speech, fs)
    if fs >= 16e3:
        used_pesq_val = mos_lqo
    else:
        used_pesq_val = pesq_mos

    Csig = 3.093 - 1.029 * llr_mean + 0.603 * used_pesq_val - 0.009 * wss_dist
    Csig = np.max((1, Csig))
    Csig = np.min((5, Csig))  # limita i valori a [1, 5]
    Cbak = 1.634 + 0.478 * used_pesq_val - 0.007 * wss_dist + 0.063 * segSNR
    Cbak = np.max((1, Cbak))
    Cbak = np.min((5, Cbak))  # limita i valori a [1, 5]
    Covl = 1.594 + 0.805 * used_pesq_val - 0.512 * llr_mean - 0.007 * wss_dist
    Covl = np.max((1, Covl))
    Covl = np.min((5, Covl))  # limita i valori a [1, 5]
    return Csig, Cbak, Covl


def pesq_score(clean_speech, processed_speech, fs, force_resample=False):
    """
    Calcola il punteggio PESQ (Perceptual Evaluation of Speech Quality).

    Args:
        clean_speech (np.ndarray): Il segnale pulito.
        processed_speech (np.ndarray): Il segnale processato.
        fs (int): La frequenza di campionamento.
        force_resample (bool): Forza il ricampionamento a 16000 Hz se necessario (default: False).

    Returns:
        float: Il punteggio PESQ.

    Raises:
        AudioMetricsException: Se la frequenza di campionamento non è valida per PESQ.
    """
    if fs != 8000 and fs != 16000:
        if force_resample:
            clean_speech = resample(clean_speech, fs, 16000)
            processed_speech = resample(processed_speech, fs, 16000)
            fs = 16000
        else:
            raise (AudioMetricsException(
                "Invalid sampling rate for PESQ! Need 8000 or 16000Hz but got " + str(fs) + "Hz"))
    if fs == 16000:
        score = pypesq.pesq(16000, clean_speech, processed_speech, 'wb')
        score = min(score, 4.5)
        score = max(-0.5, score)
        return (score)
    else:
        score = pypesq.pesq(16000, clean_speech, processed_speech, 'nb')
        score = min(score, 4.5)
        score = max(-0.5, score)
        return (score)

# Articolo originale http://cas.et.tudelft.nl/pubs/Taal2010.pdf
# Dice di ricampionare a 10kHz se non già a quella frequenza. Ho mantenuto le opzioni per regolare


def stoi_score(clean_speech, processed_speech, fs, force_resample=True, force_10k=True):
    """
    Calcola il punteggio STOI (Short-Time Objective Intelligibility).

    Args:
        clean_speech (np.ndarray): Il segnale pulito.
        processed_speech (np.ndarray): Il segnale processato.
        fs (int): La frequenza di campionamento.
        force_resample (bool): Forza il ricampionamento se necessario (default: True).
        force_10k (bool): Forza il ricampionamento a 10kHz (default: True).

    Returns:
        float: Il punteggio STOI.

    Raises:
        AudioMetricsException: Se viene forzato il ricampionamento a 10kHz e la frequenza di campionamento fornita è diversa.
    """
    if fs != 10000 and force_10k == True:
        if force_resample:
            clean_speech = resample(clean_speech, fs, 10000)
            processed_speech = resample(processed_speech, fs, 10000)
            fs = 10000
        else:
            raise (AudioMetricsException(
                "Forced 10kHz sample rate for STOI. Got " + str(fs) + "Hz"))
    return stoi(clean_speech, processed_speech, 10000, extended=False)