import numpy as np
from scipy import interpolate
from scipy.io import wavfile
import os
import random
import warnings
import torchaudio
from pydub import AudioSegment

# Ignora gli avvisi specifici
warnings.filterwarnings("ignore")
# Imposta un seme per la riproducibilità
np.random.seed(999)

# Dizionario che mappa gli indici delle classi di rumore alle loro descrizioni
noise_class_dictionary = {
    0: "air_conditioner",
    1: "car_horn",
    2: "children_playing",
    3: "dog_bark",
    4: "drilling",
    5: "engine_idling",
    6: "gun_shot",
    7: "jackhammer",
    8: "siren",
    9: "street_music"
}

# Imposta il backend audio di torchaudio
torchaudio.set_audio_backend("soundfile")


def resample(original, old_rate, new_rate):
    """Ricampiona un segnale audio da una frequenza di campionamento a un'altra.

    Args:
        original (np.ndarray): Il segnale audio originale.
        old_rate (int): La frequenza di campionamento originale.
        new_rate (int): La frequenza di campionamento desiderata.

    Returns:
        np.ndarray: Il segnale audio ricampionato. Se le frequenze di campionamento sono uguali,
                   restituisce il segnale originale.
    """
    if old_rate != new_rate:
        duration = original.shape[0] / old_rate
        time_old = np.linspace(0, duration, original.shape[0])
        time_new = np.linspace(0, duration, int(original.shape[0] * new_rate / old_rate))
        interpolator = interpolate.interp1d(time_old, original.T)
        new_audio = interpolator(time_new).T
        return new_audio
    else:
        return original

# Lista dei nomi delle cartelle (fold)
fold_names = [f"fold{i}/" for i in range(1, 11)]


def diffNoiseType(files, noise_type):
    """Filtra una lista di file audio, restituendo solo quelli che appartengono a una classe di rumore diversa da quella specificata.

    Args:
        files (list): Una lista di nomi di file audio.
        noise_type (int): L'indice della classe di rumore da escludere.

    Returns:
        list: Una lista di nomi di file audio che non appartengono alla classe di rumore specificata.
    """
    result = []
    for i in files:
        if i.endswith(".wav"):
            fname = i.split("-")
            if fname[1] != str(noise_type):
                result.append(i)
    return result


def oneNoiseType(files, noise_type):
    """Filtra una lista di file audio, restituendo solo quelli che appartengono alla classe di rumore specificata.

    Args:
        files (list): Una lista di nomi di file audio.
        noise_type (int): L'indice della classe di rumore da includere.

    Returns:
        list: Una lista di nomi di file audio che appartengono alla classe di rumore specificata.
    """
    result = []
    for i in files:
        if i.endswith(".wav"):
            fname = i.split("-")
            if fname[1] == str(noise_type):
                result.append(i)
    return result


def genNoise(filename, num_per_fold, dest):
    """Genera file audio rumorosi sovrapponendo un file audio pulito con rumori casuali da UrbanSound8K.

    Args:
        filename (str): Il nome del file audio pulito.
        num_per_fold (int): Il numero di file rumorosi da generare per ogni fold.
        dest (str): La directory di destinazione per i file rumorosi generati.
    """
    true_path = target_folder + "/" + filename
    try:
        audio_1 = AudioSegment.from_file(true_path)
    except Exception:  # Gestione più specifica delle eccezioni è preferibile
        print(f"Errore nella decodifica audio per {true_path}, saltato.")
        return

    counter = 0
    for fold in fold_names:
        dirname = Urban8Kdir + fold
        dirlist = os.listdir(dirname)
        total_noise = len(dirlist)
        samples = np.random.choice(total_noise, num_per_fold, replace=False)
        for s in samples:
            noisefile = dirlist[s]
            try:
                audio_2 = AudioSegment.from_file(dirname + "/" + noisefile)
                combined = audio_1.overlay(audio_2, times=5)
                target_dest = dest + "/" + filename[:len(filename) - 4] + "_noise_" + str(counter) + ".wav"
                combined.export(target_dest, format="wav")
                counter += 1
            except Exception:  # Gestione più specifica delle eccezioni è preferibile
                print("Si è verificato un errore di decodifica audio, saltato questo caso")



def makeCorruptedFile_singletype(filename, dest, noise_type, snr):
    """Genera un file audio corrotto sovrapponendo un file audio pulito con un rumore dello stesso tipo, con un SNR specifico.

    Args:
        filename (str): Il nome del file audio pulito.
        dest (str): La directory di destinazione per il file audio corrotto.
        noise_type (int): L'indice della classe di rumore da utilizzare.
        snr (int): Il rapporto segnale-rumore (SNR) desiderato in dB.
    """
    succ = False
    true_path = target_folder + "/" + filename
    while not succ:
        try:
            audio_1 = AudioSegment.from_file(true_path)
        except Exception:  # Gestione più specifica delle eccezioni è preferibile
            print("Si è verificato un errore di decodifica audio per il file base... saltato")
            break

        try:
            un_noised_file, _ = torchaudio.load(true_path)
            un_noised_file = un_noised_file.numpy()
            un_noised_file = np.reshape(un_noised_file, -1)
            # Crea un array di potenza audio
            un_noised_file_watts = un_noised_file ** 2
            # Crea un array di decibel audio
            un_noised_file_db = 10 * np.log10(un_noised_file_watts)
            # Calcola la potenza media del segnale e converti in dB
            un_noised_file_avg_watts = np.mean(un_noised_file_watts)
            un_noised_file_avg_db = 10 * np.log10(un_noised_file_avg_watts)
            # Calcola la potenza del rumore
            added_noise_avg_db = un_noised_file_avg_db - snr

            fold = np.random.choice(fold_names, 1, replace=False)[0]
            dirname = Urban8Kdir + fold
            dirlist = os.listdir(dirname)
            possible_noises = oneNoiseType(dirlist, noise_type)
            total_noise = len(possible_noises)
            s = np.random.choice(total_noise, 1, replace=False)[0]
            noisefile = possible_noises[s]

            noise_src_file, _ = torchaudio.load(dirname + "/" + noisefile)
            noise_src_file = noise_src_file.numpy()
            noise_src_file = np.reshape(noise_src_file, -1)
            noise_src_file_watts = noise_src_file ** 2
            noise_src_file_db = 10 * np.log10(noise_src_file_watts)
            noise_src_file_avg_watts = np.mean(noise_src_file_watts)
            noise_src_file_avg_db = 10 * np.log10(noise_src_file_avg_watts)

            db_change = added_noise_avg_db - noise_src_file_avg_db

            audio_2 = AudioSegment.from_file(dirname + "/" + noisefile)
            audio_2 = audio_2 + db_change
            combined = audio_1.overlay(audio_2, times=5)
            target_dest = dest + "/" + filename
            combined.export(target_dest, format="wav")
            succ = True
        except Exception:  # Gestione più specifica delle eccezioni è preferibile
            #print("Si è verificato un errore di decodifica audio per il file di rumore... riprovando")
            pass

def makeCorruptedFile_differenttype(filename, dest, noise_type, snr):
    """Genera un file audio corrotto sovrapponendo un file audio pulito con un rumore di tipo diverso, con un SNR specifico.

    Args:
        filename (str): Il nome del file audio pulito.
        dest (str): La directory di destinazione per il file audio corrotto.
        noise_type (int): L'indice della classe di rumore del file pulito (il rumore aggiunto sarà di un tipo diverso).
        snr (int): Il rapporto segnale-rumore (SNR) desiderato in dB.
    """
    succ = False
    true_path = target_folder + "/" + filename
    while not succ:
        try:
            audio_1 = AudioSegment.from_file(true_path)
        except Exception:  # Gestione più specifica delle eccezioni è preferibile
            print("Si è verificato un errore di decodifica audio per il file base... saltato")
            break

        try:
            un_noised_file, _ = torchaudio.load(true_path)
            un_noised_file = un_noised_file.numpy()
            un_noised_file = np.reshape(un_noised_file, -1)
            # Crea un array di potenza audio
            un_noised_file_watts = un_noised_file ** 2
            # Crea un array di decibel audio
            un_noised_file_db = 10 * np.log10(un_noised_file_watts)
            # Calcola la potenza media del segnale e converti in dB
            un_noised_file_avg_watts = np.mean(un_noised_file_watts)
            un_noised_file_avg_db = 10 * np.log10(un_noised_file_avg_watts)
            # Calcola la potenza del rumore
            added_noise_avg_db = un_noised_file_avg_db - snr

            fold = np.random.choice(fold_names, 1, replace=False)[0]
            dirname = Urban8Kdir + fold
            dirlist = os.listdir(dirname)
            possible_noises = diffNoiseType(dirlist, noise_type)
            total_noise = len(possible_noises)
            s = np.random.choice(total_noise, 1, replace=False)[0]
            noisefile = possible_noises[s]

            noise_src_file, _ = torchaudio.load(dirname + "/" + noisefile)
            noise_src_file = noise_src_file.numpy()
            noise_src_file = np.reshape(noise_src_file, -1)
            noise_src_file_watts = noise_src_file ** 2
            noise_src_file_db = 10 * np.log10(noise_src_file_watts)
            noise_src_file_avg_watts = np.mean(noise_src_file_watts)
            noise_src_file_avg_db = 10 * np.log10(noise_src_file_avg_watts)

            db_change = added_noise_avg_db - noise_src_file_avg_db

            audio_2 = AudioSegment.from_file(dirname + "/" + noisefile)
            audio_2 = audio_2 + db_change
            combined = audio_1.overlay(audio_2, times=5)
            target_dest = dest + "/" + filename
            combined.export(target_dest, format="wav")
            succ = True
        except Exception: # Gestione più specifica delle eccezioni è preferibile
            pass



Urban8Kdir = "Datasets/UrbanSound8K/audio/"
target_folder = "Datasets/clean_trainset_28spk_wav"

# Stampa le classi di rumore disponibili
for key in noise_class_dictionary:
    print("\t{} : {}".format(key, noise_class_dictionary[key]))

# Chiede all'utente di scegliere la classe di rumore
noise_type = int(input("Enter the noise class dataset to generate :\t"))

# Definisce le cartelle di input e output per i dati di addestramento
inp_folder = "Datasets/US_Class" + str(noise_type) + "_Train_Input"
op_folder = "Datasets/US_Class" + str(noise_type) + "_Train_Output"

# Crea le cartelle di input e output se non esistono
print("Generating Training Data..")
print("Making train input folder")
if not os.path.exists(inp_folder):
    os.makedirs(inp_folder)
print("Making train output folder")
if not os.path.exists(op_folder):
    os.makedirs(op_folder)

from tqdm import tqdm

# Genera i dati di addestramento
counter = 0
for file in tqdm(os.listdir(target_folder)):
    filename = os.fsdecode(file)
    if filename.endswith(".wav"):
        snr = random.randint(0, 10)
        makeCorruptedFile_singletype(filename, inp_folder, noise_type, snr)
        snr = random.randint(0, 10)
        makeCorruptedFile_differenttype(filename, op_folder, noise_type, snr)
        counter += 1


# Definisce le cartelle per i dati di test
Urban8Kdir = "Datasets/UrbanSound8K/audio/"
target_folder = "Datasets/clean_testset_wav"
inp_folder = "Datasets/US_Class" + str(noise_type) + "_Test_Input"

# Genera i dati di test
print("Generating Testing Data..")
print("Making test input folder")
if not os.path.exists(inp_folder):
    os.makedirs(inp_folder)

counter = 0
for file in tqdm(os.listdir(target_folder)):
    filename = os.fsdecode(file)
    if filename.endswith(".wav"):
        snr = random.randint(0, 10)
        makeCorruptedFile_singletype(filename, inp_folder, noise_type, snr)
        counter += 1