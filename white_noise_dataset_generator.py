import colored_noise_utils as noiser  # Presuppone che questo sia un modulo personalizzato o di terze parti
from pathlib import Path
from matplotlib import pyplot as plt  # Non utilizzato direttamente, ma potrebbe servire per visualizzazioni future
import numpy as np
import os
from tqdm import tqdm

# Definisci i percorsi per i dati di addestramento e test
TRAINING_INPUT_PATH = 'Datasets/WhiteNoise_Train_Input'
TRAINING_OUTPUT_PATH = 'Datasets/WhiteNoise_Train_Output'
TESTING_INPUT_PATH = 'Datasets/WhiteNoise_Test_Input'

# Definisci i percorsi per i file audio puliti di addestramento e test
CLEAN_TRAINING_DIR = Path('Datasets/clean_trainset_28spk_wav')
CLEAN_TESTING_DIR = Path("Datasets/clean_testset_wav")

def get_wav_files(dir_path: Path) -> list[Path]:
    """
    Recupera tutti i file .wav da una directory e dalle sue sottodirectory.

    Args:
        dir_path: Il percorso della directory da cui recuperare i file.

    Returns:
        Una lista di oggetti Path che rappresentano i file .wav trovati.
    """
    return sorted(list(dir_path.rglob('*.wav')))


# Ottieni la lista dei file audio puliti
clean_training_dir_wav_files = get_wav_files(CLEAN_TRAINING_DIR)
clean_testing_dir_wav_files = get_wav_files(CLEAN_TESTING_DIR)
print("Total training samples:", len(clean_training_dir_wav_files))


def create_directory_if_not_exists(dir_path: str):
    """
    Crea una directory se non esiste già.

    Args:
        dir_path: Il percorso della directory da creare.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

print("Generating Training data")
create_directory_if_not_exists(TRAINING_INPUT_PATH)
create_directory_if_not_exists(TRAINING_OUTPUT_PATH)



def generate_and_save_noisy_audio(clean_audio_files: list[Path], output_dir: str, color: str = 'white'):
    """
    Genera audio con rumore gaussiano colorato e lo salva in una directory specifica.

    Args:
        clean_audio_files: Una lista di oggetti Path che rappresentano i file audio puliti.
        output_dir: La directory in cui salvare i file audio con rumore.
        color: Il colore del rumore da generare (default: 'white').

    Side Effects:
        Crea file audio con rumore nelle directory specificate.
    """
    for audio_file in tqdm(clean_audio_files):
        #_ = noiser.load_audio_file(file_path=audio_file)  # Carica ma non usa l'audio non rumoroso, probabilmente per side-effect (es., verifica formato).

        # Genera due campioni di rumore con SNR casuali per ogni file pulito (input e output diversi)
        for i in range(2): #genera due volte per avere input ed output diversi.
            random_snr = np.random.randint(0, 10)
            white_gaussian_noised_audio = noiser.gen_colored_gaussian_noise(file_path=audio_file, snr=random_snr, color=color)
            if i == 0:
                save_path = os.path.join(TRAINING_INPUT_PATH, audio_file.name)
            else:
                save_path = os.path.join(TRAINING_OUTPUT_PATH, audio_file.name)
            
            noiser.save_audio_file(np_array=white_gaussian_noised_audio, file_path=save_path)


generate_and_save_noisy_audio(clean_training_dir_wav_files, TRAINING_INPUT_PATH)



print("Generating Testing data")
create_directory_if_not_exists(TESTING_INPUT_PATH)

def generate_and_save_testing_noisy_audio(clean_audio_files: list[Path], output_dir: str, color: str = 'white'):
    """
    Genera audio con rumore gaussiano colorato per il testing e lo salva in una directory specifica.

    Args:
        clean_audio_files: Una lista di oggetti Path che rappresentano i file audio puliti di test.
        output_dir:  La directory in cui salvare i file audio rumorosi di test.
        color: Il colore del rumore (default: 'white').
        
    Side Effects:
        Crea file audio con rumore nella directory di testing specificata.
    """
    for audio_file in tqdm(clean_audio_files):
        #_ = noiser.load_audio_file(file_path=audio_file)   # Carica ma non usa (vedi sopra)

        random_snr = np.random.randint(0, 10)
        white_gaussian_noised_audio = noiser.gen_colored_gaussian_noise(file_path=audio_file, snr=random_snr, color=color)
        noiser.save_audio_file(np_array=white_gaussian_noised_audio, file_path=os.path.join(output_dir, audio_file.name))

generate_and_save_testing_noisy_audio(clean_testing_dir_wav_files, TESTING_INPUT_PATH)