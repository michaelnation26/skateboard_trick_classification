RGB_CLASS_NAMES = [
    'kickflip',
    '360_kickflip',
    '50-50',
    'nosegrind',
    'boardslide',
    'tailslide',
    'fail'
]

RGB_CLASS_NAME_TO_IDX = {class_name: idx
                         for idx, class_name in enumerate(RGB_CLASS_NAMES)}

RGB_N_CLASSES = 7
RGB_FRAME_HEIGHT = 224
RGB_FRAME_WIDTH = 224
CHANNELS = 3
RGB_N_FRAMES = 64
RGB_TRAINING_BATCH_SIZE = 6
RGB_VALIDATION_BATCH_SIZE = 4
RGB_TEST_BATCH_SIZE = 16

AUDIO_CLASS_NAMES = ['air', 'fail', 'grind', 'slide']
N_AUDIO_CLASSES = len(AUDIO_CLASS_NAMES)
SPECTROGRAM_HEIGHT = 224
SPECTROGRAM_WIDTH = 224
AUDIO_TRAINING_BATCH_SIZE = 32
AUDIO_VALIDATION_BATCH_SIZE = 8
AUDIO_TEST_BATCH_SIZE = 32

MODELS_DIR = 'models'
RGB_MODEL_FILEPATH = f'{MODELS_DIR}/rgb_model.h5'
AUDIO_MODEL_FILEPATH = f'{MODELS_DIR}/audio_model.h5'
RGB_FROZEN_AUDIO_MODEL_FILEPATH = f'{MODELS_DIR}/rgb_frozen_audio_model.h5'
RGB_AUDIO_MODEL_FILEPATH = f'{MODELS_DIR}/rgb_audio_model.h5'

VALIDATION_SPLIT = 0.2
VIDEO_TRAINING_VALIDATION_DIR =  'data/training_validation/video'
VIDEO_TRAINING_DIR =  'data/training/video'
VIDEO_VALIDATION_DIR =  'data/validation/video'
VIDEO_TEST_DIR =  'data/test/video'
WAV_TRAINING_DIR = 'data/training/audio/wav'
WAV_VALIDATION_DIR = 'data/validation/audio/wav'
WAV_TEST_DIR = 'data/test/audio/wav'
SPECTROGRAM_TRAINING_DIR = 'data/training/audio/spectrogram'
SPECTROGRAM_VALIDATION_DIR = 'data/validation/audio/spectrogram'
SPECTROGRAM_TEST_DIR = 'data/test/audio/spectrogram'

AUDIO_CLASS_NAMES = [
    'air',
    'fail',
    'grind',
    'slide'
]

VIDEO_TO_AUDIO_LABEL_MAPPING = {
    '360_kickflip': 'air',
    'heelflip': 'air',
    'kickflip': 'air',
    'nollie_fakie_heelflip': 'air',
    'nollie_fakie_kickflip': 'air',
    'bs_kickflip': 'air',
    'fs_kickflip': 'air',
    '50-50': 'grind',
    'nosegrind': 'grind',
    'boardslide': 'slide',
    'tailslide': 'slide',
    'fail': 'fail'
}
