import glob
import os
import sys

import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
import numpy as np

from . import config


class DataGenerator(Sequence):

    def __init__(self, video_directory, batch_size, is_training, rgb_data_only=False, spectrogram_dir=None):
        if not rgb_data_only and not spectrogram_dir:
            raise ValueError("spectrogram_dir is required if rgb_data_only is set to False.")
        
        self.video_filepaths = self._get_filepaths(video_directory)
        self.batch_size = batch_size
        self.is_training = is_training
        self.rgb_data_only = rgb_data_only
        self.spectrogram_dir = spectrogram_dir

        if self.is_training:
            np.random.shuffle(self.video_filepaths)
            self.data_augmentor = ImageDataGenerator(
                horizontal_flip=True,
                brightness_range=[0.7, 1.3],
                zoom_range=0.3
            )
        else:
            # self.labels are used as y_true when calculating model metrics
            self.labels = [config.RGB_CLASS_NAME_TO_IDX[self._get_label_name(filepath)] 
                           for filepath in self.video_filepaths]

    def __getitem__(self, batch_num):
        n_filepaths = len(self.video_filepaths)
        idx_start = batch_num * self.batch_size
        idx_end = min((batch_num+1) * self.batch_size, n_filepaths)
        filepaths_for_batch = self.video_filepaths[idx_start:idx_end]
        X, y = self._get_batch(filepaths_for_batch)

        return X, y

    def __len__(self):
        return int(np.ceil(len(self.video_filepaths) / self.batch_size))

    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.video_filepaths)

    def _apply_data_augmentation(self, frames):
        frames_augmented = []
        img_shape = (config.RGB_FRAME_HEIGHT, config.RGB_FRAME_WIDTH, config.CHANNELS)
        transform_params = self.data_augmentor.get_random_transform(img_shape)
        for frame in frames:
            frame_augmented = self.data_augmentor.apply_transform(frame, transform_params)
            frame_augmented = frame_augmented / 255.0
            frames_augmented.append(frame_augmented)

        return np.array(frames_augmented)

    def _class_directory_is_class(self, class_directory):
        return class_directory.split("/")[-1] in config.RGB_CLASS_NAME_TO_IDX

    def _crop_frame(self, frame, crop_location):
        """
        crop_location=0.0 -> far left (landscape frame) or far top (portrait frame)
        crop_location=0.5 -> center crop
        crop_location==1.0 -> far right (landscape frame) or far bottom (portrait frame)
        """
        h, w = frame.shape[:2]
        length = min(h, w)

        y_margin = h - length
        y0 = int(y_margin * crop_location)
        y1 = y0 + length

        x_margin = w - length
        x0 = int(x_margin * crop_location)
        x1 = x0 + length

        frame_cropped = frame[y0:y1, x0:x1]
        frame_resized = cv2.resize(frame_cropped, (config.RGB_FRAME_HEIGHT, config.RGB_FRAME_WIDTH))

        return frame_resized

    def _crop_frames(self, frames, center_crop=True):
        """If center_crop is False, the crop location will be random."""
        cropped_frames = []
        crop_location = 0.5 if center_crop else np.random.random_sample()
        for frame in frames:
            cropped_frame = self._crop_frame(frame, crop_location)
            cropped_frames.append(cropped_frame)

        return np.array(cropped_frames)

    def _extract_frames_from_video(self, video_filepath):
        video = cv2.VideoCapture(video_filepath)
        if not video.isOpened():
            raise FileNotFoundError("The input video path you provided is invalid.")

        frames = []
        while video.isOpened():
            grabbed, frame_bgr = video.read()
            if not grabbed:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        video.release()

        return frames

    def _get_batch(self, batch_video_filepaths):
        batch_frames = [self._get_frames(fp) for fp in batch_video_filepaths]
        batch_frames = np.array(batch_frames)

        batch_labels = [config.RGB_CLASS_NAME_TO_IDX[self._get_label_name(fp)] 
                        for fp in batch_video_filepaths]
        batch_labels = np.array(batch_labels)

        if self.rgb_data_only:
            X, y = batch_frames, batch_labels
        else:
            batch_spectrograms = [self._get_spectrogram(fp) for fp in batch_video_filepaths]
            batch_spectrograms = np.array(batch_spectrograms)
            X, y = [batch_frames, batch_spectrograms], batch_labels

        return X, y

    def _get_frames(self, video_filepath):
        frames = self._extract_frames_from_video(video_filepath)
        frames = self._pad_frames_list(frames)
        frames = self._subsample_frames(frames)
        frames = self._crop_frames(frames, center_crop=not self.is_training)

        if self.is_training:
            frames = self._apply_data_augmentation(frames)
        else:
            frames = frames / 255.0

        return frames

    def _get_filepaths(self, directory):
        filepaths = []
        class_pathname = os.path.join(directory, "*")
        class_dirs = glob.glob(class_pathname)
        for class_dir in class_dirs:
            if not self._class_directory_is_class(class_dir):
                continue
            pathname = os.path.join(class_dir, "*")
            filepaths += glob.glob(pathname)

        return filepaths

    def _get_label_name(self, filepath):
        return filepath.split("/")[-2]

    def _get_spectrogram(self, video_filepath):
        video_label = self._get_label_name(video_filepath)
        audio_label = config.VIDEO_TO_AUDIO_LABEL_MAPPING[video_label]
        video_filename = video_filepath.split('/')[-1].split('.')[0]
        spectrogram_filepath = f'{self.spectrogram_dir}/{audio_label}/{video_label}_{video_filename}.jpg'

        spectrogram_img_bgr = cv2.imread(spectrogram_filepath)
        spectrogram_img = cv2.cvtColor(spectrogram_img_bgr, cv2.COLOR_BGR2RGB)
        spectrogram_img = spectrogram_img / 255.0
        spectrogram_img = cv2.resize(spectrogram_img, (config.SPECTROGRAM_HEIGHT, config.SPECTROGRAM_WIDTH))

        return spectrogram_img

    def _pad_frames_list(self, frames):
        """If the length of frames list is less than RGB_N_FRAMES,
        it will be padded with blank frames (RGB -> 000).
        """
        if len(frames) < config.RGB_N_FRAMES:
            n_pad_frames = config.RGB_N_FRAMES - len(frames)
            for _ in range(n_pad_frames):
                blank_frame = np.zeros((config.RGB_FRAME_HEIGHT, config.RGB_FRAME_WIDTH, config.CHANNELS))
                frames.append(blank_frame)

        return frames

    def _subsample_frames(self, video_clip_frames):
        """Frames are subsampled uniformly. i.e. A fixed number of frames are
        subsampled from video_clip_frames with equal distance from each other.
        """
        subsampled_frames = []
        current_ix = 0
        step_size = len(video_clip_frames) / float(config.RGB_N_FRAMES)
        for _ in range(config.RGB_N_FRAMES):
            frame = video_clip_frames[int(current_ix)]
            subsampled_frames.append(frame)
            current_ix += step_size

        return np.array(subsampled_frames)
