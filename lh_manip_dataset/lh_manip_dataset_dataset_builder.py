from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from PIL import Image
import pickle


class LHManipDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Wrist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(23,),
                            dtype=np.float32,
                            doc='Robot state, consists of [3x end-effector position (x, y, z) w.r.t. root_frame, '
                                '4x end-effector orientation quaternions (x, y, z, w) w.r.t. root_frame, 7x robot joint angles,'
                                '2x gripper position in (0, 0.0404), 7x robot joint velocities].',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(8,),
                        dtype=np.float32,
                        doc='Robot action, consists of [3x end-effector position offset (x, y, z) w.r.t. root_frame, '
                            '4x end-effector orientation quaternions offset (x, y, z, w) w.r.t. root_frame, 1x desired gripper opening offset].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='/media/federico/Datasets/long_horizon_manipulation_dataset/*')
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            #data = np.load(episode_path, allow_pickle=True)     # this is a list of dicts in our case
            with open(f"{episode_path}/data.pkl", "rb") as f:
                data = pickle.load(f)
            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            num_steps = len(glob.glob(f"{episode_path}/images/*.png"))
            for i in range(num_steps):
                # compute Kona language embedding
                language_embedding = self._embed([data['language_instruction'][i]])[0].numpy()
                episode.append({
                    'observation': {
                        'image': np.asarray(Image.open(f"{episode_path}/images/{i}.png")),
                        'wrist_image': np.asarray(Image.open(f"{episode_path}/images_wrist/{i}.png")),
                        'state': data['observations']['states'][i][:23],
                    },
                    'action': data['actions'][i],
                    'discount': 1.0,
                    'reward': float(i == (num_steps - 1)),
                    'is_first': i == 0,
                    'is_last': i == (num_steps - 1),
                    'is_terminal': i == (num_steps - 1),
                    'language_instruction': data['language_instruction'][i],
                    'language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }
            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        tasks_paths = glob.glob(path)
        episode_paths = []
        for task in tasks_paths:
            for i in range(10):
                episode_paths.append(f"{task}/{str(i)}")
        

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        #for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        #beam = tfds.core.lazy_imports.apache_beam
        #return (
        #        beam.Create(episode_paths)
        #        | beam.Map(_parse_example)
        #)

