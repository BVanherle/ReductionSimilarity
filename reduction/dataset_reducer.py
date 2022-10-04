import math
from typing import List

from reducer import Reducer
from mrcnn.config import Config
from mrcnn.utils import Dataset
import mrcnn.model as modellib
from training.detection import run_feature_detector
import numpy as np


class DatasetReducer:
    def __init__(self, reducer: Reducer, model: modellib.MaskRCNN, level: int, config: Config):
        assert 0 <= level <= 3

        self.reducer = reducer
        self.level = level
        self.model = model
        self.config = config
        self.feature_detector = self.model.get_feature_detector()

    def compute_feature(self, dataset: Dataset, image_id: int) -> np.array:
        image, *_ = modellib.load_image_gt(dataset, self.config, image_id)
        output = run_feature_detector(self.feature_detector, self.model, image)[self.level]
        return output.flatten()

    def compute_features(self, dataset: Dataset, image_ids: List[int]) -> np.array:
        features = []
        for i, image_id in enumerate(image_ids):
            print(f"\r\tComputing image {i}/{len(image_ids)}", end='')
            features.append(self.compute_feature(dataset, image_id))

        return np.array(features)

    def train(self, dataset: Dataset, config: Config, samples: int):
        ids = dataset.image_ids
        np.random.shuffle(ids)

        print("Computing features for training")
        features = self.compute_features(dataset, ids[:samples])

        print("\nTraining model")
        self.reducer.train(features)

    def reduce_dataset(self, dataset: Dataset, batch_size: int) -> np.array:
        ids = dataset.image_ids
        np.random.shuffle(ids)

        print("Computing embeddings for dataset")
        batch_count = math.ceil(len(ids) / batch_size)

        embeddings = None
        for batch_no in range(batch_count):
            print(f"\tComputing features for batch {batch_no}/{batch_count}")
            batch_ids = ids[batch_no * batch_size:min(batch_size * (batch_no + 1), len(ids))]
            features = self.compute_features(dataset, batch_ids)

            print(f"\tComputing embeddings for batch {batch_no}/{batch_count}")
            embedding = self.reducer.reduce_dimension(features)

            embeddings = embedding if embeddings is None else np.concatenate((embedding, embeddings))

        return embeddings
