import math
from typing import List

from reduction import Reducer, UMAPReducer, PCAReducer
from mrcnn.config import Config
from mrcnn.utils import Dataset
import mrcnn.model as modellib
from training.detection import run_feature_detector
import numpy as np

REDUCERS = {
    'umap': UMAPReducer,
    'pca': PCAReducer
}


class LevelDatasetReducer:
    def __init__(self, reducer: str, model: modellib.MaskRCNN, config: Config):
        assert reducer in REDUCERS.keys()

        self.reducer = reducer
        self.levels = 4

        self.level_reducers = [self.create_reducer() for _ in range(self.levels)]
        self.model = model
        self.config = config
        self.feature_detector = self.model.get_feature_detector()

    def create_reducer(self):
        return REDUCERS[self.reducer]()

    def compute_feature(self, dataset: Dataset, image_id: int) -> np.array:
        image, *_ = modellib.load_image_gt(dataset, self.config, image_id)
        output = run_feature_detector(self.feature_detector, self.model, image)
        return output

    def compute_features(self, dataset: Dataset, image_ids: List[int], level: int = None) -> list:
        features = []
        for i, image_id in enumerate(image_ids):
            feature = self.compute_feature(dataset, image_id)
            feature = feature[level].flatten() if level is not None else feature
            features.append(feature)

        return features

    def train(self, dataset: Dataset, samples: int):
        ids = dataset.image_ids
        np.random.shuffle(ids)

        for level in range(self.levels):
            print(f"Computing features to train level {level}")
            features = self.compute_features(dataset, ids[:samples], level=level)

            print(f"Training level {level}")
            self.level_reducers[level].train(np.array(features))

    def reduce_dataset(self, dataset: Dataset, batch_size: int) -> np.array:
        embeddings_per_level = [None] * self.levels

        ids = dataset.image_ids
        np.random.shuffle(ids)

        batch_count = math.ceil(len(ids) / batch_size)

        for batch_no in range(batch_count):
            print(f"\rComputing features for batch {batch_no}/{batch_count}", end='')
            batch_ids = ids[batch_no * batch_size:min(batch_size * (batch_no + 1), len(ids))]
            features = self.compute_features(dataset, batch_ids)

            for level in range(self.levels):
                level_features = []
                for feature in features:
                    level_features.append(feature[level].flatten())

                level_embeddings = self.level_reducers[level].reduce_dimension(np.array(level_features))
                embeddings_per_level[level] = level_embeddings if embeddings_per_level[level] is None else np.concatenate((embeddings_per_level[level], level_embeddings))

        return embeddings_per_level
