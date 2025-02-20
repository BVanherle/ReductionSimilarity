import data.mrcnn_dimo
from data import utils as data_utils
from data import mrcnn_dimo
import os, random
from mrcnn import visualize as mrcnn_visualise
from mrcnn import model as modellib
from reduction.dataset_reducer import DatasetReducer
from reduction.level_dataset_reducer import LevelDatasetReducer
from reduction.umap_reducer import UMAPReducer
from training import evaluation, detection
from utils import visualize, interactions, file_io
import configparser
import cv2
import numpy as np
from training import mrcnn as mrcnn_training
import tensorflow.keras.backend as K
from utils import plotting
import logging.config
from reduction import similarity

config = configparser.ConfigParser()
config.read('config.ini')

DIMO_PATH = config['USER_SETTINGS']['dimo_path']

logfile = "log.txt"
logging.basicConfig(filename=logfile, level=logging.INFO, format='%(message)s')
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})


def test_batch(batch_file: str):
    model_tests = file_io.read_test_batch(batch_file)

    for test in model_tests:
        dataset = data.mrcnn_dimo.get_test_dimo_dataset(DIMO_PATH, [test.test_subset])
        config = data.mrcnn_dimo.get_test_dimo_config(dataset, test.model_id)

        model = mrcnn_training.load_model(test.model_id, config)
        results = detection.get_detections_dataset(dataset, model, config)
        ap_50 = evaluation.compute_map(results, dataset, config, 0.5)
        ap_75 = evaluation.compute_map(results, dataset, config, 0.75)
        ap = evaluation.compute_coco_ap(results, dataset, config)

        test.metrics = {
            "ap": ap,
            "ap_50": ap_50,
            "ap_75": ap_75,
        }

    filename = f"{batch_file.split('.')[0]}_results.csv"
    file_io.write_test_metrics(model_tests, filename)


def train_subsets(subsets: list, model_id: str = None, augment: bool = False, transfer_learning: bool = False,
                  train_image_counts: list = None, ft_subsets: list = None, ft_image_count: int = None,
                  layers: str = None, save_all: bool = False, dimo_path: str = None):
    target_dimo_path = dimo_path if dimo_path is not None else DIMO_PATH
    # load training set
    train, val, config = mrcnn_dimo.get_dimo_datasets(target_dimo_path, subsets, train_image_counts=train_image_counts)

    # if specified, load fintuning dataset
    ft_train = None
    if ft_subsets:
        ft_train, _, _ = mrcnn_dimo.get_dimo_datasets(target_dimo_path, ft_subsets, train_image_counts=ft_image_count)

    # load model to continue training, if specified
    model = mrcnn_training.load_model(model_id, config, mode="training") if model_id else None
    layers = layers if layers else 'heads'
    # train model
    mrcnn_training.train(train, val, config, augment=augment, use_coco_weights=transfer_learning,
                         checkpoint_model=model, ft_train_set=ft_train, layers=layers, save_all=save_all)


def prepare_subsets(subsets: list, override: bool = False, split_scenes: bool = False, dimo_path: str = None):
    target_dimo_path = dimo_path if dimo_path is not None else DIMO_PATH
    # data_utils.create_dimo_masks(target_dimo_path, subsets, override=override)
    data_utils.create_dimo_train_split(target_dimo_path, subsets, seed=10, split_scenes=split_scenes)


def show_subsets(subsets: list, dimo_path: str = None):
    target_dimo_path = dimo_path if dimo_path is not None else DIMO_PATH
    dataset_train, dataset_val, config = mrcnn_dimo.get_dimo_datasets(target_dimo_path, subsets)
    config.USE_MINI_MASK = False

    print(f"training images: {len(dataset_train.image_ids)}")
    print(f"validation images: {len(dataset_val.image_ids)}")

    while True:
        image_id = random.choice(dataset_train.image_ids)
        image_info = dataset_train.image_info[image_id]
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_train, config, image_id)
        # Compute Bounding box
        mrcnn_visualise.display_instances(image, gt_bbox, gt_mask, gt_class_id, dataset_train.class_names,
                                          title=image_info['id'])


def test_subsets(subsets: list, model_id: str, save_results: bool = False, dimo_path: str = None):
    dataset = data.mrcnn_dimo.get_test_dimo_dataset(DIMO_PATH, subsets)
    config = data.mrcnn_dimo.get_test_dimo_config(dataset, model_id)

    model = mrcnn_training.load_model(model_id, config)
    results = detection.get_detections_dataset(dataset, model, config)
    ap_50 = evaluation.compute_map(results, dataset, config, 0.5)
    ap_75 = evaluation.compute_map(results, dataset, config, 0.75)
    ap = evaluation.compute_coco_ap(results, dataset, config)

    print(f"AP = {ap}")
    print(f"AP 50 = {ap_50}")
    print(f"AP 75 = {ap_75}")

    if save_results:
        visualize.save_results(results, mrcnn_dimo.get_dataset_images(dataset, config), "results/", dataset.class_names)
    else:
        visualize.show_results(results, mrcnn_dimo.get_dataset_images(dataset, config), dataset.class_names)


def test_folder(folder: str, model_id: str, num_classes: int, select_roi=False, save_folder=None):
    config = data.mrcnn_dimo.DimoInferenceConfig(num_classes=num_classes)
    model = mrcnn_training.load_model(model_id, config)

    images = [cv2.cvtColor(cv2.imread(os.path.join(folder, file)), cv2.COLOR_BGR2RGB) for file in os.listdir(folder) if
              file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg")]
    class_names = [str(i) for i in range(num_classes)]

    if select_roi:
        rois = interactions.select_rois(images)
        images = [image[r[0]:r[1], r[2]: r[3]] for r, image in zip(rois, images)]

    results = detection.get_detections_images(images, model)

    if save_folder:
        visualize.save_results(results, images, save_folder, class_names)
    else:
        visualize.show_results(results, images, class_names)


def test_epochs(subsets: list, models: list):
    test_frequency = 2
    dataset = data.mrcnn_dimo.get_test_dimo_dataset(DIMO_PATH, subsets)

    for model_id in models:
        config = data.mrcnn_dimo.get_test_dimo_config(dataset, model_id)
        available_epochs = np.array(
            mrcnn_training.get_available_epochs(f"{mrcnn_training.get_model_folder()}/{model_id}"))
        test_epochs = np.arange(0, np.max(available_epochs) + 1, test_frequency).astype(np.int)
        test_epochs[0] += 1
        tested_epochs = []
        aps = []
        for epoch in test_epochs:
            if epoch in available_epochs:
                model = mrcnn_training.load_model(model_id, config, epoch)
                results = detection.get_detections_dataset(dataset, model, config)
                ap = evaluation.compute_coco_ap(results, dataset, config)

                tested_epochs.append(epoch)
                aps.append(ap)

                del model
                K.clear_session()

        file_io.write_model_epochs(model_id, aps, tested_epochs)


def compare_feature_maps(model_id: str):
    subsets = ["real_jaigo_000-150", "sim_jaigo_real_light_real_pose", "sim_jaigo_real_light_rand_pose",
               "sim_jaigo_rand_light_real_pose", "sim_jaigo_rand_light_rand_pose"]

    results = {}

    images_per_dataset = 20

    for level in range(4):
        results[str(level)] = {}
        print(f"Creating Reducer for level {level}\n")
        total_dataset, val, _ = data.mrcnn_dimo.get_dimo_datasets(DIMO_PATH, subsets,
                                                                  train_image_counts=[images_per_dataset] * len(
                                                                      subsets))
        config = data.mrcnn_dimo.get_test_dimo_config(total_dataset, model_id)
        model = mrcnn_training.load_model(model_id, config)

        reducer = UMAPReducer()
        dataset_reducer = DatasetReducer(reducer, model, level, config)
        dataset_reducer.train(total_dataset, samples=5)

        for set in subsets:
            print(f"Reducing dimension of {set}")
            subset_dataset, val, _ = data.mrcnn_dimo.get_dimo_datasets(DIMO_PATH, [set],
                                                                       train_image_counts=[images_per_dataset])

            embedding = dataset_reducer.reduce_dataset(subset_dataset, batch_size=2)
            results[str(level)][set] = embedding.tolist()

        del model
        K.clear_session()

    file_io.write_embeddings(results, "embeddings/umap.json")


def compare_level_reducer(model_id: str):
    images_per_dataset = 1000
    reducer = 'umap'

    subsets = ["real_jaigo_000-150", "sim_jaigo_real_light_real_pose", "sim_jaigo_real_light_rand_pose",
               "sim_jaigo_rand_light_real_pose", "sim_jaigo_rand_light_rand_pose"]

    total_dataset, val, _ = data.mrcnn_dimo.get_dimo_datasets(DIMO_PATH, subsets,
                                                              train_image_counts=[images_per_dataset] * len(subsets))
    config = data.mrcnn_dimo.get_test_dimo_config(total_dataset, model_id)
    model = mrcnn_training.load_model(model_id, config)

    level_reducer = LevelDatasetReducer(reducer, model, config)

    level_reducer.train(total_dataset, 400)

    results = {i: {} for i in range(4)}

    for set in subsets:
        print(f"\nReducing dimension of {set}")
        subset_dataset, val, _ = data.mrcnn_dimo.get_dimo_datasets(DIMO_PATH, [set],
                                                                   train_image_counts=[images_per_dataset])

        embedding = level_reducer.reduce_dataset(subset_dataset, batch_size=100)

        for level in range(len(embedding)):
            results[level][set] = np.array(embedding[level]).tolist()

    file_io.write_embeddings(results, f"embeddings/{reducer}.json")


def show_embedding(file_name: str):
    titles = ["real", "synth", "synth, rand pose", "synth, rand light", "synth, rand all"]
    data = file_io.read_embeddings(file_name)
    plotting.plot_feature_maps(data, titles)


def compute_overlaps(file_name: str):
    target_name = "real_jaigo_000-150"
    embedding = file_io.read_embeddings(file_name)

    scores = []

    for level in embedding.keys():
        level_scores = []
        target_embedding = np.array(embedding[level][target_name])
        for dataset_name in embedding[level].keys():
            dataset_embedding = np.array(embedding[level][dataset_name])
            score = similarity.gaussian_mixture_overlap(dataset_embedding, target_embedding)
            print(f"Level {level}: {dataset_name} --> {score}")
            level_scores.append(score)

        scores.append(level_scores)

    plotting.plot_scores_ap(scores, [78.83, 77.32, 69.32, 76.36, 69.56])

    return scores


if __name__ == "__main__":
    #compare_level_reducer("dimo20220315T0958")
    #show_embedding("embeddings/umap.json")
    compute_overlaps("embeddings/pca.json")
