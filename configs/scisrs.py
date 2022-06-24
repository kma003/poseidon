import ml_collections

def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.wandb_project_name = "poseidon"
    config.data_path = "data/scisrs/sig_images/yolo_images_dataset"
    config.custom_path = "metadata/scisrs"

    config.train_path = "train.csv"
    config.val_path = "val.csv"

    # config.augmentation_mode = "random_flip"

    config.batch_size = 2
    config.num_classes = 1
    config.input_shape = (500, 500, 3)

    config.epochs = 300
    config.steps_per_epoch = 1000
    config.validation_steps = 300

    config.metrics = "basic"

    return config