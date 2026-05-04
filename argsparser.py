import argparse


def add_common_arguments(parser):
    parser.add_argument(
        "--num_workers",
        default=0,
        type=int,
        help="How many subprocesses to use for data loading. '0' means that the data will be loaded in the main process. "
        "This parameter will be use in Pytorch DataLoader class",
    )
    parser.add_argument(
        "--working_dir",
        default="./",
        type=str,
        help="Directory to store execution results",
    )
    parser.add_argument(
        "--dataset_size", default=1.0, type=float, help="Percentage of dataset to use"
    )
    parser.add_argument(
        "--dataset", default="moving_mnist", type=str, help="Name of dataset to use"
    )
    parser.add_argument(
        "--normalize_values",
        action="store_true",
        help="It will normalize the image pixels values between 0 and 1",
    )
    parser.add_argument(
        "--light_conv",
        default="standard",
        type=str,
        help="Convolution module to use to build the model",
    )
    parser.add_argument(
        "--light_conv_pointwise",
        default="standard",
        type=str,
        help="Convolution module for 1x1 cases to use to build the model",
    )
    parser.add_argument(
        "--result_folder_suffix",
        default="",
        type=str,
        help="Suffix name of folder result. If not set, "
        "the execution datetime will be used.",
    )


def create_train_parser():
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)

    parser.add_argument(
        "--discr_in_channels",
        default=2,
        type=int,
        help="Number of channels to be use in the discriminator",
    )

    parser.add_argument(
        "--batch_size", default=10, type=int, help="Training batch size"
    )
    parser.add_argument(
        "--num_epochs", default=100, type=int, help="Number of epochs to train"
    )
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate")
    parser.add_argument(
        "--lr_beta1",
        default=0.5,
        type=float,
        help="Learning rate Beta 1 for Adam optimizer",
    )
    parser.add_argument(
        "--lr_beta2",
        default=0.9,
        type=float,
        help="Learning rate Beta 2 for Adam optimizer",
    )
    parser.add_argument(
        "--check_model_summary",
        action="store_true",
        help="Shows models summary details using "
        "torchinfo instead of training the models",
    )
    parser.add_argument(
        "--start_epoch_saving",
        default=1,
        type=int,
        help="Epoch number to start saving model weights",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed number, to guarantee reproducibility",
    )
    parser.add_argument(
        "--use_seed",
        action="store_true",
        help="Use or not use a random seed number to get the same random weights",
    )
    # Moving MNIST specific argument
    parser.add_argument(
        "--split_validation",
        action="store_true",
        help="Defines if Moving MNIST dataset "
        "will use a validation set or not [Only for Moving MNIST dataset]",
    )

    return parser


def create_predict_parser():
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)

    parser.add_argument(
        "--pixel_thresholds",
        default=[0.5, 2, 5, 10, 30],
        type=float,
        nargs="*",
        help="Pixel thresholds to use to calculate " "score metrics",
    )
    parser.add_argument(
        "--pixel_balancing_weights",
        default=[1, 1, 2, 5, 10, 30],
        type=float,
        nargs="*",
        help="Pixel balancing weights to use " "to calculate metrics",
    )
    parser.add_argument(
        "--model_weight_file_path",
        type=str,
        required=True,
        help="Directory where model weight file was saved",
    )
    parser.add_argument(
        "--save_sequence_images",
        action="store_true",
        help="The prediction will save the real and predicted sequences images",
    )
    parser.add_argument(
        "--save_sequence_npy_files",
        action="store_true",
        help="The prediction will save the real and predicted sequences .npy files",
    )
    parser.add_argument(
        "--dataset_mode",
        choices=["test", "val"],
        default="test",
        type=str,
        help="Dataset type to select in predict. " "Can be 'test' or 'val'",
    )

    parser.add_argument(
        "--skip_run_evaluation",
        action="store_true",
        help="Skip run model evaluation",
    )

    return parser

def validate_train_args(args):
    if args.num_epochs <= 0:
        raise argparse.ArgumentTypeError(
            f"'num_epochs' must be positive. Get value: {args.num_epochs}"
        )

    if args.num_epochs < args.start_epoch_saving:
        raise argparse.ArgumentTypeError(
            f"'num_epochs' (={args.num_epochs}) must be greather than 'start_epoch_saving' (={args.start_epoch_saving})"
        )
