import os
import argparse
import tensorflow as tf
from AUGAN import AUGAN
from utils import set_path

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--dataset_dir", dest="dataset_dir", default="bdd100k", help="path of the dataset"
)
parser.add_argument(
    "--experiment_name",
    dest="experiment_name",
    type=str,
    default="bdd_exp",
    help="name of experiment",
)
parser.add_argument("--epoch", dest="epoch", type=int, default=20, help="# of epoch")
parser.add_argument(
    "--epoch_step",
    dest="epoch_step",
    type=int,
    default=10,
    help="# of epoch to decay lr",
)

parser.add_argument(
    "--batch_size", dest="batch_size", type=int, default=1, help="# images in batch"
)
parser.add_argument(
    "--train_size",
    dest="train_size",
    type=int,
    default=1e8,
    help="# images used to train",
)
parser.add_argument(
    "--load_size",
    dest="load_size",
    type=int,
    default=286,
    help="scale images to this size",
)
parser.add_argument(
    "--fine_size",
    dest="fine_size",
    type=int,
    default=256,
    help="then crop to this size",
)
parser.add_argument(
    "--ngf",
    dest="ngf",
    type=int,
    default=64,
    help="# of gen filters in first conv layer",
)
parser.add_argument(
    "--ndf",
    dest="ndf",
    type=int,
    default=64,
    help="# of discri filters in first conv layer",
)
parser.add_argument(
    "--n_d", dest="n_d", type=int, default=2, help="# of discriminators"
)
parser.add_argument(
    "--n_scale", dest="n_scale", type=int, default=2, help="# of scales"
)
parser.add_argument(
    "--gpu", dest="gpu", type=int, default=0, help="# index of gpu device"
)
parser.add_argument(
    "--input_nc", dest="input_nc", type=int, default=3, help="# of input image channels"
)
parser.add_argument(
    "--output_nc",
    dest="output_nc",
    type=int,
    default=3,
    help="# of output image channels",
)
parser.add_argument(
    "--lr", dest="lr", type=float, default=0.0002, help="initial learning rate for adam"
)
parser.add_argument(
    "--beta1", dest="beta1", type=float, default=0.5, help="momentum term of adam"
)
parser.add_argument(
    "--which_direction", dest="which_direction", default="AtoB", help="AtoB or BtoA "
)
parser.add_argument("--phase", dest="phase", default="test", help="train, test")
parser.add_argument(
    "--save_freq",
    dest="save_freq",
    type=int,
    default=1000,
    help="save a model every save_freq iterations",
)
parser.add_argument(
    "--print_freq",
    dest="print_freq",
    type=int,
    default=100,
    help="print the debug information every print_freq iterations",
)
parser.add_argument(
    "--L1_lambda",
    dest="L1_lambda",
    type=float,
    default=10.0,
    help="weight on L1 term in objective",
)
parser.add_argument(
    "--conf_lambda",
    dest="conf_lambda",
    type=float,
    default=1.0,
    help="weight on L1 term in objective",
)
parser.add_argument(
    "--use_resnet",
    dest="use_resnet",
    type=bool,
    default=True,
    help="generation network using reidule block",
)
parser.add_argument(
    "--use_lsgan",
    dest="use_lsgan",
    type=bool,
    default=True,
    help="gan loss defined in lsgan",
)
parser.add_argument(
    "--use_uncertainty",
    dest="use_uncertainty",
    type=bool,
    default=True,
    help="max size of image pool, 0 means do not use image pool",
)
parser.add_argument(
    "--max_size",
    dest="max_size",
    type=int,
    default=50,
    help="max size of image pool, 0 means do not use image pool",
)
parser.add_argument(
    "--continue_train",
    dest="continue_train",
    type=bool,
    default=False,
    help="if continue training, load the latest model: 1: true, 0: false",
)
parser.add_argument(
    "--save_conf",
    dest="save_conf",
    type=bool,
    default=False,
    help="save conf map in test phase",
)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


def main(args):
    """Summary

    Args:
        args (_type_): _description_
    """
    set_path(args, args.experiment_name)

    tfconfig = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=tfconfig) as sess:
        model = AUGAN(sess, args)
        model.train(args) if args.phase == "train" else model.test(args)


if __name__ == "__main__":
    main(args)

# python main.py --dataset_dir alderley --phase test --experiment_name alderley_exp --batch_size 1 --load_size 286 --fine_size 256
