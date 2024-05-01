import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

from utils import set_path
from main import args
from AUGAN import AUGAN
from utils import load_test_data

tf.compat.v1.disable_eager_execution()

ddata = r"F:\Pictures\image.webp"

A = load_test_data(ddata)

tfconfig = tf.compat.v1.ConfigProto(allow_soft_placement=True)
set_path(args, args.experiment_name)
tfconfig.gpu_options.allow_growth = True
with tf.compat.v1.Session(config=tfconfig) as sess:
    model = AUGAN(sess, args)
    print(model)
    init_op = tf.compat.v1.global_variables_initializer()
    model.sess.run(init_op)
    model_dir = "%s_%s" % (model.dataset_dir, model.image_size)
    full_checkpoint_dir = os.path.join(args.checkpoint_dir, model_dir)
    full_checkpoint_dir = os.path.abspath(full_checkpoint_dir)
    # result = model.load(full_checkpoint_dir)
    # print(f"Loading model checkpoint..: {result}")
    print(full_checkpoint_dir)
    ckpt = tf.train.get_checkpoint_state(full_checkpoint_dir)
    print(f"Checkpoint state {ckpt} - path {ckpt.model_checkpoint_path}")
    if ckpt and ckpt.model_checkpoint_path:
        print("we have done it!")
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        model.saver.restore(sess, os.path.join(full_checkpoint_dir, ckpt_name))

    out_var, refine_var, in_var, rec_var, cycle_var, percep_var, conf_var = (
        model.testB,
        model.refine_testB,
        model.test_A,
        model.rec_testA,
        model.rec_cycle_A,
        model.testA_percep,
        model.test_pred_confA,
    )

    sample_image = [load_test_data(ddata, args.fine_size)]
    sample_image = np.array(sample_image).astype(np.float32)
    (fake_img,) = model.sess.run([out_var], feed_dict={in_var: sample_image})

fig, ax = plt.subplots()
ax.imshow(fake_img[0])
fig.savefig(os.path.expanduser("~/test.png"))
