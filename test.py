import tensorflow as tf
from models import generator_resnet
import numpy as np

tf.compat.v1.disable_eager_execution()

A = np.random.rand(256, 256)
A_tensor = tf.compat.v1.placeholder(shape=(1, 256, 256, 3), dtype=tf.float32)
options = {"output_c_dim": 3, "gf_dim": 64, "df_dim": 32}
# gf_dim args.ngf,
# df_dim args.ndf // args.n_d,
# output_c_dim args.output_nc,
generator_resnet(A_tensor, options=options)

# Vreemd... shift + enter blijft mis gaan...
# I removed this from keybindings.json && !findInputFocussed && !jupyter.ownsSelection && !notebookEditorFocused && !replaceInputFocussed
# Because otherwise it would not trigger
