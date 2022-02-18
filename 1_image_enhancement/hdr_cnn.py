# Source: https://github.com/gabrieleilertsen/hdrcnn

import cv2
import tensorflow.compat.v1 as tf
import network
import numpy as np

tf.disable_v2_behavior()

def readLDR(file, sz, clip=True, sc=1.0):
    x_buffer = cv2.imread(file)[...,::-1]

    # Clip image, so that ratio is not changed by image resize
    if clip:
        sz_in = [float(x) for x in x_buffer.shape]
        sz_out = [float(x) for x in sz]

        r_in = sz_in[1]/sz_in[0]
        r_out = sz_out[1]/sz_out[0]

        if r_out / r_in > 1.0:
            sx = sz_in[1]
            sy = sx/r_out
        else:
            sy = sz_in[0]
            sx = sy*r_out

        yo = np.maximum(0.0, (sz_in[0]-sy)/2.0)
        xo = np.maximum(0.0, (sz_in[1]-sx)/2.0)

        x_buffer = x_buffer[int(yo):int(yo+sy),int(xo):int(xo+sx),:]

    # Image resize and conversion to float
    x_buffer = cv2.resize(x_buffer, sz, interpolation=cv2.INTER_LINEAR)
    x_buffer = x_buffer.astype(np.float32)/255.0

    # Scaling and clipping
    if sc > 1.0:
        x_buffer = np.minimum(1.0, sc*x_buffer)

    x_buffer = x_buffer[np.newaxis,:,:,:]

    return x_buffer

width = 1024
height = 768
sx = int(np.maximum(32, np.round(width/32.0)*32))
sy = int(np.maximum(32, np.round(height/32.0)*32))

img_file = "data/verona_color.jpg"
x_buffer = readLDR(img_file, (sy,sx))

x = tf.placeholder(tf.float32, shape=[1, sy, sx, 3])
net = network.model(x)
y = network.get_final(net, x)

sess = tf.InteractiveSession()

load_params = tl.files.load_npz(name="hdrcnn_params.npz")
tl.files.assign_params(sess, load_params, net)
feed_dict = {x: np.maximum(x_buffer, 0.0)}
y_predict = sess.run([y], feed_dict=feed_dict)
y_gamma = np.power(np.maximum(y_predict, 0.0), 0.5)
