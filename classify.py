## classify.py -- actually classify a sequence with DeepSpeech
##
## Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import scipy.io.wavfile as wav

import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
from collections import namedtuple
sys.path.append("DeepSpeech")
import DeepSpeech

from tf_logits import get_logits


# These are the tokens that we're allowed to use.
# The - token is special and corresponds to the epsilon
# value in CTC decoding, and can not occur in the phrase.
toks = " abcdefghijklmnopqrstuvwxyz'-"
restore_path = "deepspeech-0.4.1-checkpoint/model.v0.4.1"

def classify(input, psearch):
    with tf.Session() as sess:
        _, audio = wav.read(input)
        N = len(audio)
        new_input = tf.placeholder(tf.float32, [1, N])
        lengths = tf.placeholder(tf.int32, [1])

        # get logits (probability matrix) from deepspeech
        with tf.variable_scope("", reuse=tf.AUTO_REUSE):
            logits = get_logits(new_input, lengths)

        saver = tf.train.Saver()
        saver.restore(sess, restore_path)
         
        # decode them using either greedy or beam search
        decoded, _ = tf.nn.ctc_beam_search_decoder(logits, lengths, merge_repeated=False, beam_width=(1 if psearch=="greedy" else 100))

        #print('logits shape', logits.shape)
        length = (len(audio)-1)//320
        r = sess.run(decoded, {new_input: [audio],
                               lengths: [length]})

        return "".join([toks[x] for x in r[0].values])

print(classify(sys.argv[1], sys.argv[2]))