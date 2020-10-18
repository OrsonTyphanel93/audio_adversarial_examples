## classify.py -- actually classify a sequence with DeepSpeech
##
## Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import numpy as np
import tensorflow as tf

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

def classify(input):
    with tf.Session() as sess:
        _, audio = wav.read(input)
        N = len(audio)
        new_input = tf.placeholder(tf.float32, [1, N])
        lengths = tf.placeholder(tf.int32, [1])

        with tf.variable_scope("", reuse=tf.AUTO_REUSE):
            logits = get_logits(new_input, lengths)

        saver = tf.train.Saver()
        saver.restore(sess, restore_path)
         
        # beam_width=500
        decoded, _ = tf.nn.ctc_beam_search_decoder(logits, lengths, merge_repeated=False, beam_width=1)

        #print('logits shape', logits.shape)
        length = (len(audio)-1)//320
        r = sess.run(decoded, {new_input: [audio],
                               lengths: [length]})

        return "".join([toks[x] for x in r[0].values])

#print(classify("sample-000000.wav"))
print(classify("adv.wav"))
