from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
from flask import Flask, request, redirect, url_for, render_template, jsonify, json
import io
from keras.models import load_model
from werkzeug.utils import secure_filename
import os
import cv2
import sys
import glob
import time
import sys

import argparse

from model import *

# initialize our Flask application
UPLOAD_FOLDER = 'static'
MIDI_FOLDER = 'static/MIDI'

ALLOWED_EXTENSIONS = set(['mid', 'midi'])

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MIDI_FOLDER'] = MIDI_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/uploaded', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            feedback = 'No file found.'
            return jsonify(msg=feedback, success=False)
        file = request.files['file']
        if file.filename == '':
            feedback = 'No file name.'
            return jsonify(msg=feedback, success=False)
        if request.form.get("fromGenre") == request.form.get("toGenre"):
            feedback = 'You must select different genres.'
            return jsonify(msg=feedback, success=False)
        if file and allowed_file(file.filename):

            print('file and allowed_file(file.filename)')

            millis = int(round(time.time() * 1000))
            millis = str(millis)

            filename = millis + secure_filename(file.filename)
            f = request.files['file']
            file_location = os.path.join(app.config['MIDI_FOLDER'], filename)
            f.save(file_location)

            # run analysis

            midiToNpy(millis, filename)

            tfconfig = tf.ConfigProto(allow_soft_placement=True)
            tfconfig.gpu_options.allow_growth = True
            with tf.Session(config=tfconfig) as sess:

                model = cyclegan(sess, args)
                print(request.form.get("fromGenre"))
                print(request.form.get("toGenre"))
                model.test(args, request.form.get("fromGenre"), request.form.get("toGenre"), millis, secure_filename(file.filename))

            # analysis done

            return jsonify(original_filename = secure_filename(file.filename), millis = millis, to_genre = request.form.get("toGenre"), success = True)

        else:
            feedback = 'It seems you haven\'t uploaded a MIDI file. Your file must be of type midi. If you have a file in a different audio format like MP3 or WAV, please convert it to midi first.'
            return jsonify(msg=feedback, success=False)

if __name__ == "__main__":
    # ec2 machine gives maximum recursion depth exceeded error without this - fine on local machine
    sys.setrecursionlimit(10000)

    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_dir', dest='dataset_dir', default='JAZZ2ROCK', help='path of the dataset')
    parser.add_argument('--dataset_A_dir', dest='dataset_A_dir', default='JC_J', help='path of the dataset of domain A')
    parser.add_argument('--dataset_B_dir', dest='dataset_B_dir', default='JC_C', help='path of the dataset of domain B')
    # parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='# of epoch')
    # parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=10, help='# of epoch to decay lr')
    parser.add_argument('--epoch', dest='epoch', type=int, default=10, help='# of epoch')
    parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=1, help='# of epoch to decay lr')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='# images in batch')
    parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
    parser.add_argument('--load_size', dest='load_size', type=int, default=286, help='scale images to this size')
    parser.add_argument('--fine_size', dest='fine_size', type=int, default=128, help='then crop to this size')
    parser.add_argument('--time_step', dest='time_step', type=int, default=64, help='time step of pianoroll')
    parser.add_argument('--pitch_range', dest='pitch_range', type=int, default=84, help='pitch range of pianoroll')
    parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
    parser.add_argument('--input_nc', dest='input_nc', type=int, default=1, help='# of input image channels')
    parser.add_argument('--output_nc', dest='output_nc', type=int, default=1, help='# of output image channels')
    parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
    parser.add_argument('--phase', dest='phase', default='train', help='train, test')
    parser.add_argument('--save_freq', dest='save_freq', type=int, default=1000,
                        help='save a model every save_freq iterations')
    parser.add_argument('--print_freq', dest='print_freq', type=int, default=100,
                        help='print the debug information every print_freq iterations')
    parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False,
                        help='if continue training, load the latest model: 1: true, 0: false')
    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
    parser.add_argument('--sample_dir', dest='sample_dir', default='./samples', help='sample are saved here')
    parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
    parser.add_argument('--log_dir', dest='log_dir', default='./log', help='logs are saved here')
    parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=10.0,
                        help='weight on L1 term in objective')
    parser.add_argument('--gamma', dest='gamma', type=float, default=1.0, help='weight of extra discriminators')
    parser.add_argument('--use_midi_G', dest='use_midi_G', type=bool, default=False,
                        help='select generator for midinet')
    parser.add_argument('--use_midi_D', dest='use_midi_D', type=bool, default=False,
                        help='select disciminator for midinet')
    parser.add_argument('--use_lsgan', dest='use_lsgan', type=bool, default=False, help='gan loss defined in lsgan')
    parser.add_argument('--max_size', dest='max_size', type=int, default=50,
                        help='max size of image pool, 0 means do not use image pool')
    parser.add_argument('--sigma_c', dest='sigma_c', type=float, default=1.0,
                        help='sigma of gaussian noise of classifiers')
    parser.add_argument('--sigma_d', dest='sigma_d', type=float, default=1.0,
                        help='sigma of gaussian noise of discriminators')
    parser.add_argument('--model', dest='model', default='base', help='three different models, base, partial, full')
    parser.add_argument('--type', dest='type', default='cyclegan', help='cyclegan or classifier')

    args = parser.parse_args()

    # if running on ec2 (port 80 gives permission error)
    # app.run(host = "0.0.0.0", port = 5000, debug = True, threaded = False)

    # if running on local machine
    app.run(host="0.0.0.0", port=80, debug=True, threaded=False)
