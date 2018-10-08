from __future__ import division
from __future__ import print_function
import os
import time
from shutil import copyfile
from glob import glob
import tensorflow as tf
import numpy as np
import config
from collections import namedtuple
from module import *
from utils import *
from ops import *
from metrics import *
import glob
import time

# os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']

class cyclegan(object):
    def __init__(self, sess, args):
        print('/')
        self.sess = sess
        self.batch_size = args.batch_size
        self.image_size = args.fine_size  # cropped size
        self.time_step = args.time_step
        self.pitch_range = args.pitch_range
        self.input_c_dim = args.input_nc  # number of input image channels
        self.output_c_dim = args.output_nc  # number of output image channels
        self.L1_lambda = args.L1_lambda
        self.gamma = args.gamma
        self.sigma_d = args.sigma_d
        self.dataset_dir = args.dataset_dir
        self.dataset_A_dir = args.dataset_A_dir
        self.dataset_B_dir = args.dataset_B_dir
        self.sample_dir = args.sample_dir

        self.model = args.model
        self.discriminator = discriminator
        self.generator = generator_resnet
        self.criterionGAN = mae_criterion

        OPTIONS = namedtuple('OPTIONS', 'batch_size '
                                        'image_size '
                                        'gf_dim '
                                        'df_dim '
                                        'output_c_dim '
                                        'is_training')
        self.options = OPTIONS._make((args.batch_size,
                                      args.fine_size,
                                      args.ngf,
                                      args.ndf,
                                      args.output_nc,
                                      args.phase == 'train'))

        self._build_model()
        self.saver = tf.train.Saver(max_to_keep=30)
        self.now_datetime = get_now_datetime()
        self.pool = ImagePool(args.max_size)

    def _build_model(self):

        # define some placeholders
        self.real_data = tf.placeholder(tf.float32, [self.batch_size, self.time_step, self.pitch_range,
                                                     self.input_c_dim + self.output_c_dim], name='real_A_and_B')
        if self.model != 'base':
            self.real_mixed = tf.placeholder(tf.float32, [self.batch_size, self.time_step, self.pitch_range,
                                                          self.input_c_dim], name='real_A_and_B_mixed')

        self.real_A = self.real_data[:, :, :, :self.input_c_dim]
        self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

        self.gaussian_noise = tf.placeholder(tf.float32, [self.batch_size, self.time_step, self.pitch_range,
                                                          self.input_c_dim], name='gaussian_noise')
        # Generator: A - B - A
        self.fake_B = self.generator(self.real_A, self.options, False, name="generatorA2B")
        self.fake_A_ = self.generator(self.fake_B, self.options, False, name="generatorB2A")
        # Generator: B - A - B
        self.fake_A = self.generator(self.real_B, self.options, True, name="generatorB2A")
        self.fake_B_ = self.generator(self.fake_A, self.options, True, name="generatorA2B")
        # to binary
        self.real_A_binary = to_binary(self.real_A, 0.5)
        self.real_B_binary = to_binary(self.real_B, 0.5)
        self.fake_A_binary = to_binary(self.fake_A, 0.5)
        self.fake_B_binary = to_binary(self.fake_B, 0.5)
        self.fake_A__binary = to_binary(self.fake_A_, 0.5)
        self.fake_B__binary = to_binary(self.fake_B_, 0.5)

        # Discriminator: Fake
        self.DB_fake = self.discriminator(self.fake_B + self.gaussian_noise, self.options,
                                          reuse=False, name="discriminatorB")
        self.DA_fake = self.discriminator(self.fake_A + self.gaussian_noise, self.options,
                                          reuse=False, name="discriminatorA")
        # Discriminator: Real
        self.DA_real = self.discriminator(self.real_A + self.gaussian_noise, self.options, reuse=True,
                                          name="discriminatorA")
        self.DB_real = self.discriminator(self.real_B + self.gaussian_noise, self.options, reuse=True,
                                          name="discriminatorB")

        self.fake_A_sample = tf.placeholder(tf.float32, [self.batch_size, self.time_step, self.pitch_range,
                                                         self.input_c_dim], name='fake_A_sample')
        self.fake_B_sample = tf.placeholder(tf.float32, [self.batch_size, self.time_step, self.pitch_range,
                                                         self.input_c_dim], name='fake_B_sample')
        self.DA_fake_sample = self.discriminator(self.fake_A_sample + self.gaussian_noise,
                                                 self.options, reuse=True, name="discriminatorA")
        self.DB_fake_sample = self.discriminator(self.fake_B_sample + self.gaussian_noise,
                                                 self.options, reuse=True, name="discriminatorB")
        if self.model != 'base':
            # Discriminator: All
            self.DA_real_all = self.discriminator(self.real_mixed + self.gaussian_noise, self.options, reuse=False,
                                                  name="discriminatorA_all")
            self.DA_fake_sample_all = self.discriminator(self.fake_A_sample + self.gaussian_noise,
                                                         self.options, reuse=True, name="discriminatorA_all")
            self.DB_real_all = self.discriminator(self.real_mixed + self.gaussian_noise, self.options, reuse=False,
                                                  name="discriminatorB_all")
            self.DB_fake_sample_all = self.discriminator(self.fake_B_sample + self.gaussian_noise,
                                                         self.options, reuse=True, name="discriminatorB_all")
        # Generator loss
        self.cycle_loss = self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
                          + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)
        self.g_loss_a2b = self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) + self.cycle_loss
        self.g_loss_b2a = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) + self.cycle_loss
        self.g_loss = self.g_loss_a2b + self.g_loss_b2a - self.cycle_loss
        # Discriminator loss
        self.db_loss_real = self.criterionGAN(self.DB_real, tf.ones_like(self.DB_real))
        self.db_loss_fake = self.criterionGAN(self.DB_fake_sample, tf.zeros_like(self.DB_fake_sample))
        self.db_loss = (self.db_loss_real + self.db_loss_fake) / 2
        self.da_loss_real = self.criterionGAN(self.DA_real, tf.ones_like(self.DA_real))
        self.da_loss_fake = self.criterionGAN(self.DA_fake_sample, tf.zeros_like(self.DA_fake_sample))
        self.da_loss = (self.da_loss_real + self.da_loss_fake) / 2
        self.d_loss = self.da_loss + self.db_loss

        if self.model != 'base':
            self.db_all_loss_real = self.criterionGAN(self.DB_real_all, tf.ones_like(self.DB_real_all))
            self.db_all_loss_fake = self.criterionGAN(self.DB_fake_sample_all, tf.zeros_like(self.DB_fake_sample_all))
            self.db_all_loss = (self.db_all_loss_real + self.db_all_loss_fake) / 2
            self.da_all_loss_real = self.criterionGAN(self.DA_real_all, tf.ones_like(self.DA_real_all))
            self.da_all_loss_fake = self.criterionGAN(self.DA_fake_sample_all, tf.zeros_like(self.DA_fake_sample_all))
            self.da_all_loss = (self.da_all_loss_real + self.da_all_loss_fake) / 2
            self.d_all_loss = self.da_all_loss + self.db_all_loss
            self.D_loss = self.d_loss + self.gamma * self.d_all_loss

        # Define all summaries
        self.g_loss_a2b_sum = tf.summary.scalar("g_loss_a2b", self.g_loss_a2b)
        self.g_loss_b2a_sum = tf.summary.scalar("g_loss_b2a", self.g_loss_b2a)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.cycle_loss_sum = tf.summary.scalar("cycle_loss", self.cycle_loss)
        self.g_sum = tf.summary.merge([self.g_loss_a2b_sum, self.g_loss_b2a_sum, self.g_loss_sum, self.cycle_loss_sum])
        self.db_loss_sum = tf.summary.scalar("db_loss", self.db_loss)
        self.da_loss_sum = tf.summary.scalar("da_loss", self.da_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.db_loss_real_sum = tf.summary.scalar("db_loss_real", self.db_loss_real)
        self.db_loss_fake_sum = tf.summary.scalar("db_loss_fake", self.db_loss_fake)
        self.da_loss_real_sum = tf.summary.scalar("da_loss_real", self.da_loss_real)
        self.da_loss_fake_sum = tf.summary.scalar("da_loss_fake", self.da_loss_fake)
        if self.model != 'base':
            self.d_all_loss_sum = tf.summary.scalar("d_all_loss", self.d_all_loss)
            self.D_loss_sum = tf.summary.scalar("D_loss", self.d_loss)
            self.d_sum = tf.summary.merge([self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum,
                                           self.db_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum,
                                           self.d_loss_sum, self.d_all_loss_sum, self.D_loss_sum])
        else:
            self.d_sum = tf.summary.merge([self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum,
                                           self.db_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum,
                                           self.d_loss_sum])

        # Test
        self.test_A = tf.placeholder(tf.float32, [None, self.time_step, self.pitch_range,
                                                  self.input_c_dim], name='test_A')
        self.test_B = tf.placeholder(tf.float32, [None, self.time_step, self.pitch_range,
                                                  self.output_c_dim], name='test_B')
        # A - B - A
        self.testB = self.generator(self.test_A, self.options, True, name="generatorA2B")
        self.testA_ = self.generator(self.testB, self.options, True, name='generatorB2A')
        # B - A - B
        self.testA = self.generator(self.test_B, self.options, True, name="generatorB2A")
        self.testB_ = self.generator(self.testA, self.options, True, name='generatorA2B')
        # to binary
        self.test_A_binary = to_binary(self.test_A, 0.5)
        self.test_B_binary = to_binary(self.test_B, 0.5)
        self.testA_binary = to_binary(self.testA, 0.5)
        self.testB_binary = to_binary(self.testB, 0.5)
        self.testA__binary = to_binary(self.testA_, 0.5)
        self.testB__binary = to_binary(self.testB_, 0.5)

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        for var in t_vars:
            print(var.name)

    def load(self, checkpoint_dir, fromGenre, toGenre):

        print('loading model...')

        # fromGenre = request.form.get("fromGenre")
        # toGenre = request.form.get("toGenre")

        model_dir = fromGenre + '2' + toGenre

        print(model_dir)

        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        print('checkpoint_dir is ' + str(checkpoint_dir))

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print('ckpt_name is ' + ckpt_name)
            # self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, 'cyclegan.model'))
            return True
        else:
            return False

    def test(self, args, fromGenre, toGenre, millis, filename):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        '''
        if args.which_direction == 'AtoB':
            sample_files = glob('./datasets/{}/test/*.*'.format(self.dataset_A_dir))
        elif args.which_direction == 'BtoA':
            sample_files = glob('./datasets/{}/test/*.*'.format(self.dataset_B_dir))
        else:
            raise Exception('--which_direction must be AtoB or BtoA')
        sample_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
        '''

        if self.load(args.checkpoint_dir, fromGenre, toGenre):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        out_origin, out_var, out_var_cycle, in_var = (
        self.test_A_binary, self.testB_binary, self.testA__binary, self.test_A)

        '''
        test_dir_mid = os.path.join(args.test_dir, '{}2{}_{}_{}_{}/{}/mid'.format(self.dataset_A_dir,
                                                                                  self.dataset_B_dir,
                                                                                  self.now_datetime,
                                                                                  self.model,
                                                                                  self.sigma_d,
                                                                                  args.which_direction))
        if not os.path.exists(test_dir_mid):
            os.makedirs(test_dir_mid)

        test_dir_npy = os.path.join(args.test_dir, '{}2{}_{}_{}_{}/{}/npy'.format(self.dataset_A_dir,
                                                                                  self.dataset_B_dir,
                                                                                  self.now_datetime,
                                                                                  self.model,
                                                                                  self.sigma_d,
                                                                                  args.which_direction))
        if not os.path.exists(test_dir_npy):
            os.makedirs(test_dir_npy)
        '''

        # sample_path = os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/phrase_test/')
        sample_files = glob.glob('static/MIDI/' + millis + '/phrase_test/*.*')
        print(os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/phrase_test/'))
        ##sample_files = glob(os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/phrase_test/*.*'))

        print('Processing midi...')
        for idx in range(len(sample_files)):
            sample_npy = np.load(sample_files[idx]) * 1.
            sample_npy_re = sample_npy.reshape(1, sample_npy.shape[0], sample_npy.shape[1], 1)
            if not os.path.exists(os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/modified_MIDI')):
                os.makedirs(os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/modified_MIDI'))

            #midi_path_origin = os.path.join(UPLOAD_FOLDER, '{}_origin.mid'.format(idx + 1))
            midi_path_transfer = os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/modified_MIDI/' + str(idx + 1) + '.mid')
            #midi_path_cycle = os.path.join(UPLOAD_FOLDER, '{}_cycle.mid'.format(idx + 1))

            origin_midi, fake_midi, fake_midi_cycle = self.sess.run([out_origin, out_var, out_var_cycle],
                                                                    feed_dict={in_var: sample_npy_re})
            #save_midis(origin_midi, midi_path_origin)
            save_midis(fake_midi, midi_path_transfer)
            #save_midis(fake_midi_cycle, midi_path_cycle)


            #npy_path_origin = os.path.join(UPLOAD_FOLDER, 'origin')
            #npy_path_cycle = os.path.join(UPLOAD_FOLDER, 'cycle')
            if not os.path.exists(os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/modified_NPY')):
                os.makedirs(os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/modified_NPY'))
            npy_path_transfer = os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/modified_NPY')
            #if not os.path.exists(npy_path_transfer):
                #os.makedirs(npy_path_transfer)
            #if not os.path.exists(npy_path_cycle):
                #os.makedirs(npy_path_cycle)
            #np.save(os.path.join(npy_path_origin, '{}_origin.npy'.format(idx + 1)), origin_midi)
            np.save(os.path.join(npy_path_transfer, '{}_transfer.npy'.format(idx + 1)), fake_midi)
            #np.save(os.path.join(npy_path_cycle, '{}_cycle.npy'.format(idx + 1)), fake_midi_cycle)

        '''concatenate resulting numpy arrays'''

        npy_files = glob.glob(os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/modified_NPY/*.*'))

        #npy_files = [f for f in os.listdir(os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/modified_NPY'))]
        #range(len(l))

        print('printing numpy files...')
        print(npy_files)

        #allArrays = np.array([])
        count = 0
        #for x in range(len(npy_files)):
        for x in sorted(npy_files):
            x = np.load(x)
            if count == 0:
                allArrays = x
            else:
                print(x.shape)
                allArrays = np.concatenate([allArrays, x])
            count += 1

        print('printing allArrays.shape...')
        print(allArrays.shape)

        save_midis(allArrays, os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/to' + toGenre + '_' + filename + '.mid'))

# put midi files to be converted in datasets/MIDI/jazz/jazz_midi
# datasets/MIDI' + millis + '/phrase_test is where numpy arrays are saved in the end

import numpy as np
import glob
import datetime
import math
import random
import os
import shutil
import matplotlib.pyplot as plt
import pretty_midi
from pypianoroll import Multitrack, Track
import librosa.display
from utils import *

import os
import json
import errno
from pypianoroll import Multitrack, Track
import pretty_midi
import shutil

UPLOAD_FOLDER = 'static'
MIDI_FOLDER = 'static/MIDI'
test_ratio = 0.1
LAST_BAR_MODE = 'remove'


def make_sure_path_exists(path):
    """Create all intermediate-level directories if the given path does not
    exist"""
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def get_midi_path(root):
    """Return a list of paths to MIDI files in `root` (recursively)"""
    filepaths = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith('.mid'):
                filepaths.append(os.path.join(dirpath, filename))
    return filepaths


def get_midi_info(pm):
    """Return useful information from a pretty_midi.PrettyMIDI instance"""
    if pm.time_signature_changes:
        pm.time_signature_changes.sort(key=lambda x: x.time)
        first_beat_time = pm.time_signature_changes[0].time
    else:
        first_beat_time = pm.estimate_beat_start()
    tc_times, tempi = pm.get_tempo_changes()
    if len(pm.time_signature_changes) == 1:
        time_sign = '{}/{}'.format(pm.time_signature_changes[0].numerator,
                                   pm.time_signature_changes[0].denominator)
    else:
        time_sign = None
    midi_info = {
        'first_beat_time': first_beat_time,
        'num_time_signature_change': len(pm.time_signature_changes),
        'time_signature': time_sign,
        'tempo': tempi[0] if len(tc_times) == 1 else None
    }
    return midi_info


def midi_filter(midi_info):
    """Return True for qualified midi files and False for unwanted ones"""
    if midi_info['first_beat_time'] > 0.0:
        return False
    elif midi_info['num_time_signature_change'] > 1:
        return False
    elif midi_info['time_signature'] not in ['4/4']:
        return False
    return True


def get_merged(multitrack):
    """Return a `pypianoroll.Multitrack` instance with piano-rolls merged to
    five tracks (Bass, Drums, Guitar, Piano and Strings)"""
    category_list = {'Bass': [], 'Drums': [], 'Guitar': [], 'Piano': [], 'Strings': []}
    program_dict = {'Piano': 0, 'Drums': 0, 'Guitar': 24, 'Bass': 32, 'Strings': 48}
    for idx, track in enumerate(multitrack.tracks):
        if track.is_drum:
            category_list['Drums'].append(idx)
        elif track.program // 8 == 0:
            category_list['Piano'].append(idx)
        elif track.program // 8 == 3:
            category_list['Guitar'].append(idx)
        elif track.program // 8 == 4:
            category_list['Bass'].append(idx)
        else:
            category_list['Strings'].append(idx)
    tracks = []
    for key in category_list:
        if category_list[key]:
            merged = multitrack[category_list[key]].get_merged_pianoroll()
            tracks.append(Track(merged, program_dict[key], key == 'Drums', key))
        else:
            tracks.append(Track(None, program_dict[key], key == 'Drums', key))
    return Multitrack(None, tracks, multitrack.tempo, multitrack.downbeat, multitrack.beat_resolution, multitrack.name)


'''
def converter(filepath):
    """Save a multi-track piano-roll converted from a MIDI file to target
    dataset directory and update MIDI information to `midi_dict`"""
    try:
        midi_name = os.path.splitext(os.path.basename(filepath))[0]
        print('printing midi_name in the converter function...')
        print(midi_name)
        multitrack = Multitrack(beat_resolution=24, name=midi_name)
        pm = pretty_midi.PrettyMIDI(filepath)
        midi_info = get_midi_info(pm)
        multitrack.parse_pretty_midi(pm)
        merged = get_merged(multitrack)
        print('printing merged...')
        print(merged)
        converter_path = os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/converter')
        print('still ok1')
        make_sure_path_exists(converter_path)
        print('still ok2')
        merged.save(os.path.join(converter_path, midi_name + '.npz'))
        print('still ok3')
        return [midi_name, midi_info]
    except:
        print('SOMETHING WRONG')
        return None
'''


def converter(filepath, millis):
    """Save a multi-track piano-roll converted from a MIDI file to target
    dataset directory and update MIDI information to `midi_dict`"""
    midi_name = os.path.splitext(os.path.basename(filepath))[0]
    print('printing midi_name in the converter function...')
    print(midi_name)
    multitrack = Multitrack(beat_resolution=24, name=midi_name)
    pm = pretty_midi.PrettyMIDI(filepath)
    midi_info = get_midi_info(pm)
    print(pm)
    print('printing midi_info...')
    print(midi_info)
    multitrack.parse_pretty_midi(pm)
    merged = get_merged(multitrack)
    print('printing merged...')
    print(merged)
    converter_path = os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/converter')
    print('still ok1')
    make_sure_path_exists(converter_path)
    print('still ok2')
    merged.save(os.path.join(converter_path, midi_name + '.npz'))
    print('still ok3')
    return [midi_name, midi_info]


def get_bar_piano_roll(piano_roll):
    if int(piano_roll.shape[0] % 64) is not 0:
        if LAST_BAR_MODE == 'fill':
            piano_roll = np.concatenate((piano_roll, np.zeros((64 - piano_roll.shape[0] % 64, 128))), axis=0)
        elif LAST_BAR_MODE == 'remove':
            piano_roll = np.delete(piano_roll, np.s_[-int(piano_roll.shape[0] % 64):], axis=0)
    piano_roll = piano_roll.reshape(-1, 64, 128)
    return piano_roll


def to_binary(bars, threshold=0.0):
    """Turn velocity value into boolean"""
    track_is_max = tf.equal(bars, tf.reduce_max(bars, axis=-1, keep_dims=True))
    track_pass_threshold = (bars > threshold)
    out_track = tf.logical_and(track_is_max, track_pass_threshold)
    return out_track


def midiToNpy(millis, filename):
    tf.reset_default_graph()

    converter_path = os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/converter')
    cleaner_path = os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/cleaner')

    if not os.path.exists(os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/converter')):
        os.makedirs(os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/converter'))
    if not os.path.exists(os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/cleaner')):
        os.makedirs(os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/cleaner'))

    """1. divide the original set into train and test sets"""
    '''
    l = [f for f in os.listdir(MIDI_FOLDER)]
    print(len(l))
    # idx = np.random.choice(len(l), int(test_ratio * len(l)), replace=False)
    idx = np.random.choice(len(l), int(len(l)), replace=False)
    print(len(idx))
    print('?')
    '''

    """2. convert_clean.py"""
    #midi_paths = get_midi_path(MIDI_FOLDER)
    #print('printing midi_paths...')
    #print(midi_paths)
    midi_dict = {}
    #kv_pairs = [converter(midi_path, millis) for midi_path in midi_paths]
    file_location = os.path.join(MIDI_FOLDER, millis, filename)
    print('file_location is ' + file_location)
    kv_pairs = [converter(file_location, millis)]
    print('printing kv_pairs...')
    print(kv_pairs)
    for kv_pair in kv_pairs:
        if kv_pair is not None:
            midi_dict[kv_pair[0]] = kv_pair[1]
    if not os.path.exists(os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/json')):
        os.makedirs(os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/json'))
    # with open(os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/json'), 'w') as outfile:
    with open(os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/json/midis.json'), 'w') as outfile:
        print('printing midi_dict...')
        print(midi_dict)
        json.dump(midi_dict, outfile)
        print("[Done] {} file converted".format(len(midi_dict)))
    ##with open(os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/json/midis.json'), 'w') as infile:
    # midi_dict = json.load(infile)
    count = 0
    make_sure_path_exists(cleaner_path)
    midi_dict_clean = {}
    for key in midi_dict:
        if midi_filter(midi_dict[key]):
            midi_dict_clean[key] = midi_dict[key]
            count += 1
            shutil.copyfile(os.path.join(converter_path, key + '.npz'), os.path.join(cleaner_path, key + '.npz'))
    with open(os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/midis_clean.json'), 'w') as outfile:
        json.dump(midi_dict_clean, outfile)
    print("[Done] {} files out of {} have been successfully cleaned".format(count, len(midi_dict)))

    if count == 0:
        print('couldn\'t be cleaned.')
        return False

    """3. choose the clean midi from original sets"""
    if not os.path.exists(os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/cleaner_midi')):
        os.makedirs(os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/cleaner_midi'))
    l = [f for f in os.listdir(os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/cleaner'))]
    print(l)
    print(len(l))
    for i in l:
        shutil.copy(os.path.join(UPLOAD_FOLDER, 'MIDI', millis, os.path.splitext(i)[0] + '.mid'),
                    os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/cleaner_midi', os.path.splitext(i)[0] + '.mid'))

    """4. merge and crop"""
    if not os.path.exists(os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/cleaner_midi_gen')):
        os.makedirs(os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/cleaner_midi_gen'))
    if not os.path.exists(os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/cleaner_npy')):
        os.makedirs(os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/cleaner_npy'))
    l = [f for f in os.listdir(os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/cleaner_midi'))]
    print(l)
    count = 0
    for i in range(len(l)):
        try:
            multitrack = Multitrack(beat_resolution=4, name=os.path.splitext(l[i])[0])
            x = pretty_midi.PrettyMIDI(os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/cleaner_midi', l[i]))
            multitrack.parse_pretty_midi(x)
            category_list = {'Piano': [], 'Drums': []}
            program_dict = {'Piano': 0, 'Drums': 0}
            for idx, track in enumerate(multitrack.tracks):
                if track.is_drum:
                    category_list['Drums'].append(idx)
                else:
                    category_list['Piano'].append(idx)
            tracks = []
            merged = multitrack[category_list['Piano']].get_merged_pianoroll()
            print(merged.shape)
            pr = get_bar_piano_roll(merged)
            print(pr.shape)
            pr_clip = pr[:, :, 24:108]
            print(pr_clip.shape)
            if int(pr_clip.shape[0] % 4) != 0:
                pr_clip = np.delete(pr_clip, np.s_[-int(pr_clip.shape[0] % 4):], axis=0)
            pr_re = pr_clip.reshape(-1, 64, 84, 1)
            print('printing pr_re.shape...')
            print(pr_re.shape)
            save_midis(pr_re, os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/cleaner_midi_gen',
                                           os.path.splitext(l[i])[0] + '.mid'))
            np.save(os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/cleaner_npy', os.path.splitext(l[i])[0] + '.npy'),
                    pr_re)
        except:
            count += 1
            print('Wrong', l[i])
            continue
    print(count)

    """5. concatenate into a big binary numpy array file"""
    l = [f for f in os.listdir(os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/cleaner_npy'))]
    print(l)
    if len(l) > 0:
        train = np.load(os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/cleaner_npy', l[0]))
        print(train.shape, np.max(train))
        for i in range(1, len(l)):
            print(i, l[i])
            t = np.load(os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/cleaner_npy', l[i]))
            train = np.concatenate((train, t), axis=0)
        print('printing train.shape...')
        print(train.shape)
        np.save(os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/jazz_test_piano.npy'), (train > 0.0))

        """6. separate numpy array file into single phrases"""
        if not os.path.exists(os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/phrase_test')):
            os.makedirs(os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/phrase_test'))
        x = np.load(os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/jazz_test_piano.npy'))
        print(x.shape)
        count = 0
        for i in range(x.shape[0]):
            if np.max(x[i]):
                count += 1
                np.save(os.path.join(UPLOAD_FOLDER, 'MIDI/' + millis + '/phrase_test/jazz_piano_test_{}.npy'.format(i + 1)),
                        x[i])
                print(x[i].shape)
            if count == 11216:
                break
        print(count)
        return True
    else:
        print('Some other issue - though shouldn\'t ever get here')
        return False
