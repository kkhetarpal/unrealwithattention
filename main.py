# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import threading

import signal
import math
import os
import time

from environment.environment import Environment
from model.model import UnrealModel
from train.trainer import Trainer
from train.rmsprop_applier import RMSPropApplier
from options import get_options


#---------------------------------------------------#
#---------------------------------------------------#
#SAM imports
#---------------------------------------------------#
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import Input
from keras.models import Model
import os, cv2, sys, time
from sam.config import *
from sam.utilities import preprocess_image, preprocess_images, preprocess_maps, preprocess_fixmaps, postprocess_predictions
from sam.models import sam_vgg, sam_resnet, kl_divergence, correlation_coefficient, nss


USE_GPU = True  # To use GPU, set True

# get command line args
flags = get_options("training")

def log_uniform(lo, hi, rate):
  log_lo = math.log(lo)
  log_hi = math.log(hi)
  v = log_lo * (1-rate) + log_hi * rate
  return math.exp(v)


class Application(object):
  def __init__(self):
    pass
  
  def train_function(self, parallel_index, preparing):
    """ Train each environment. """
    
    trainer = self.trainers[parallel_index]
    if preparing:
      trainer.prepare()
    
    # set start_time
    trainer.set_start_time(self.start_time)
  
    while True:
      if self.stop_requested:
        break
      if self.terminate_reqested:
        trainer.stop()
        break
      if self.global_t > flags.max_time_step:
        trainer.stop()
        break
      if parallel_index == 0 and self.global_t > self.next_save_steps:
        # Save checkpoint
        self.save()

      #Each env calls its own process #Process has sub tasks called within
      diff_global_t = trainer.process(self.sess,
                                      self.global_t,
                                      self.summary_writer,
                                      self.summary_op,
                                      self.score_input)
      self.global_t += diff_global_t

  def run(self):
    device = "/cpu:0"
    if USE_GPU:
      device = "/gpu:1"
    
    initial_learning_rate = log_uniform(flags.initial_alpha_low,
                                        flags.initial_alpha_high,
                                        flags.initial_alpha_log_rate)
    
    self.global_t = 0
    
    self.stop_requested = False
    self.terminate_reqested = False
    
    action_size = Environment.get_action_size(flags.env_type,
                                              flags.env_name)

    print("Device value:",device)
    self.global_network = UnrealModel(action_size,
                                      -1,
                                      flags.use_pixel_change,
                                      flags.use_value_replay,
                                      flags.use_reward_prediction,
                                      flags.use_attention_basenetwork,
                                      flags.pixel_change_lambda,
                                      flags.entropy_beta,
                                      device)
    self.trainers = []
    
    learning_rate_input = tf.placeholder("float")
    
    grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
                                  decay = flags.rmsp_alpha,
                                  momentum = 0.0,
                                  epsilon = flags.rmsp_epsilon,
                                  clip_norm = flags.grad_norm_clip,
                                  device = device)

    attention_network = None
    if flags.use_attention_basenetwork:
        timeSAMcompilestart = time.time()
        x = Input((3, shape_r, shape_c))
        x_maps = Input((nb_gaussian, shape_r_gt, shape_c_gt))
        # version (0 for SAM-VGG and 1 for SAM-ResNet)
        if version == 0:
            attention_network = Model(input=[x, x_maps], output=sam_vgg([x, x_maps]))
            # print("Compiling SAM-VGG")
            attention_network.compile(RMSprop(lr=1e-4), loss=[kl_divergence, correlation_coefficient, nss])
        elif version == 1:
            attention_network = Model(input=[x, x_maps], output=sam_resnet([x, x_maps]))
            print("Compiling SAM-ResNet")
            attention_network.compile(RMSprop(lr=1e-4), loss=[kl_divergence, correlation_coefficient, nss])
        else:
            raise NotImplementedError
        if version == 0:
            print("Loading SAM-VGG weights")
            attention_network.load_weights('weights/sam-vgg_salicon_weights.pkl')
        elif version == 1:
            print("Loading SAM-ResNet weights")
            attention_network.load_weights('/home/ml/kkheta2/lab/unrealwithattention/sam/weights/sam-resnet_salicon_weights.pkl')
            sys.stdout.flush()
        timeSAMcompilestop = time.time()
        timeSAMcompile = timeSAMcompilestop - timeSAMcompilestart
        print("Time fo SAM compile: ",timeSAMcompile)
    for i in range(flags.parallel_size):             #Trainer creates a UnrealModel in init
      #print("Adhed: " + str(flags.use_attention_basenetwork))
      trainer = Trainer(i,
                        self.global_network,
                        initial_learning_rate,
                        learning_rate_input,
                        grad_applier,
                        flags.env_type,
                        flags.env_name,
                        flags.use_pixel_change,
                        flags.use_value_replay,
                        flags.use_reward_prediction,
                        flags.use_attention_basenetwork,
                        flags.pixel_change_lambda,
                        flags.entropy_beta,
                        flags.local_t_max,
                        flags.gamma,
                        flags.gamma_pc,
                        flags.experience_history_size,
                        flags.max_time_step,
                        device,
                        attention_network=attention_network)

      self.trainers.append(trainer)
    
    # prepare session
    config = tf.ConfigProto(log_device_placement=False,
                            allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    self.sess = tf.Session(config=config)
    
    self.sess.run(tf.global_variables_initializer())
    
    # summary for tensorboard
    self.score_input = tf.placeholder(tf.int32)
    tf.summary.scalar("score", self.score_input)
    
    self.summary_op = tf.summary.merge_all()
    self.summary_writer = tf.summary.FileWriter(flags.log_file,
                                                self.sess.graph)
    
    # init or load checkpoint with saver
    self.saver = tf.train.Saver(self.global_network.get_vars())
    
    checkpoint = tf.train.get_checkpoint_state(flags.checkpoint_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
      self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
      print("checkpoint loaded:", checkpoint.model_checkpoint_path)
      tokens = checkpoint.model_checkpoint_path.split("-")
      # set global step
      self.global_t = int(tokens[1])
      print(">>> global step set: ", self.global_t)
      # set wall time
      wall_t_fname = flags.checkpoint_dir + '/' + 'wall_t.' + str(self.global_t)
      with open(wall_t_fname, 'r') as f:
        self.wall_t = float(f.read())
        self.next_save_steps = (self.global_t + flags.save_interval_step) // flags.save_interval_step * flags.save_interval_step
        
    else:
      print("Could not find old checkpoint")
      # set wall time
      self.wall_t = 0.0
      self.next_save_steps = flags.save_interval_step
  
    # run training threads  ## Each Env is Running Here Parallel
    self.train_threads = []
    for i in range(flags.parallel_size):
      self.train_threads.append(threading.Thread(target=self.train_function, args=(i,True)))
      
    signal.signal(signal.SIGINT, self.signal_handler)
  
    # set start time
    self.start_time = time.time() - self.wall_t
  
    for t in self.train_threads:
      t.start()
  
    print('Press Ctrl+C to stop')
    signal.pause()

  def save(self):
    """ Save checkpoint. 
    Called from therad-0.
    """
    self.stop_requested = True
  
    # Wait for all other threads to stop
    for (i, t) in enumerate(self.train_threads):
      if i != 0:
        t.join()
  
    # Save
    if not os.path.exists(flags.checkpoint_dir):
      os.mkdir(flags.checkpoint_dir)
  
    # Write wall time
    wall_t = time.time() - self.start_time
    wall_t_fname = flags.checkpoint_dir + '/' + 'wall_t.' + str(self.global_t)
    with open(wall_t_fname, 'w') as f:
      f.write(str(wall_t))
  
    print('Start saving.')
    self.saver.save(self.sess,
                    flags.checkpoint_dir + '/' + 'checkpoint',
                    global_step = self.global_t)
    print('End saving.')  
    
    self.stop_requested = False
    self.next_save_steps += flags.save_interval_step
  
    # Restart other threads
    for i in range(flags.parallel_size):
      if i != 0:
        thread = threading.Thread(target=self.train_function, args=(i,False))
        self.train_threads[i] = thread
        thread.start()
    
  def signal_handler(self, signal, frame):
    print('You pressed Ctrl+C!')
    self.terminate_reqested = True

def main(argv):
  app = Application()
  app.run()

if __name__ == '__main__':
  tf.app.run()
