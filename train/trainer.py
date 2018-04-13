# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time

from environment.environment import Environment
from model.model import UnrealModel
from train.experience import Experience, ExperienceFrame

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

LOG_INTERVAL = 100
PERFORMANCE_LOG_INTERVAL = 1000

# noinspection PyInterpreter,PyInterpreter
class Trainer(object):
  def __init__(self,
               thread_index,
               global_network,
               initial_learning_rate,
               learning_rate_input,
               grad_applier,
               env_type,
               env_name,
               use_pixel_change,
               use_value_replay,
               use_reward_prediction,
               use_attention_basenetwork,
               pixel_change_lambda,
               entropy_beta,
               local_t_max,
               gamma,
               gamma_pc,
               experience_history_size,
               max_global_time_step,
               device,
               attention_network=None):

    self.thread_index = thread_index
    self.learning_rate_input = learning_rate_input
    self.env_type = env_type
    self.env_name = env_name
    self.use_pixel_change = use_pixel_change
    self.use_value_replay = use_value_replay
    self.use_reward_prediction = use_reward_prediction
    self.use_attention_basenetwork = use_attention_basenetwork
    self.local_t_max = local_t_max
    self.gamma = gamma
    self.gamma_pc = gamma_pc
    self.experience_history_size = experience_history_size
    self.max_global_time_step = max_global_time_step
    self.action_size = Environment.get_action_size(env_type, env_name)
    
    self.local_network = UnrealModel(self.action_size,
                                     thread_index,
                                     use_pixel_change,
                                     use_value_replay,
                                     use_reward_prediction,
                                     use_attention_basenetwork,
                                     pixel_change_lambda,
                                     entropy_beta,
                                     device)
    self.local_network.prepare_loss()

    self.apply_gradients = grad_applier.minimize_local(self.local_network.total_loss,
                                                       global_network.get_vars(),
                                                       self.local_network.get_vars())
    
    self.sync = self.local_network.sync_from(global_network)
    self.experience = Experience(self.experience_history_size)
    self.local_t = 0
    self.initial_learning_rate = initial_learning_rate
    self.episode_reward = 0
    # For log output
    self.prev_local_t = 0

    self.attention_network = attention_network

  def prepare(self):
    self.environment = Environment.create_environment(self.env_type,
                                                      self.env_name,
                                                      use_attention_basenetwork=self.use_attention_basenetwork, #Using Attention
                                                      attention_network=self.attention_network)
  def stop(self):
    self.environment.stop()
    
  def _anneal_learning_rate(self, global_time_step):
    learning_rate = self.initial_learning_rate * (self.max_global_time_step - global_time_step) / self.max_global_time_step
    if learning_rate < 0.0:
      learning_rate = 0.0
    return learning_rate

  
  def choose_action(self, pi_values):
    return np.random.choice(range(len(pi_values)), p=pi_values)

  
  def _record_score(self, sess, summary_writer, summary_op, score_input, score, global_t):
    summary_str = sess.run(summary_op, feed_dict={
      score_input: score
    })
    summary_writer.add_summary(summary_str, global_t)
    summary_writer.flush()

    
  def set_start_time(self, start_time):
    self.start_time = start_time


  def _fill_experience(self, sess):
    """
    Fill experience buffer until buffer is full.
    """
    prev_state = self.environment.last_state
    last_action = self.environment.last_action
    last_reward = self.environment.last_reward
    last_action_reward = ExperienceFrame.concat_action_and_reward(last_action,
                                                                  self.action_size,
                                                                  last_reward)
    
    pi_, _ = self.local_network.run_base_policy_and_value(sess,
                                                          self.environment.last_state,
                                                          last_action_reward)
    action = self.choose_action(pi_)


    if self.use_attention_basenetwork:
      new_state, reward, terminal, pixel_change = self.environment.process_with_attention(action)
    else:
      new_state, reward, terminal, pixel_change = self.environment.process(action)
    
    frame = ExperienceFrame(prev_state, reward, action, terminal, pixel_change,
                            last_action, last_reward)
    self.experience.add_frame(frame)
    
    if terminal:
      self.environment.reset()
    if self.experience.is_full():
      self.environment.reset()
      print("Replay buffer filled")


  def _print_log(self, global_t):
    if (self.thread_index == 0) and (self.local_t - self.prev_local_t >= PERFORMANCE_LOG_INTERVAL):
      self.prev_local_t += PERFORMANCE_LOG_INTERVAL
      elapsed_time = time.time() - self.start_time
      steps_per_sec = global_t / elapsed_time
      print("### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
        global_t,  elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))
    

  def _process_base(self, sess, global_t, summary_writer, summary_op, score_input):
    # [Base A3C]
    states = []
    last_action_rewards = []
    actions = []
    rewards = []
    values = []

    terminal_end = False

    start_lstm_state = self.local_network.base_lstm_state_out

    #outfolder = '/home/ml/kkheta2/lab/unrealwithattention/attentionframes/'
    # t_max times loop
    for z in range(self.local_t_max):
      # Prepare last action reward
      last_action = self.environment.last_action
      last_reward = self.environment.last_reward
      last_action_reward = ExperienceFrame.concat_action_and_reward(last_action,
                                                                    self.action_size,
                                                                    last_reward)
      pi_, value_ = self.local_network.run_base_policy_and_value(sess,
                                                                   self.environment.last_state,
                                                                   last_action_reward)

      action = self.choose_action(pi_)

      states.append(self.environment.last_state)

      last_action_rewards.append(last_action_reward)
      actions.append(action)
      values.append(value_)

      if (self.thread_index == 0) and (self.local_t % LOG_INTERVAL == 0):
        print("pi={}".format(pi_))
        print(" V={}".format(value_))

      prev_state = self.environment.last_state

      # Process game
      if self.use_attention_basenetwork:
        new_state, reward, terminal, pixel_change = self.environment.process_with_attention(action)
      else:
        new_state, reward, terminal, pixel_change = self.environment.process(action)

      frame = ExperienceFrame(prev_state, reward, action, terminal, pixel_change,
                              last_action, last_reward)

      # Store to experience
      self.experience.add_frame(frame)

      self.episode_reward += reward

      rewards.append( reward )

      self.local_t += 1

      if terminal:
        terminal_end = True
        print("score={}".format(self.episode_reward))
        sys.stdout.flush()

        self._record_score(sess, summary_writer, summary_op, score_input,
                           self.episode_reward, global_t)
          
        self.episode_reward = 0
        self.environment.reset()
        self.local_network.reset_state()
        break

    R = 0.0
    if not terminal_end:
      R = self.local_network.run_base_value(sess, new_state, frame.get_action_reward(self.action_size))

    actions.reverse()
    states.reverse()
    rewards.reverse()
    values.reverse()

    batch_si = []
    batch_a = []
    batch_adv = []
    batch_R = []

    for(ai, ri, si, Vi) in zip(actions, rewards, states, values):
      R = ri + self.gamma * R
      adv = R - Vi
      a = np.zeros([self.action_size])
      a[ai] = 1.0

      batch_si.append(si)
      batch_a.append(a)
      batch_adv.append(adv)
      batch_R.append(R)

    batch_si.reverse()
    batch_a.reverse()
    batch_adv.reverse()
    batch_R.reverse()
    
    return batch_si, last_action_rewards, batch_a, batch_adv, batch_R, start_lstm_state

  
  def _process_pc(self, sess):
    # [pixel change]
    # Sample 20+1 frame (+1 for last next state)
    pc_experience_frames = self.experience.sample_sequence(self.local_t_max+1)
    # Reverse sequence to calculate from the last
    pc_experience_frames.reverse()

    batch_pc_si = []
    batch_pc_a = []
    batch_pc_R = []
    batch_pc_last_action_reward = []
    
    pc_R = np.zeros([20,20], dtype=np.float32)
    if not pc_experience_frames[1].terminal:
      pc_R = self.local_network.run_pc_q_max(sess,
                                             pc_experience_frames[0].state,
                                             pc_experience_frames[0].get_last_action_reward(self.action_size))


    for frame in pc_experience_frames[1:]:
      pc_R = frame.pixel_change + self.gamma_pc * pc_R
      a = np.zeros([self.action_size])
      a[frame.action] = 1.0
      last_action_reward = frame.get_last_action_reward(self.action_size)
      
      batch_pc_si.append(frame.state)
      batch_pc_a.append(a)
      batch_pc_R.append(pc_R)
      batch_pc_last_action_reward.append(last_action_reward)

    batch_pc_si.reverse()
    batch_pc_a.reverse()
    batch_pc_R.reverse()
    batch_pc_last_action_reward.reverse()
    
    return batch_pc_si, batch_pc_last_action_reward, batch_pc_a, batch_pc_R

  
  def _process_vr(self, sess):
    # [Value replay]
    # Sample 20+1 frame (+1 for last next state)
    vr_experience_frames = self.experience.sample_sequence(self.local_t_max+1)
    # Reverse sequence to calculate from the last
    vr_experience_frames.reverse()

    batch_vr_si = []
    batch_vr_R = []
    batch_vr_last_action_reward = []

    vr_R = 0.0
    if not vr_experience_frames[1].terminal:
      vr_R = self.local_network.run_vr_value(sess,
                                             vr_experience_frames[0].state,
                                             vr_experience_frames[0].get_last_action_reward(self.action_size))
    
    # t_max times loop
    for frame in vr_experience_frames[1:]:
      vr_R = frame.reward + self.gamma * vr_R
      batch_vr_si.append(frame.state)
      batch_vr_R.append(vr_R)
      last_action_reward = frame.get_last_action_reward(self.action_size)
      batch_vr_last_action_reward.append(last_action_reward)

    batch_vr_si.reverse()
    batch_vr_R.reverse()
    batch_vr_last_action_reward.reverse()

    return batch_vr_si, batch_vr_last_action_reward, batch_vr_R

  
  def _process_rp(self):
    # [Reward prediction]
    rp_experience_frames = self.experience.sample_rp_sequence()
    # 4 frames

    batch_rp_si = []
    batch_rp_c = []
    
    for i in range(3):
      batch_rp_si.append(rp_experience_frames[i].state)

    # one hot vector for target reward
    r = rp_experience_frames[3].reward
    rp_c = [0.0, 0.0, 0.0]
    if r == 0:
      rp_c[0] = 1.0 # zero
    elif r > 0:
      rp_c[1] = 1.0 # positive
    else:
      rp_c[2] = 1.0 # negative
    batch_rp_c.append(rp_c)
    return batch_rp_si, batch_rp_c
  
  
  def process(self, sess, global_t, summary_writer, summary_op, score_input):
    # Fill experience replay buffer
    if not self.experience.is_full():
      self._fill_experience(sess)
      return 0

    start_local_t = self.local_t

    cur_learning_rate = self._anneal_learning_rate(global_t)

    # Copy weights from shared to local
    sess.run( self.sync )

    # [Base]
    #timebaseprocess_start = time.time()
    batch_si, batch_last_action_rewards, batch_a, batch_adv, batch_R, start_lstm_state = \
          self._process_base(sess,
                             global_t,
                             summary_writer,
                             summary_op,
                             score_input)
    #timebaseprocess_stop = time.time()
    #timebaseprocess = timebaseprocess_stop - timebaseprocess_start
    #print("Time for base A3C process: ", timebaseprocess)
    #sys.stdout.flush()

    feed_dict = {
      self.local_network.base_input: batch_si,
      self.local_network.base_last_action_reward_input: batch_last_action_rewards,
      self.local_network.base_a: batch_a,
      self.local_network.base_adv: batch_adv,
      self.local_network.base_r: batch_R,
      self.local_network.base_initial_lstm_state: start_lstm_state,
      # [common]
      self.learning_rate_input: cur_learning_rate
    }

    # [Pixel change]
    if self.use_pixel_change:
      #timePCprocess_start = time.time()
      batch_pc_si, batch_pc_last_action_reward, batch_pc_a, batch_pc_R = self._process_pc(sess)
      #timePCprocess_stop = time.time()
      #timePCprocess = timePCprocess_stop - timePCprocess_start
      #print("Time for base Pixel Change process: ", timePCprocess)
      #sys.stdout.flush()

      pc_feed_dict = {
        self.local_network.pc_input: batch_pc_si,
        self.local_network.pc_last_action_reward_input: batch_pc_last_action_reward,
        self.local_network.pc_a: batch_pc_a,
        self.local_network.pc_r: batch_pc_R
      }
      feed_dict.update(pc_feed_dict)

    # [Value replay]
    if self.use_value_replay:
      #timeVRprocess_start = time.time()
      batch_vr_si, batch_vr_last_action_reward, batch_vr_R = self._process_vr(sess)
      #timeVRprocess_stop = time.time()
      #timeVRprocess = timeVRprocess_stop - timeVRprocess_start
      #print("Time for base Value Replay process: ", timeVRprocess)
      #sys.stdout.flush()
      
      vr_feed_dict = {
        self.local_network.vr_input: batch_vr_si,
        self.local_network.vr_last_action_reward_input : batch_vr_last_action_reward,
        self.local_network.vr_r: batch_vr_R
      }
      feed_dict.update(vr_feed_dict)

    # [Reward prediction]
    if self.use_reward_prediction:
      #timeRPprocess_start = time.time()
      batch_rp_si, batch_rp_c = self._process_rp()
      #timeRPprocess_stop = time.time()
      #timeRPprocess = timeRPprocess_stop - timeRPprocess_start
      #print("Time for base Reward Prediction process: ", timeRPprocess)
      #sys.stdout.flush()
      rp_feed_dict = {
        self.local_network.rp_input: batch_rp_si,
        self.local_network.rp_c_target: batch_rp_c
      }
      feed_dict.update(rp_feed_dict)

    # Calculate gradients and copy them to global network.
    sess.run( self.apply_gradients, feed_dict=feed_dict )
    
    self._print_log(global_t)
    
    # Return advanced local step size
    diff_local_t = self.local_t - start_local_t
    return diff_local_t