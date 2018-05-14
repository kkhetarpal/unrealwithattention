# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from multiprocessing import Process, Pipe
import numpy as np
import cv2
import gym

from environment import environment
from sam.spectral_residual_saliency import SpectralResidualSaliency

COMMAND_RESET     = 0
COMMAND_ACTION    = 1
COMMAND_TERMINATE = 2
GlobalImageId = 0


def preprocess_frame(observation):
  # observation shape = (210, 160, 3)
  observation = observation.astype(np.float32)
  resized_observation = cv2.resize(observation, (84, 84))
  resized_observation = resized_observation / 255.0
  return resized_observation



def mysaliency_on_frame_colormap(saliency, frame):
    height, width, _ = frame.shape
    saliency_new = np.broadcast_to(np.expand_dims(saliency, 2), (saliency.shape[0],saliency.shape[1],3))
    heatmap = cv2.applyColorMap(saliency_new, cv2.COLORMAP_JET)
    result = heatmap * 0.4 + frame * 0.5
    return result.astype('uint8')


def spectralsaliency(inputimage):
    srs = SpectralResidualSaliency(inputimage)
    map = srs.get_saliency_map()
    return map


def spectralsaliency_for_colormap(inputimage):
    srs = SpectralResidualSaliency(inputimage)
    map = srs.get_saliency_map()
    map = map * 255
    return map.astype('uint8')


def preprocess_frame_with_attention(image):
    #global GlobalImageId
    #GlobalImageId += 1
    image_salmap = spectralsaliency_for_colormap(image) #spectral saliency
    image_with_attention = mysaliency_on_frame_colormap(image_salmap, image)  # heatmap
    #outname = 'S' + str(GlobalImageId)
    #cv2.imwrite('/home/ml/kkheta2/lab/unrealwithattention/attentionframes/MZuma/' + '%s' % outname + '.png', image_with_attention)
    image_with_attention = image_with_attention.astype(np.float32)
    image_with_attention = image_with_attention / 255.0
    image_with_attention = cv2.resize(image_with_attention, (84, 84))  # reverting back to 84*84 for baseline code
    return image_with_attention



def worker(conn, env_name):
  env = gym.make(env_name)
  env.reset()
  conn.send(0)
  
  while True:
    command, arg = conn.recv()

    if command == COMMAND_RESET:
      obs = env.reset()
      #state = preprocess_frame(obs)
      state = preprocess_frame_with_attention(obs)
      conn.send(state)
    elif command == COMMAND_ACTION:
      reward = 0
      for i in range(4):
        obs, r, terminal, _ = env.step(arg)
        reward += r
        if terminal:
          break
      #state = preprocess_frame(obs)
      state = preprocess_frame_with_attention(obs)
      conn.send([state, reward, terminal])
    elif command == COMMAND_TERMINATE:
      break
    else:
      print("bad command: {}".format(command))
  env.close()
  conn.send(0)
  conn.close()


class GymEnvironment(environment.Environment):
  @staticmethod
  def get_action_size(env_name):
    env = gym.make(env_name)
    action_size = env.action_space.n
    env.close()
    return action_size
  
  def __init__(self, env_name):
    environment.Environment.__init__(self)

    self.conn, child_conn = Pipe()
    self.proc = Process(target=worker, args=(child_conn, env_name))
    self.proc.start()
    self.conn.recv()
    self.reset()

  def reset(self):
    self.conn.send([COMMAND_RESET, 0])
    self.last_state = self.conn.recv()
    
    self.last_action = 0
    self.last_reward = 0

  def stop(self):
    self.conn.send([COMMAND_TERMINATE, 0])
    ret = self.conn.recv()
    self.conn.close()
    self.proc.join()
    print("gym environment stopped")

  def process(self, action):
    self.conn.send([COMMAND_ACTION, action])
    state, reward, terminal = self.conn.recv()
    
    pixel_change = self._calc_pixel_change(state, self.last_state)
    self.last_state = state
    self.last_action = action
    self.last_reward = reward
    return state, reward, terminal, pixel_change

  def process_with_attention(self, action):
    self.conn.send([COMMAND_ACTION, action])
    state, reward, terminal = self.conn.recv()

    pixel_change = self._calc_pixel_change(state, self.last_state)
    self.last_state = state
    self.last_action = action
    self.last_reward = reward
    return state, reward, terminal, pixel_change