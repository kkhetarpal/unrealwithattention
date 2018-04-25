# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from multiprocessing import Process, Pipe
import numpy as np
import deepmind_lab

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
from sam.spectral_residual_saliency import SpectralResidualSaliency
from environment import environment

COMMAND_RESET     = 0
COMMAND_ACTION    = 1
COMMAND_TERMINATE = 2
GlobalImageId = 0


sys.stdout.flush()

def generator(b_s, phase_gen='train'):
    if phase_gen == 'train':
        images = [imgs_train_path + f for f in os.listdir(imgs_train_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        maps = [maps_train_path + f for f in os.listdir(maps_train_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        fixs = [fixs_train_path + f for f in os.listdir(fixs_train_path) if f.endswith('.mat')]
    elif phase_gen == 'val':
        images = [imgs_val_path + f for f in os.listdir(imgs_val_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        maps = [maps_val_path + f for f in os.listdir(maps_val_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        fixs = [fixs_val_path + f for f in os.listdir(fixs_val_path) if f.endswith('.mat')]
    else:
        raise NotImplementedError
    images.sort()
    maps.sort()
    fixs.sort()
    gaussian = np.zeros((b_s, nb_gaussian, shape_r_gt, shape_c_gt))
    counter = 0
    while True:
        Y = preprocess_maps(maps[counter:counter+b_s], shape_r_out, shape_c_out)
        Y_fix = preprocess_fixmaps(fixs[counter:counter + b_s], shape_r_out, shape_c_out)
        yield [preprocess_images(images[counter:counter + b_s], shape_r, shape_c), gaussian], [Y, Y, Y_fix]
        counter = (counter + b_s) % len(images)

def generator_test(b_s, imgs_test_path):
    images = [imgs_test_path + f for f in os.listdir(imgs_test_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    images.sort()

    gaussian = np.zeros((b_s, nb_gaussian, shape_r_gt, shape_c_gt))

    counter = 0
    while True:
        yield [preprocess_images(images[counter:counter + b_s], shape_r, shape_c), gaussian]
        counter = (counter + b_s) % len(images)

def generator_test_singleimage(b_s, image):
    gaussian = np.zeros((b_s, nb_gaussian, shape_r_gt, shape_c_gt))

    while True:
        yield [preprocess_image(image, shape_r, shape_c), gaussian]


def saliencyattentivemodel(inputimage):
    x = Input((3, shape_r, shape_c))
    x_maps = Input((nb_gaussian, shape_r_gt, shape_c_gt))

    # version (0 for SAM-VGG and 1 for SAM-ResNet)
    if version == 0:
        m = Model(input=[x, x_maps], output=sam_vgg([x, x_maps]))
        #print("Compiling SAM-VGG")
        m.compile(RMSprop(lr=1e-4), loss=[kl_divergence, correlation_coefficient, nss])
    elif version == 1:
        m = Model(input=[x, x_maps], output=sam_resnet([x, x_maps]))
        #print("Compiling SAM-ResNet")
        m.compile(RMSprop(lr=1e-4), loss=[kl_divergence, correlation_coefficient, nss])
    else:
        raise NotImplementedError
    # Output Folder Path
    output_folder = 'predictions/'
    nb_imgs_test = 1 #len(file_names)

    if nb_imgs_test % b_s != 0:
        print("The number of test images should be a multiple of the batch size. Please change your batch size in config.py accordingly.")
        exit()

    if version == 0:
        print("Loading SAM-VGG weights")
        m.load_weights('weights/sam-vgg_salicon_weights.pkl')
    elif version == 1:
        #print("Loading SAM-ResNet weights")
        m.load_weights('/home/ml/kkheta2/lab/unrealwithattention/sam/weights/sam-resnet_salicon_weights.pkl')

    predictions = m.predict_generator(generator_test_singleimage(b_s=b_s, image=inputimage), nb_imgs_test)[0]
    #outname = 'Attention.jpg' + str(i)
    original_image = inputimage
    res = postprocess_predictions(predictions[0], original_image.shape[0], original_image.shape[1])
    #cv2.imwrite(output_folder + '%s' % outname, res.astype(int))
    return res.astype('uint8')


def saliencyattentivemodel_modified(attention_network, inputimage):
    predictions = attention_network.predict_generator(generator_test_singleimage(b_s=b_s, image=inputimage), 1)[0]
    #outname = 'Attention.jpg' + str(i)
    original_image = inputimage
    #res = postprocess_predictions(predictions[0], original_image.shape[0], original_image.shape[1])
    res = postprocess_predictions(predictions[0], 360, 480)
    #cv2.imwrite(output_folder + '%s' % outname, res.astype(int))
    return res.astype('uint8')


#Overlay to Generate GBVS style heatmap on original image
def mysaliency_on_frame_colormap(saliency, frame):
    height, width, _ = frame.shape
    saliency_new = np.broadcast_to(np.expand_dims(saliency, 2), (saliency.shape[0],saliency.shape[1],3))
    heatmap = cv2.applyColorMap(saliency_new, cv2.COLORMAP_JET)
    result = heatmap * 0.4 + frame * 0.5
    return result.astype('uint8')


#My Saliency Function: Gives Regions Which are salient and rest all is black
def mysaliency_on_frame(saliency, frame):
    pmax = saliency.max()
    saliency_normalized = saliency/pmax
    saliency_normalized_new = np.broadcast_to(np.expand_dims(saliency_normalized, 2), (saliency.shape[0],saliency.shape[1],3))
    frame = frame * saliency_normalized_new
    return frame.astype('uint8')



def spectralsaliency(inputimage):
    srs = SpectralResidualSaliency(inputimage)
    map = srs.get_saliency_map()
    return map


def spectralsaliency_for_colormap(inputimage):
    srs = SpectralResidualSaliency(inputimage)
    map = srs.get_saliency_map()
    map = map * 255
    return map.astype('uint8')


def worker(conn, env_name):
  level = env_name
  env = deepmind_lab.Lab(
    level,
    ['RGB_INTERLACED'],
    config={
      'fps': str(60),
      'width': str(480),        #84
      'height': str(360)        #84
    })
  conn.send(0)
  
  while True:
    command, arg = conn.recv()

    if command == COMMAND_RESET:
      env.reset()
      obs = env.observations()['RGB_INTERLACED']
      conn.send(obs)
    elif command == COMMAND_ACTION:
      reward = env.step(arg, num_steps=4)
      terminal = not env.is_running()
      if not terminal:
        obs = env.observations()['RGB_INTERLACED']
      else:
        obs = 0
      conn.send([obs, reward, terminal])
    elif command == COMMAND_TERMINATE:
      break
    else:
      print("bad command: {}".format(command))
  env.close()      
  conn.send(0)
  conn.close()


def _action(*entries):
  return np.array(entries, dtype=np.intc)


class LabEnvironment(environment.Environment):
  ACTION_LIST = [
    _action(-20,   0,  0,  0, 0, 0, 0), # look_left
    _action( 20,   0,  0,  0, 0, 0, 0), # look_right
    #_action(  0,  10,  0,  0, 0, 0, 0), # look_up
    #_action(  0, -10,  0,  0, 0, 0, 0), # look_down
    _action(  0,   0, -1,  0, 0, 0, 0), # strafe_left
    _action(  0,   0,  1,  0, 0, 0, 0), # strafe_right
    _action(  0,   0,  0,  1, 0, 0, 0), # forward
    _action(  0,   0,  0, -1, 0, 0, 0), # backward
    #_action(  0,   0,  0,  0, 1, 0, 0), # fire
    #_action(  0,   0,  0,  0, 0, 1, 0), # jump
    #_action(  0,   0,  0,  0, 0, 0, 1)  # crouch
  ]

  @staticmethod
  def get_action_size(env_name):
    return len(LabEnvironment.ACTION_LIST)
  
  def __init__(self, env_name, use_attention_basenetwork=False, attention_network=None):
    environment.Environment.__init__(self)

    self.attention_network = attention_network
    self.use_attention_basenetwork = use_attention_basenetwork
    print("Attention Flag value:", self.use_attention_basenetwork)
    self.conn, child_conn = Pipe()
    self.proc = Process(target=worker, args=(child_conn, env_name))
    self.proc.start()
    self.conn.recv()
    self.reset()


  def reset(self):
    self.conn.send([COMMAND_RESET, 0])
    obs = self.conn.recv()
    if self.use_attention_basenetwork:
        self.last_state = self._preprocess_frame_with_attention(obs)
    self.last_state = self._preprocess_frame(obs)
    self.last_action = 0
    self.last_reward = 0

  def stop(self):
    self.conn.send([COMMAND_TERMINATE, 0])
    ret = self.conn.recv()
    self.conn.close()
    self.proc.join()
    print("lab environment stopped")
    
  def _preprocess_frame(self, image):
    image = cv2.resize(image, (84,84)) #reverting back to 84*84 for baseline code
    image = image.astype(np.float32)
    image = image / 255.0
    return image

  def _preprocess_frame_with_attention(self, image):
    #global GlobalImageId
    #GlobalImageId += 1
    image_salmap = spectralsaliency_for_colormap(image) #spectral saliency
    #image_salmap = saliencyattentivemodel_modified(self.attention_network, image)   #saliencyattentivemodel(image)
    #image_with_attention = mysaliency_on_frame(image_salmap, image)   #only salient parts
    image_with_attention = mysaliency_on_frame_colormap(image_salmap, image)  # heatmap
    #outname = 'S' + str(GlobalImageId)
    #cv2.imwrite('/home/ml/kkheta2/lab/unrealwithattention/attentionframes/' + '%s' % outname + '.png', image_with_attention)
    image_with_attention = image_with_attention.astype(np.float32)
    image_with_attention = image_with_attention / 255.0
    image_with_attention = cv2.resize(image_with_attention, (84, 84))  # reverting back to 84*84 for baseline code
    return image_with_attention


  def process(self, action):
    real_action = LabEnvironment.ACTION_LIST[action]

    self.conn.send([COMMAND_ACTION, real_action])
    obs, reward, terminal = self.conn.recv()

    if not terminal:
      timepreprocframe_start = time.time()
      state = self._preprocess_frame(obs)
      timepreprocframe_stop = time.time()
      timepreprocframe = timepreprocframe_stop - timepreprocframe_start
      print("Time to preprocess frame without attention: ", timepreprocframe)
      sys.stdout.flush()
    else:
      state = self.last_state
    
    pixel_change = self._calc_pixel_change(state, self.last_state)
    self.last_state = state
    self.last_action = action
    self.last_reward = reward
    return state, reward, terminal, pixel_change

  def process_with_attention(self, action):
    real_action = LabEnvironment.ACTION_LIST[action]

    self.conn.send([COMMAND_ACTION, real_action])
    obs, reward, terminal = self.conn.recv()

    if not terminal:
      state = self._preprocess_frame_with_attention(obs)
    else:
      state = self.last_state

    pixel_change = self._calc_pixel_change(state, self.last_state)
    self.last_state = state
    self.last_action = action
    self.last_reward = reward
    return state, reward, terminal, pixel_change

  def saliency_and_preprocess(self, salmap, obs):
    image_with_attention = mysaliency_on_frame(salmap, obs)  # only salient parts
    #global GlobalImageId
    #GlobalImageId += 1
    #outname = 'S' + str(GlobalImageId)
    #cv2.imwrite('/home/ml/kkheta2/lab/unrealwithattention/attentionframes/' + '%s' % outname + '.png',image_with_attention)
    image_with_attention = image_with_attention.astype(np.float32)
    image_with_attention = image_with_attention / 255.0
    state = cv2.resize(image_with_attention, (84, 84))  # reverting back to 84*84 for baseline code
    return state


  def process_with_attention_sampled(self, action, timestep):
    real_action = LabEnvironment.ACTION_LIST[action]

    self.conn.send([COMMAND_ACTION, real_action])
    obs, reward, terminal = self.conn.recv()

    if not terminal:
      if (timestep % 10 == 0):
        self.state_salmap = saliencyattentivemodel_modified(self.attention_network, obs)
        #cv2.imwrite('/home/ml/kkheta2/lab/unrealwithattention/attentionframes/' + 'Salmap' + '.png',self.state_salmap)
        state = self.saliency_and_preprocess(self.state_salmap, obs)
      else:
        state = self.saliency_and_preprocess(self.state_salmap, obs)
    else:
      state = self.last_state

    pixel_change = self._calc_pixel_change(state, self.last_state)
    self.last_state = state
    self.last_action = action
    self.last_reward = reward
    return state, reward, terminal, pixel_change
