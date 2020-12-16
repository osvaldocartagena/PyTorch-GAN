#!/usr/bin/env python

"""
Este programa permite mover al Duckiebot dentro del simulador
usando el teclado.
"""

import sys
import argparse
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
import numpy as np
import cv2

from PIL import Image

### Imports from cyclegan
import sys

import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time


import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

#Lugar donde tengo el repo
sys.path.append('C:/Users/HP7/Desktop/PyTorch-GAN/implementations/cyclegan')
from models import *
from datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch
###


MAPA='4way'
# Se leen los argumentos de entrada
parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default="Duckietown-udem1-v1")
parser.add_argument('--map-name', default=MAPA)
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
parser.add_argument('--seed', default=1, type=int, help='seed')
args = parser.parse_args()

# Definición del environment
if args.env_name and args.env_name.find('Duckietown') != -1:
    env = DuckietownEnv(
        seed = args.seed,
        map_name = args.map_name,
        draw_curve = args.draw_curve,
        draw_bbox = args.draw_bbox,
        domain_rand = args.domain_rand,
        frame_skip = args.frame_skip,
        distortion = args.distortion,
    )
else:
    env = gym.make(args.env_name)

# Se reinicia el environment
env.reset()

#valores por default
Tensor =torch.Tensor
channels=3
img_height=256
img_width=256
input_shape = (channels, img_height, img_width)
n_residual_blocks=9
G_AB = GeneratorResNet(input_shape, n_residual_blocks)
G_BA = GeneratorResNet(input_shape, n_residual_blocks)

dataset_name='DuckieGan'
epoch=9 #numero del modelo mas reciente?
#Tuve que decargar la carpeta save_models (donde estan los modelos) desde el drive
G_AB.load_state_dict(torch.load("C:/Users/HP7/Desktop/PyTorch-GAN/implementations/cyclegan/saved_models/%s/G_AB_%d.pth" % (dataset_name, epoch-1),map_location='cpu'))
G_BA.load_state_dict(torch.load("C:/Users/HP7/Desktop/PyTorch-GAN/implementations/cyclegan/saved_models/%s/G_BA_%d.pth" % (dataset_name, epoch-1),map_location='cpu'))



k=1
while True:

    # Captura la tecla que está siendo apretada y almacena su valor en key
    key = cv2.waitKey(30)

    # Si la tecla es Esc, se sale del loop y termina el programa
    if key == 27:
        break

    # La acción de Duckiebot consiste en dos valores:
    # velocidad lineal y velocidad de giro
    # En este caso, ambas velocidades son 0 (acción por defecto)
    action = np.array([0,0])

    # Definir acción en base a la tecla apretada

    # Esto es avanzar recto hacia adelante al apretar la tecla w
    if key == ord('w'):
        action = np.array([1, 0.0])

        
    # Esto es retroceder apretando s
    if key == ord('s'):
        action = np.array([-1, 0.0])


    # Esto es girar a la derecha
    if key == ord('d'):
        action = np.array([0.0, -1])


    # Esto es girar a la izquierda
    if key == ord('a'):
        action = np.array([0.0, 1])


    



    # Se ejecuta la acción definida anteriormente y se retorna la observación (obs),
    # la evaluación (reward), etc
    obs, reward, done, info = env.step(action)
    # obs consiste en un imagen de 640 x 480 x 3
    #print(obs)
    ### Desde cyclegan
    
    #intentos fallidos de la tranformación
    '''transforms_ = [
        transforms.Resize((img_height, img_width), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]'''

    '''transforms_ = transforms.Compose([
        transforms.Resize((img_height, img_width), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])'''

    # Image transformations
    transforms_ = transforms.Compose([
        transforms.Resize((256, 256), 3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    #array a imagen
    im=Image.fromarray(obs)

    #otro intento fallido?
    '''imgR=cv2.resize(obs,(256,256))
    
    imt=torch.from_numpy(imgR)
    
    imt = imt.unsqueeze(0)'''
    
    #aplica la tranformacinn a la imagen
    imt=transforms_(im)
    #G_AB.eval()
    G_BA.eval()
    real_B = Variable(imt) #creo que esto da lo mismo
    fake_A = G_BA(imt) ####aqui es donde falla

    
    # done significa que el Duckiebot chocó con un objeto o se salió del camino
    if done:
        print('done!')
        # En ese caso se reinicia el simulador
        env.reset()

    # Se muestra en una ventana llamada "patos" la observación del simulador
    cv2.imshow("patos", cv2.cvtColor(fake_A, cv2.COLOR_RGB2BGR))


    
    #esto era para obtener los frames
    '''f=2
    if k%f==0:
        cv2.imwrite('C:/Users/HP7/Desktop/Cosas de la U/img sim/'+MAPA+'-'+str(k//f)+'.jpg',cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
    k=k+1
    if (k/f)%50==0: print(k//f)
    if k//f==1251:
        print('listo')
        break'''

# Se cierra el environment y termina el programa
env.close()
