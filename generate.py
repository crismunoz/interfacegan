import numpy as np
import cv2
from deepface import DeepFace
from tqdm import tqdm
import pickle
import argparse
import os.path
import io
import PIL.Image
from generate_utils import build_generator,sample_codes
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--output_path', type=str)
parser.add_argument('--att_value', type=float, default=0)
parser.add_argument('--attribute', type=str, default=None)
parser.add_argument('--latent_space_type', type=str, default='Z')
parser.add_argument('--tag', type=float)
args = parser.parse_args()

num_samples = 1
age = 0
eyeglasses = 0
gender = 0
pose = 0
smile = 0

if args.attribute=='gender':
  gender = args.att_value
elif args.attribute=='age':
  age = args.att_value
else:
  raise f"Unknown attribute: {args.attribute}"  

#param ['pggan_celebahq','stylegan_celebahq', 'stylegan_ffhq']
model_name = "stylegan_ffhq" 

#param ['Z', 'W']
latent_space_type = args.latent_space_type
generator = build_generator(model_name)

ATTRS = ['age', 'eyeglasses', 'gender', 'pose', 'smile']
boundaries = {}
for i, attr_name in enumerate(ATTRS):
  boundary_name = f'{model_name}_{attr_name}'
  if generator.gan_type == 'stylegan' and latent_space_type == 'W':
    boundaries[attr_name] = np.load(f'boundaries/{boundary_name}_w_boundary.npy')
  else:
    boundaries[attr_name] = np.load(f'boundaries/{boundary_name}_boundary.npy')

if generator.gan_type == 'stylegan' and latent_space_type == 'W':
  synthesis_kwargs = {'latent_space_type': 'W'}
else:
  synthesis_kwargs = {}

results = []
os.makedirs(args.output_path,exist_ok=True)
for img_idx in tqdm(range(500)):
  noise_seed = np.random.randint(10000000)
  latent_codes = sample_codes(generator, num_samples, latent_space_type, noise_seed)

  new_codes = latent_codes.copy()
  for attr_name in ATTRS:
    new_codes += boundaries[attr_name] * eval(attr_name)

  new_image = generator.easy_synthesize(new_codes, **synthesis_kwargs)['image'][0]
  img_path = os.path.join(args.output_path,f'img_{args.tag}_{img_idx}.png')
  cv2.imwrite(img_path, new_image)
  try:
    result = {}
    result['metadata'] = DeepFace.analyze(img_path = img_path, prog_bar=False)
  except:
    result = {'metadata':None}
  result['path'] = img_path
  results.append(result)
pickle.dump(results, open(os.path.join(args.output_path,'metadata.pk'), 'wb'))