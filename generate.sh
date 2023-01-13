#!/bin/bash

values=(-2.0 -1.6 -1.2 -0.8 -0.4 0 0.4 0.8 1.2 1.6 2.0)
tag_values=("n20" "n16" "n12" "n8" "n4" 0 4 8 12 16 20)
WORKDIR=/home/cristian/interfacegan
export PYTHONPATH=$PYTHONPATH:$WORKDIR

latent_space=$1
attribute="age"
echo "ATRIBUTE:" $attribute
echo "latent_space:" $latent_space
for i in "${!values[@]}"; do
  python $WORKDIR/generate.py --latent_space_type "${latent_space}" \
  --att_value "${values[i]}" \
  --attribute "${attribute}" \
  --output_path /mnt/h/"Meu Drive"/HAI/ResultsDetection/"${latent_space}"/stylegan_ffhq_"${attribute}"_"${tag_values[i]}"
done

attribute="gender"
echo "ATRIBUTE:" $attribute
echo "latent_space:" $latent_space
for i in "${!values[@]}"; do
  python $WORKDIR/generate.py --latent_space_type "${latent_space}" \
  --att_value "${values[i]}" \
  --attribute "${attribute}" \
  --output_path /mnt/h/"Meu Drive"/HAI/ResultsDetection/"${latent_space}"/stylegan_ffhq_"${attribute}"_"${tag_values[i]}"
done