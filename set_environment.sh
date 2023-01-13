#!/bin/bash

pip install Ninja
pip install deepface
git clone https://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@github.com/holistic-ai/face_generator.git
git clone https://github.com/saic-mdal/CIPS.git

wget https://www.dropbox.com/s/t74z87pk3cf8ny7/pggan_celebahq.pth?dl=1 -O models/pretrain/pggan_celebahq.pth --quiet
wget https://www.dropbox.com/s/nmo2g3u0qt7x70m/stylegan_celebahq.pth?dl=1 -O models/pretrain/stylegan_celebahq.pth --quiet
wget https://www.dropbox.com/s/qyv37eaobnow7fu/stylegan_ffhq.pth?dl=1 -O models/pretrain/stylegan_ffhq.pth --quiet