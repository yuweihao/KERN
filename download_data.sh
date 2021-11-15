# This script downloads the Visual Genome dataset as well as the metadata
# needed for the scene graph generation task.

mkdir data/visual_genome

wget -nc -P data/visual_genome https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
unzip -d data/visual_genome data/visual_genome/images.zip
rm data/visual_genome/images.zip

wget -nc -P data/visual_genome https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
unzip -d data/visual_genome data/visual_genome/images2.zip
rm data/visual_genome/images2.zip

find data/visual_genome/ -name \*.jpg -exec mv -t data/visual_genome {} +
find data/visual_genome/ -name VG* -exec rm -rf {} +

wget -nc -P data/stanford_filtered http://cvgl.stanford.edu/scene-graph/VG/image_data.json
wget -nc -P data/stanford_filtered http://cvgl.stanford.edu/scene-graph/dataset/VG-SGG.h5
wget -nc -P data/stanford_filtered http://cvgl.stanford.edu/scene-graph/dataset/VG-SGG-dicts.json

mkdir -p checkpoints/vgdet

wget -nc --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=11zKRr2OF5oclFL47kjFYBOxScotQzArX' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=11zKRr2OF5oclFL47kjFYBOxScotQzArX" -O checkpoints/vgdet/vg-faster-rcnn.tar && rm -rf /tmp/cookies.txt
