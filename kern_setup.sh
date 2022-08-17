# This script automates the KERN setup on your machine
# Author: Francesco Maria Turno


# Download the Visual Genome dataset as well as the metadata
# needed for the scene graph generation task.

if [ ! -d "data/visual_genome" ]; then
	  mkdir data/visual_genome

	    wget -nc -P data/visual_genome https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
	      unzip -d data/visual_genome data/visual_genome/images.zip
	        rm data/visual_genome/images.zip

		  wget -nc -P data/visual_genome https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
		    unzip -d data/visual_genome data/visual_genome/images2.zip
		      rm data/visual_genome/images2.zip

		        find data/visual_genome/ -name \*.jpg -exec mv -t data/visual_genome {} +
			  find data/visual_genome/ -name VG* -exec rm -rf {} +
fi

wget -nc -P data/stanford_filtered http://cvgl.stanford.edu/scene-graph/VG/image_data.json
wget -nc -P data/stanford_filtered http://cvgl.stanford.edu/scene-graph/dataset/VG-SGG.h5
wget -nc -P data/stanford_filtered http://cvgl.stanford.edu/scene-graph/dataset/VG-SGG-dicts.json

mkdir -p checkpoints/vgdet

wget -nc --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=11zKRr2OF5oclFL47kjFYBOxScotQzArX' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=11zKRr2OF5oclFL47kjFYBOxScotQzArX" -O checkpoints/vgdet/vg-faster-rcnn.tar && rm -rf /tmp/cookies.txt

mkdir -p pretrained

wget -nc --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Tvxf0OCjRKut8m_iNDgtcz_PNfIQ3Let' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Tvxf0OCjRKut8m_iNDgtcz_PNfIQ3Let" -O pretrained/kern_sgdet.pkl && rm -rf /tmp/cookies.txt

wget -nc --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1yY0bb2zPJZC3lumK1mQWSQ0NM0WMSFUK' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1yY0bb2zPJZC3lumK1mQWSQ0NM0WMSFUK" -O pretrained/kern_sgcls.pkl && rm -rf /tmp/cookies.txt

# Download checkpoints [kern_sgcls_predcls, kern_sgdet] for later use

wget -nc --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1F2WBSGRHmJD9K1LT8ImkGOCuZraood21' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1F2WBSGRHmJD9K1LT8ImkGOCuZraood21" -O checkpoints/kern_sgcls_predcls.tar && rm -rf /tmp/cookies.txt

wget -nc --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1hAx4MpMiwofABQi9H6_Jb0Qjp016JX7T' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1hAx4MpMiwofABQi9H6_Jb0Qjp016JX7T" -O checkpoints/kern_sgdet.tar && rm -rf /tmp/cookies.txt

# The script assumes that Docker and NVIDIA drivers are already installed
docker build -t cuda9 .
docker run -it -v $PWD:/kern --gpus all cuda9
