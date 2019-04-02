"""
Configuration file!
"""
import os
from argparse import ArgumentParser
import numpy as np

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(ROOT_PATH, 'data')

def path(fn):
    return os.path.join(DATA_PATH, fn)

def stanford_path(fn):
    return os.path.join(DATA_PATH, 'stanford_filtered', fn)

# =============================================================================
# Update these with where your data is stored ~~~~~~~~~~~~~~~~~~~~~~~~~

VG_IMAGES = '/home/yuweihao/data/visual-genome/VGdata'
RCNN_CHECKPOINT_FN = path('faster_rcnn_500k.h5')

IM_DATA_FN = stanford_path('image_data.json')
VG_SGG_FN = stanford_path('VG-SGG.h5')
VG_SGG_DICT_FN = stanford_path('VG-SGG-dicts.json')
PROPOSAL_FN = stanford_path('proposals.h5')

# =============================================================================
# =============================================================================


MODES = ('sgdet', 'sgcls', 'predcls')

BOX_SCALE = 1024  # Scale at which we have the boxes
IM_SCALE = 592      # Our images will be resized to this res without padding

# Proposal assignments
BG_THRESH_HI = 0.5
BG_THRESH_LO = 0.0

RPN_POSITIVE_OVERLAP = 0.7
# IOU < thresh: negative example
RPN_NEGATIVE_OVERLAP = 0.3

# Max number of foreground examples
RPN_FG_FRACTION = 0.5
FG_FRACTION = 0.25
# Total number of examples
RPN_BATCHSIZE = 256
ROIS_PER_IMG = 256
REL_FG_FRACTION = 0.25
RELS_PER_IMG = 256

RELS_PER_IMG_REFINE = 64

BATCHNORM_MOMENTUM = 0.01
ANCHOR_SIZE = 16

ANCHOR_RATIOS = (0.23232838, 0.63365731, 1.28478321, 3.15089189) #(0.5, 1, 2)
ANCHOR_SCALES = (2.22152954, 4.12315647, 7.21692515, 12.60263013, 22.7102731) #(4, 8, 16, 32)

class ModelConfig(object):
    """Wrapper class for model hyperparameters."""
    def __init__(self):
        """
        Defaults
        """
        self.ckpt = None
        self.save_dir = None
        self.lr = None
        self.batch_size = None
        self.val_size = None
        self.l2 = None
        self.adamwd = None
        self.clip = None
        self.num_gpus = None
        self.num_workers = None
        self.print_interval = None
        self.mode = None
        self.test = False
        self.adam = False
        self.cache = None
        self.use_proposals=False
        self.use_resnet=False
        self.num_epochs=None
        self.pooling_dim = None

        self.use_ggnn_obj = False
        self.ggnn_obj_time_step_num = None
        self.ggnn_obj_hidden_dim = None
        self.ggnn_obj_output_dim = None
        self.use_obj_knowledge = False
        self.obj_knowledge = None

        self.use_ggnn_rel = False
        self.ggnn_rel_time_step_num = None
        self.ggnn_rel_hidden_dim = None
        self.ggnn_rel_output_dim = None
        self.use_rel_knowledge = False
        self.rel_knowledge = None

        self.tb_log_dir = None
        self.save_rel_recall = None

        self.parser = self.setup_parser()
        self.args = vars(self.parser.parse_args())

        print("~~~~~~~~ Hyperparameters used: ~~~~~~~")
        for x, y in self.args.items():
            print("{} : {}".format(x, y))

        self.__dict__.update(self.args)

        if len(self.ckpt) != 0:
            self.ckpt = os.path.join(ROOT_PATH, self.ckpt)
        else:
            self.ckpt = None

        if len(self.cache) != 0:
            if len(self.cache.split('/')) > 1:
                file_len = len(self.cache.split('/')[-1])
                cache_dir = self.cache[:-file_len]
                cache_dir = os.path.join(ROOT_PATH, cache_dir)
                if not os.path.exists(cache_dir):
                    os.mkdir(cache_dir)
            self.cache = os.path.join(ROOT_PATH, self.cache)
        else:
            self.cache = None

        if len(self.save_dir) == 0:
            self.save_dir = None
        else:
            self.save_dir = os.path.join(ROOT_PATH, self.save_dir)
            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)

        if len(self.tb_log_dir) != 0:
            self.tb_log_dir = os.path.join(ROOT_PATH, self.tb_log_dir)
            if not os.path.exists(self.tb_log_dir):
                os.makedirs(self.tb_log_dir) # help make multi depth directories, such as summaries/kern_predcls
        else:
            self.tb_log_dir = None

        if len(self.save_rel_recall) != 0:
            if len(self.save_rel_recall.split('/')) > 1:
                file_len = len(self.save_rel_recall.split('/')[-1])
                save_rel_recall_dir = self.save_rel_recall[:-file_len]
                save_rel_recall_dir = os.path.join(ROOT_PATH, save_rel_recall_dir)
                if not os.path.exists(save_rel_recall_dir):
                    os.mkdir(save_rel_recall_dir)
            self.save_rel_recall = os.path.join(ROOT_PATH, self.save_rel_recall)
        else:
            self.save_rel_recall = None
        

        assert self.val_size >= 0

        if self.mode not in MODES:
            raise ValueError("Invalid mode: mode must be in {}".format(MODES))


        if self.ckpt is not None and not os.path.exists(self.ckpt):
            raise ValueError("Ckpt file ({}) doesnt exist".format(self.ckpt))

    def setup_parser(self):
        """
        Sets up an argument parser
        :return:
        """
        parser = ArgumentParser(description='training code')


        parser.add_argument('-ckpt', dest='ckpt', help='Filename to load from', type=str, default='')
        parser.add_argument('-save_dir', dest='save_dir',
                            help='Directory to save things to, such as checkpoints/save', default='', type=str)

        parser.add_argument('-ngpu', dest='num_gpus', help='cuantos GPUs tienes', type=int, default=1)
        parser.add_argument('-nwork', dest='num_workers', help='num processes to use as workers', type=int, default=1)

        parser.add_argument('-lr', dest='lr', help='learning rate', type=float, default=1e-3)

        parser.add_argument('-b', dest='batch_size', help='batch size per GPU',type=int, default=2)
        parser.add_argument('-val_size', dest='val_size', help='val size to use (if 0 we wont use val)', type=int, default=5000)

        parser.add_argument('-l2', dest='l2', help='weight decay of SGD', type=float, default=1e-4)
        parser.add_argument('-adamwd', dest='adamwd', help='weight decay of adam', type=float, default=0.0)

        parser.add_argument('-clip', dest='clip', help='gradients will be clipped to have norm less than this', type=float, default=5.0)
        parser.add_argument('-p', dest='print_interval', help='print during training', type=int,
                            default=100)
        parser.add_argument('-m', dest='mode', help='mode in {sgdet, sgcls, predcls}', type=str, default='sgdet')


        parser.add_argument('-cache', dest='cache', help='where should we cache predictions', type=str,
                            default='')

        parser.add_argument('-adam', dest='adam', help='use adam', action='store_true')
        parser.add_argument('-test', dest='test', help='test set', action='store_true')

        parser.add_argument('-nepoch', dest='num_epochs', help='Number of epochs to train the model for',type=int, default=50)
        parser.add_argument('-resnet', dest='use_resnet', help='use resnet instead of VGG', action='store_true')
        parser.add_argument('-proposals', dest='use_proposals', help='Use Xu et als proposals', action='store_true')
        parser.add_argument('-pooling_dim', dest='pooling_dim', help='Dimension of pooling', type=int, default=4096)


        parser.add_argument('-use_ggnn_obj', dest='use_ggnn_obj', help='use GGNN_obj module', action='store_true')
        parser.add_argument('-ggnn_obj_time_step_num', dest='ggnn_obj_time_step_num', help='time step number of GGNN_obj', type=int, default=3)
        parser.add_argument('-ggnn_obj_hidden_dim', dest='ggnn_obj_hidden_dim', help='node hidden state dimension of GGNN_obj', type=int, default=512)
        parser.add_argument('-ggnn_obj_output_dim', dest='ggnn_obj_output_dim', help='node output feature dimension of GGNN_obj', type=int, default=512)
        parser.add_argument('-use_obj_knowledge', dest='use_obj_knowledge', help='use object cooccurrence knowledge', action='store_true')
        parser.add_argument('-obj_knowledge', dest='obj_knowledge', help='Filename to load matrix of object cooccurrence knowledge', type=str, default='')


        parser.add_argument('-use_ggnn_rel', dest='use_ggnn_rel', help='use GGNN_rel module', action='store_true')
        parser.add_argument('-ggnn_rel_time_step_num', dest='ggnn_rel_time_step_num', help='time step number of GGNN_rel', type=int, default=3)
        parser.add_argument('-ggnn_rel_hidden_dim', dest='ggnn_rel_hidden_dim', help='node hidden state dimension of GGNN_rel', type=int, default=512)
        parser.add_argument('-ggnn_rel_output_dim', dest='ggnn_rel_output_dim', help='node output feature dimension of GGNN_rel', type=int, default=512)
        parser.add_argument('-use_rel_knowledge', dest='use_rel_knowledge', help='use cooccurrence knowledge of object pairs and relationships', action='store_true')
        parser.add_argument('-rel_knowledge', dest='rel_knowledge', help='Filename to load matrix of cooccurrence knowledge of object pairs and relationships', type=str, default='')


        parser.add_argument('-tb_log_dir', dest='tb_log_dir', help='dir to save tensorboard summaries', type=str, default='')
        parser.add_argument('-save_rel_recall', dest='save_rel_recall', help='dir to save relationship results', type=str, default='')

        return parser
