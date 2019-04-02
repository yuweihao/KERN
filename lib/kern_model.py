"""
KERN models
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence
from lib.resnet import resnet_l4
from config import BATCHNORM_MOMENTUM
from lib.fpn.nms.functions.nms import apply_nms

from lib.fpn.box_utils import bbox_overlaps, center_size
from lib.get_union_boxes import UnionBoxesAndFeats
from lib.fpn.proposal_assignments.rel_assignments import rel_assignments
from lib.object_detector import ObjectDetector, gather_res, load_vgg
from lib.pytorch_misc import transpose_packed_sequence_inds, to_onehot, arange, enumerate_by_image, diagonal_inds, Flattener
from lib.surgery import filter_dets
from lib.fpn.roi_align.functions.roi_align import RoIAlignFunction
from lib.ggnn import GGNNObj, GGNNRel


MODES = ('sgdet', 'sgcls', 'predcls')



class GGNNObjReason(nn.Module):
    """
    Module for object classification
    """
    def __init__(self, mode='sgdet', num_obj_cls=151, obj_dim=4096,   
                 time_step_num=3, hidden_dim=512, output_dim=512,
                 use_knowledge=True, knowledge_matrix=''):
        super(GGNNObjReason, self).__init__()
        assert mode in MODES
        self.mode = mode
        self.num_obj_cls = num_obj_cls
        self.obj_proj = nn.Linear(obj_dim, hidden_dim)
        self.ggnn_obj = GGNNObj(num_obj_cls=num_obj_cls, time_step_num=time_step_num, hidden_dim=hidden_dim, 
                                output_dim=output_dim, use_knowledge=use_knowledge, prior_matrix=knowledge_matrix)

    def forward(self, im_inds, obj_fmaps, obj_labels):
        """
        Reason object classes using knowledge of object cooccurrence
        """

        if self.mode == 'predcls':
            # in task 'predcls', there is no need to run GGNN_obj
            obj_dists = Variable(to_onehot(obj_labels.data, self.num_obj_cls))
            return obj_dists
        else:
            input_ggnn = self.obj_proj(obj_fmaps)

            lengths = []
            for i, s, e in enumerate_by_image(im_inds.data):
                lengths.append(e - s)
            obj_cum_add = np.cumsum([0] + lengths)
            obj_dists = torch.cat([self.ggnn_obj(input_ggnn[obj_cum_add[i] : obj_cum_add[i+1]]) for i in range(len(lengths))], 0)
            return obj_dists




class GGNNRelReason(nn.Module):
    """
    Module for relationship classification.
    """
    def __init__(self, mode='sgdet', num_obj_cls=151, num_rel_cls=51, obj_dim=4096, rel_dim=4096, 
                time_step_num=3, hidden_dim=512, output_dim=512,
                use_knowledge=True, knowledge_matrix=''):

        super(GGNNRelReason, self).__init__()
        assert mode in MODES
        self.mode = mode
        self.num_obj_cls = num_obj_cls
        self.num_rel_cls = num_rel_cls
        self.obj_dim = obj_dim
        self.rel_dim = rel_dim


        self.obj_proj = nn.Linear(self.obj_dim, hidden_dim)
        self.rel_proj = nn.Linear(self.rel_dim, hidden_dim)

        self.ggnn_rel = GGNNRel(num_rel_cls=num_rel_cls, time_step_num=time_step_num, hidden_dim=hidden_dim, 
                                output_dim=output_dim, use_knowledge=use_knowledge, prior_matrix=knowledge_matrix)

    def forward(self, obj_fmaps, obj_logits, rel_inds, vr, obj_labels=None, boxes_per_cls=None):
        """
        Reason relationship classes using knowledge of object and relationship coccurrence.
        """

        # print(rel_inds.shape)
        # (num_rel, 3)
        if self.mode == 'predcls':
            obj_dists2 = Variable(to_onehot(obj_labels.data, self.num_obj_cls))
        else:
            obj_dists2 = obj_logits

        if self.mode == 'sgdet' and not self.training:
            # NMS here for baseline
            probs = F.softmax(obj_dists2, 1)
            nms_mask = obj_dists2.data.clone()
            nms_mask.zero_()
            for c_i in range(1, obj_dists2.size(1)):
                scores_ci = probs.data[:, c_i]
                boxes_ci = boxes_per_cls.data[:, c_i]

                keep = apply_nms(scores_ci, boxes_ci,
                                    pre_nms_topn=scores_ci.size(0), post_nms_topn=scores_ci.size(0),
                                    nms_thresh=0.3)
                nms_mask[:, c_i][keep] = 1

            obj_preds = Variable(nms_mask * probs.data, volatile=True)[:,1:].max(1)[1] + 1
        else:
            obj_preds = obj_labels if obj_labels is not None else obj_dists2[:,1:].max(1)[1] + 1

        sub_obj_preds = torch.cat((obj_preds[rel_inds[:, 1]].view(-1, 1), obj_preds[rel_inds[:, 2]].view(-1, 1)), 1)

        obj_fmaps = self.obj_proj(obj_fmaps)
        vr = self.rel_proj(vr)
        input_ggnn = torch.stack([torch.cat([obj_fmaps[rel_ind[1]].unsqueeze(0), 
                                             obj_fmaps[rel_ind[2]].unsqueeze(0), 
                                             vr[index].repeat(self.num_rel_cls, 1)], 0) 
                                 for index, rel_ind in enumerate(rel_inds)])

        rel_dists = self.ggnn_rel(rel_inds[:, 1:], sub_obj_preds, input_ggnn)

        return obj_dists2, obj_preds, rel_dists




class VRFC(nn.Module):
    """
    Module for relationship classification just using a fully connected layer.
    """
    def __init__(self, mode, rel_dim, num_obj_cls, num_rel_cls):
        super(VRFC, self).__init__()
        self.mode = mode
        self.rel_dim = rel_dim
        self.num_obj_cls = num_obj_cls
        self.num_rel_cls = num_rel_cls
        self.vr_fc = nn.Linear(self.rel_dim, self.num_rel_cls)

    def forward(self, obj_logits, vr, obj_labels=None, boxes_per_cls=None):
        if self.mode == 'predcls':
            obj_dists2 = Variable(to_onehot(obj_labels.data, self.num_obj_cls))
        else:
            obj_dists2 = obj_logits

        if self.mode == 'sgdet' and not self.training:
            # NMS here for baseline
            probs = F.softmax(obj_dists2, 1)
            nms_mask = obj_dists2.data.clone()
            nms_mask.zero_()
            for c_i in range(1, obj_dists2.size(1)):
                scores_ci = probs.data[:, c_i]
                boxes_ci = boxes_per_cls.data[:, c_i]

                keep = apply_nms(scores_ci, boxes_ci,
                                    pre_nms_topn=scores_ci.size(0), post_nms_topn=scores_ci.size(0),
                                    nms_thresh=0.3)
                nms_mask[:, c_i][keep] = 1

            obj_preds = Variable(nms_mask * probs.data, volatile=True)[:,1:].max(1)[1] + 1
        else:
            obj_preds = obj_labels if obj_labels is not None else obj_dists2[:,1:].max(1)[1] + 1


        rel_dists = self.vr_fc(vr)

        return obj_dists2, obj_preds, rel_dists       




class KERN(nn.Module):
    """
    Knowledge-Embedded Routing Network 
    """
    def __init__(self, classes, rel_classes, mode='sgdet', num_gpus=1, 
                 require_overlap_det=True, pooling_dim=4096, use_resnet=False, thresh=0.01,
                 use_proposals=False,
                 use_ggnn_obj=False,
                 ggnn_obj_time_step_num=3,
                 ggnn_obj_hidden_dim=512,
                 ggnn_obj_output_dim=512,
                 use_ggnn_rel=False,
                 ggnn_rel_time_step_num=3,
                 ggnn_rel_hidden_dim=512,
                 ggnn_rel_output_dim=512,
                 use_obj_knowledge=True,
                 use_rel_knowledge=True,
                 obj_knowledge='',
                 rel_knowledge=''):

        """
        :param classes: Object classes
        :param rel_classes: Relationship classes. None if were not using rel mode
        :param mode: (sgcls, predcls, or sgdet)
        :param num_gpus: how many GPUS 2 use
        :param require_overlap_det: Whether two objects must intersect
        """
        super(KERN, self).__init__()
        self.classes = classes
        self.rel_classes = rel_classes
        self.num_gpus = num_gpus
        assert mode in MODES
        self.mode = mode
        self.pooling_size = 7
        self.obj_dim = 2048 if use_resnet else 4096
        self.rel_dim = self.obj_dim
        self.pooling_dim = pooling_dim

        self.use_ggnn_obj=use_ggnn_obj
        self.use_ggnn_rel = use_ggnn_rel

        self.require_overlap = require_overlap_det and self.mode == 'sgdet'

        self.detector = ObjectDetector(
            classes=classes,
            mode=('proposals' if use_proposals else 'refinerels') if mode == 'sgdet' else 'gtbox',
            use_resnet=use_resnet,
            thresh=thresh,
            max_per_img=64
        )


        self.union_boxes = UnionBoxesAndFeats(pooling_size=self.pooling_size, stride=16,
                                              dim=1024 if use_resnet else 512)

        if use_resnet:
            self.roi_fmap = nn.Sequential(
                resnet_l4(relu_end=False),
                nn.AvgPool2d(self.pooling_size),
                Flattener(),
            )
        else:
            roi_fmap = [
                Flattener(),
                load_vgg(use_dropout=False, use_relu=False, use_linear=pooling_dim == 4096, pretrained=False).classifier,
            ]
            if pooling_dim != 4096:
                roi_fmap.append(nn.Linear(4096, pooling_dim))
            self.roi_fmap = nn.Sequential(*roi_fmap)
            self.roi_fmap_obj = load_vgg(pretrained=False).classifier

        if self.use_ggnn_obj:
            self.ggnn_obj_reason = GGNNObjReason(mode=self.mode,
                                                 num_obj_cls=len(self.classes), 
                                                 obj_dim=self.obj_dim,
                                                 time_step_num=ggnn_obj_time_step_num,
                                                 hidden_dim=ggnn_obj_hidden_dim,
                                                 output_dim=ggnn_obj_output_dim,
                                                 use_knowledge=use_obj_knowledge,
                                                 knowledge_matrix=obj_knowledge)

        if self.use_ggnn_rel:
            self.ggnn_rel_reason = GGNNRelReason(mode=self.mode, 
                                                 num_obj_cls=len(self.classes), 
                                                 num_rel_cls=len(rel_classes), 
                                                 obj_dim=self.obj_dim, 
                                                 rel_dim=self.rel_dim, 
                                                 time_step_num=ggnn_rel_time_step_num, 
                                                 hidden_dim=ggnn_rel_hidden_dim, 
                                                 output_dim=ggnn_obj_output_dim,
                                                 use_knowledge=use_rel_knowledge,
                                                 knowledge_matrix=rel_knowledge)
        else:
            self.vr_fc_cls = VRFC(self.mode, self.rel_dim, len(self.classes), len(self.rel_classes))

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_rels(self):
        return len(self.rel_classes)

    def visual_rep(self, features, rois, pair_inds):
        """
        Classify the features
        :param features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4]
        :param rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        :param pair_inds inds to use when predicting
        :return: score_pred, a [num_rois, num_classes] array
                 box_pred, a [num_rois, num_classes, 4] array
        """
        assert pair_inds.size(1) == 2
        uboxes = self.union_boxes(features, rois, pair_inds)
        return self.roi_fmap(uboxes)

    def get_rel_inds(self, rel_labels, im_inds, box_priors):
        # Get the relationship candidates
        if self.training:
            rel_inds = rel_labels[:, :3].data.clone()
        else:
            rel_cands = im_inds.data[:, None] == im_inds.data[None]
            rel_cands.view(-1)[diagonal_inds(rel_cands)] = 0
            if self.require_overlap:
                rel_cands = rel_cands & (bbox_overlaps(box_priors.data,
                                                       box_priors.data) > 0)

                # if there are fewer then 100 things then we might as well add some?
                amt_to_add = 100 - rel_cands.long().sum()

            rel_cands = rel_cands.nonzero()

            if rel_cands.dim() == 0:
                rel_cands = im_inds.data.new(1, 2).fill_(0)

            rel_inds = torch.cat((im_inds.data[rel_cands[:, 0]][:, None], rel_cands), 1)

        return rel_inds

    def obj_feature_map(self, features, rois):
        """
        Gets the ROI features
        :param features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4] (features at level p2)
        :param rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        :return: [num_rois, #dim] array
        """
        feature_pool = RoIAlignFunction(self.pooling_size, self.pooling_size, spatial_scale=1 / 16)(
            features, rois)
        return self.roi_fmap_obj(feature_pool.view(rois.size(0), -1))

    def forward(self, x, im_sizes, image_offset,
                gt_boxes=None, gt_classes=None, gt_rels=None, proposals=None, train_anchor_inds=None,
                return_fmap=False):
        """
        Forward pass for detection
        :param x: Images@[batch_size, 3, IM_SIZE, IM_SIZE]
        :param im_sizes: A numpy array of (h, w, scale) for each image.
        :param image_offset: Offset onto what image we're on for MGPU training (if single GPU this is 0)
        :param gt_boxes:

        Training parameters:
        :param gt_boxes: [num_gt, 4] GT boxes over the batch.
        :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
        :param train_anchor_inds: a [num_train, 2] array of indices for the anchors that will
                                  be used to compute the training loss. Each (img_ind, fpn_idx)
        :return: If train:
            scores, boxdeltas, labels, boxes, boxtargets, rpnscores, rpnboxes, rellabels
            
            if test:
            prob dists, boxes, img inds, maxscores, classes
            
        """


        result = self.detector(x, im_sizes, image_offset, gt_boxes, gt_classes, gt_rels, proposals,
                               train_anchor_inds, return_fmap=True)
        if result.is_none():
            return ValueError("heck")

        im_inds = result.im_inds - image_offset
        boxes = result.rm_box_priors

        if self.training and result.rel_labels is None:
            assert self.mode == 'sgdet'
            result.rel_labels = rel_assignments(im_inds.data, boxes.data, result.rm_obj_labels.data,
                                                gt_boxes.data, gt_classes.data, gt_rels.data,
                                                image_offset, filter_non_overlap=True,
                                                num_sample_per_gt=1)


        rel_inds = self.get_rel_inds(result.rel_labels, im_inds, boxes)
        rois = torch.cat((im_inds[:, None].float(), boxes), 1)

        result.obj_fmap = self.obj_feature_map(result.fmap.detach(), rois)

        if self.use_ggnn_obj:          
                result.rm_obj_dists = self.ggnn_obj_reason(im_inds, 
                                                           result.obj_fmap,
                                                           result.rm_obj_labels if self.training or self.mode == 'predcls' else None)

        vr = self.visual_rep(result.fmap.detach(), rois, rel_inds[:, 1:])


        if self.use_ggnn_rel:
            result.rm_obj_dists, result.obj_preds, result.rel_dists = self.ggnn_rel_reason(
                obj_fmaps=result.obj_fmap,
                obj_logits=result.rm_obj_dists,
                vr=vr,
                rel_inds=rel_inds,
                obj_labels=result.rm_obj_labels if self.training or self.mode == 'predcls' else None,
                boxes_per_cls=result.boxes_all
            )   
        else:
            result.rm_obj_dists, result.obj_preds, result.rel_dists = self.vr_fc_cls(
                obj_logits=result.rm_obj_dists,
                vr=vr,
                obj_labels=result.rm_obj_labels if self.training or self.mode == 'predcls' else None,
                boxes_per_cls=result.boxes_all)


        if self.training:
            return result

        twod_inds = arange(result.obj_preds.data) * self.num_classes + result.obj_preds.data
        result.obj_scores = F.softmax(result.rm_obj_dists, dim=1).view(-1)[twod_inds]

        # Bbox regression
        if self.mode == 'sgdet':
            bboxes = result.boxes_all.view(-1, 4)[twod_inds].view(result.boxes_all.size(0), 4)
        else:
            # Boxes will get fixed by filter_dets function.
            bboxes = result.rm_box_priors

        rel_rep = F.softmax(result.rel_dists, dim=1)

        return filter_dets(bboxes, result.obj_scores,
                           result.obj_preds, rel_inds[:, 1:], rel_rep)

    def __getitem__(self, batch):
        """ Hack to do multi-GPU training"""
        batch.scatter()

        if self.num_gpus == 1:
            return self(*batch[0])

        replicas = nn.parallel.replicate(self, devices=list(range(self.num_gpus)))
        outputs = nn.parallel.parallel_apply(replicas, [batch[i] for i in range(self.num_gpus)])

        if self.training:
            return gather_res(outputs, 0, dim=0)
        return outputs




