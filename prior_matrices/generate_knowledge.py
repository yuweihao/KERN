"""
Generate knowledge matrices
"""

import numpy as np
from dataloaders.visual_genome import VG
from lib.fpn.box_intersections_cpu.bbox import bbox_overlaps
from lib.pytorch_misc import nonintersecting_2d_inds
from collections import Counter


def get_obj_cooccurrence_mat(train_data=VG(mode='train', filter_duplicate_rels=False, num_val_im=5000)):
    """
    Get the object cooccurrence matrix, where the (i, j) means P(o_j | o_i)
    :param train_data: 
    """


    mat = np.zeros((
        train_data.num_classes,
        train_data.num_classes,
    ), dtype=np.int64)
    sum_obj_pict = np.zeros(train_data.num_classes, dtype=np.int64)

    for ex_ind in range(len(train_data)):
        gt_classes = train_data.gt_classes[ex_ind].copy()
        gt_classes_list = list(set(gt_classes)) 
        for i in gt_classes_list:
            sum_obj_pict[i] += 1
        inds = np.transpose(np.nonzero(1 - np.eye(len(gt_classes_list), dtype=np.int64)))
        for (i, j) in inds:
            
            # Thank @ystluffy for finding that it is 
            # more accurate to replace ```mat[gt_classes[i], gt_classes[j]] += 1``` with mat[gt_classes_list[i], gt_classes_list[j]] +=1.
            # However, since the checkpoints we released were trained by the old code, 
            # if you want to use our checkpoint, you still need to use the old code ```mat[gt_classes[i], gt_classes[j]] += 1```.
            
            # mat[gt_classes_list[i], gt_classes_list[j]] +=1 # If you want to train models by yourself, please uncomment this code because it is more accurate.
            mat[gt_classes[i], gt_classes[j]] += 1 # If you want to use our checkpoint, you still need to use this code. If you want to train models by yourself, please comment it.
        for key, value in dict(Counter(gt_classes)).items():
            if value >= 2:
                mat[key, key] += 1

    sum_obj_pict[0] = 1 # because idx 0 means background, and the value is zero, divide zero will occurr an error, so add 1.

    obj_cooccurrence_matrix = mat / np.expand_dims(sum_obj_pict, axis=1)

    return obj_cooccurrence_matrix


def get_counts(train_data=VG(mode='train', filter_duplicate_rels=False, num_val_im=5000), must_overlap=True):
    """
    Get counts of all of the relations. Used for modeling directly P(rel | o1, o2)
    :param train_data: 
    :param must_overlap: 
    :return: 
    """
    fg_matrix = np.zeros((
        train_data.num_classes,
        train_data.num_classes,
        train_data.num_predicates,
    ), dtype=np.int64)

    bg_matrix = np.zeros((
        train_data.num_classes,
        train_data.num_classes,
    ), dtype=np.int64)

    for ex_ind in range(len(train_data)):
        gt_classes = train_data.gt_classes[ex_ind].copy()
        gt_relations = train_data.relationships[ex_ind].copy()
        gt_boxes = train_data.gt_boxes[ex_ind].copy()

        # For the foreground, we'll just look at everything
        o1o2 = gt_classes[gt_relations[:, :2]]
        for (o1, o2), gtr in zip(o1o2, gt_relations[:,2]):
            fg_matrix[o1, o2, gtr] += 1

        # For the background, get all of the things that overlap.
        o1o2_total = gt_classes[np.array(
            box_filter(gt_boxes, must_overlap=must_overlap), dtype=int)]
        for (o1, o2) in o1o2_total:
            bg_matrix[o1, o2] += 1

    return fg_matrix, bg_matrix


def box_filter(boxes, must_overlap=False):
    """ Only include boxes that overlap as possible relations. 
    If no overlapping boxes, use all of them."""
    n_cands = boxes.shape[0]

    overlaps = bbox_overlaps(boxes.astype(np.float), boxes.astype(np.float)) > 0
    np.fill_diagonal(overlaps, 0)

    all_possib = np.ones_like(overlaps, dtype=np.bool)
    np.fill_diagonal(all_possib, 0)

    if must_overlap:
        possible_boxes = np.column_stack(np.where(overlaps))

        if possible_boxes.size == 0:
            possible_boxes = np.column_stack(np.where(all_possib))
    else:
        possible_boxes = np.column_stack(np.where(all_possib))
    return possible_boxes



if __name__ == '__main__':
    obj_matrix = get_obj_cooccurrence_mat()
    np.save('prior_matrices/obj_matrix', obj_matrix)

    fg_matrix, bg_matrix = get_counts(must_overlap=True)
    bg_matrix += 1
    fg_matrix[:, :, 0] = bg_matrix
    rel_matrix = fg_matrix / fg_matrix.sum(2)[:, :, None]
    np.save('prior_matrices/rel_matrix', rel_matrix)


