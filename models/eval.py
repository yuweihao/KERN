import argparse

import config
import dataloaders.movies
import lib.evaluation.sg_eval
import lib.kern_model
import lib.pytorch_misc
import torch
import torch.utils.data
import tqdm


def main(path):
    conf = config.ModelConfig()

    # Set up movies dataset as a PyTorch data loader.
    dataset = dataloaders.movies.Movies(path)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=dataloaders.movies.Movies.collate_fn
    )

    # Set up and load pretrained KERN model.
    model = lib.kern_model.KERN(
        classes=dataset.objects,
        rel_classes=dataset.predicates,
        num_gpus=conf.num_gpus,
        require_overlap_det=True,
        use_resnet=conf.use_resnet,
        use_proposals=conf.use_proposals,
        use_ggnn_obj=conf.use_ggnn_obj,
        ggnn_obj_time_step_num=conf.ggnn_obj_time_step_num,
        ggnn_obj_hidden_dim=conf.ggnn_obj_hidden_dim,
        ggnn_obj_output_dim=conf.ggnn_obj_output_dim,
        use_obj_knowledge=conf.use_obj_knowledge,
        obj_knowledge=conf.obj_knowledge,
        use_ggnn_rel=conf.use_ggnn_rel,
        ggnn_rel_time_step_num=conf.ggnn_rel_time_step_num,
        ggnn_rel_hidden_dim=conf.ggnn_rel_hidden_dim,
        ggnn_rel_output_dim=conf.ggnn_rel_output_dim,
        use_rel_knowledge=conf.use_rel_knowledge,
        rel_knowledge=conf.rel_knowledge,
    )
    model.cuda()
    checkpoint = torch.load(conf.ckpt)
    lib.pytorch_misc.optimistic_restore(model, checkpoint["state_dict"])
    model.eval()

    # Evaluators
    evaluator = {"sgdet": lib.evaluation.sg_eval.BasicSceneGraphEvaluator("sgdet")}
    evaluators = [
        (index, predicate, {"sgdet": BasicSceneGraphEvaluator("sgdet")})
        for index, predicate in enumerate(dataset.predicates)
        if index > 0
    ]

    # Interate data and predict.
    for batch_index, batch in tqdm.tqdm(enumerate(data_loader)):
        # This is very much abuse of the indexing operation, but in the
        # implementation of KERN (see kern_model.py) it is mentioned as a "hack
        # to do multi-GPU training". Nevertheless, this produces the output
        # that we are interested in.
        output = model[batch]

        if conf.num_gpus == 1:
            output = [output]

        for result_index, (
            boxes,
            objects,
            object_scores,
            predicates,
            predicate_scores,
        ) in enumerate(output):
            print(boxes, objects, object_scores, predicates, predicate_scores)
            break

        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data/movies")

    args, _rest = parser.parse_known_args()

    main(path=args.path)
