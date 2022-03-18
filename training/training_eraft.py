import torch

from ERAFT.model.utils import bilinear_sampler
from training.training_volume import eval_volume_custom



def forward_eraft(model, sample, dataset, **kwargs) :
    im1 = sample['event_volume_old']
    im2 = sample['event_volume_new']

    flow_low_res, flow_list = model(image1=im1, image2=im2, flow_init=None)

    return (flow_low_res, flow_list)


def forward_eraft_custom(model, sample, dataset, **kwargs) :
    im1 = torch.stack(sample['event_volume_old'])
    im2 = torch.stack(sample['event_volume_new'])

    flow_low_res, flow_list = model(image1=im1, image2=im2, flow_init=None)

    return (flow_low_res, flow_list)


def eval_eraft_out(out,
                   sample, lfunc, **kwargs) :
    flow_low_res, flow_list = out
    pred_refined = flow_list[-1]

    valid = sample['valid2D']

    pred_eval = [pred_refined[i, :, valid[i, :]].transpose(1, 0) for i in range(len(pred_refined))]

    flow_dev = [sample['flow'][i, sample['valid2D'][i, :], :] for i in range(len(pred_refined))]

    loss = torch.cat([lfunc(pred_eval[i], flow_dev[i]).reshape(1)
                      for i in range(len(pred_refined))]).sum()

    return (loss,
            [predi.detach() for predi in pred_eval],
            [flowi.detach() for flowi in flow_dev],
            [torch.flip(torch.nonzero(sample['valid2D'][i, :]),
                        dims=[1])
             for i in range(len(pred_refined))])


def eval_eraft_custom(out, sample, lfunc, **kwargs) :
    flow_low_res, flow_list = out
    pred_flow = flow_list[-1]

    return eval_volume_custom(pred_flow, sample, lfunc)


