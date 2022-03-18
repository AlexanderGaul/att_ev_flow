import torch

from ERAFT.model.utils import bilinear_sampler


def eval_volume(volume, sample, lfunc) :
    N, C, H, W = volume.shape
    assert C == 2
    # TODO: figure out if this is such a smart choice
    preds = [volume[i][:, sample['flow_mask'][i]] for i in range(N)]
    flows = [sample['flow_frame'][i][:, sample['flow_mask'][i]] for i in range(N)]

    loss = torch.cat([lfunc(preds[i],
                            flows[i]).reshape(1)
                      for i in range(N)]).sum()

    return(loss,
           [predi.detach().transpose(1, 0) for predi in preds],
           [flowi.detach().transpose(1, 0) for flowi in flows],
           [maski.detach().nonzero().flip(1) for maski in sample['flow_mask']])




def eval_volume_custom(volume, sample, lfunc) :
    # volume shape N C H W
    N, C, H, W = volume.shape
    assert C == 2
    assert N == len(sample['coords'])
    # maybe shape N C H_out W_out
    pred_sampled = [bilinear_sampler(volume[[i], :], sample['coords'][i].reshape((1, 1, -1, 2)))
                    .reshape(2, -1).transpose(1, 0)
                    for i in range(N)]
    """pred_sampled = [volume[i, :,
                           sample['coords'][i][:, 1].long(),
                           sample['coords'][i][:, 0].long()].transpose(1, 0) 
                           for i in range(N)]"""

    loss = torch.cat([lfunc(pred_sampled[i], sample['flows'][i]).reshape(1)
                     for i in range(N)]).sum()

    return (loss,
            [predi.detach() for predi in pred_sampled],
            [flowi.detach() for flowi in sample['flows']],
            [coordi.detach() for coordi in sample['coords']])