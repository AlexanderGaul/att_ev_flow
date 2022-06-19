import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import imageio
import argparse
import json
from pathlib import Path

from models.convs.raft_style_wrapper import WrappedTransformer
from models.convs.conv_transformer import ConvTransformer
from ERAFT.model.eraft import ERAFT

from training.training_unet import EventUnetTrainer
from training.training_eraft import ERaftTrainer

from training.training_report import paint_pictures_flow

from plot import create_event_frame_picture

from train_scripts.train import set_up, update

from data.dsec_test import DSECFlowTest
from data.event_datasets import EventFlowDataset

from data.utils import collate_dict_adaptive

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=open, nargs='+')
parser.add_argument("--output_path", type=str)
parser.add_argument("--checkpoint", type=str)

def main() :
    args = parser.parse_args()
    config = {}

    if args.config_file is not None :
        for cfg in args.config_file :
            config = update(config, json.load(cfg))

    if hasattr(args, 'output_path') and args.output_path is not None :
        output_path = Path(args.output_path)
        print(output_path)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if not os.path.exists(output_path / "submission"):
            os.makedirs(output_path / "submission")
        if not os.path.exists(output_path / "visualization"):
            os.makedirs(output_path / "visualization")

    #model, dataset, _, trainer, config = set_up(config)

    if config['setup']['model'] == 'WrappedTransformer':
        model = WrappedTransformer(t_dims=config['dataset']['t_bins'],
                                   **config['model'])
    elif config['setup']['model'] == 'ConvTransformer' :
        model = ConvTransformer(t_dims=config['dataset']['t_bins'],
                                   **config['model'])
    trainer = EventUnetTrainer(lfunc=torch.nn.L1Loss())
    if config['setup']['model'] == 'ERaft' :
        model = ERAFT({'subtype': 'standard'},
                      n_first_channels=config['dataset']['t_bins'],
                      **config['model'])
        trainer = ERaftTrainer(lfunc=torch.nn.L1Loss())

    dataset = EventFlowDataset(**config['dataset'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'])

    for sample in dataset :
        print(sample['seq_name'])
        print(sample['out_file_name'])
        sample = collate_dict_adaptive([sample])
        sample = trainer.sample_to_device(sample, device)
        # TODO: sample to device
        # colate one sample
        # eval
        with torch.no_grad() :
            out = trainer.forward(model, sample)
        flow_frame = out['pred'][0].detach().cpu().numpy()

        write_flow_submission(flow_frame,
                              Path(args.output_path) / "submission" / sample['seq_name'][0],
                              sample['out_file_name'][0] + ".png")
        e_im = create_event_frame_picture(sample['event_volume'][0].detach().cpu().numpy())
        ims = paint_pictures_flow(e_im,
                                  flow_frame.transpose(1, 2, 0))

        if not os.path.exists(output_path / "visualization" / sample['seq_name'][0]) :
            os.mkdir(output_path / "visualization" / sample['seq_name'][0])
        imageio.imwrite(output_path / "visualization" / sample['seq_name'][0] /
                        (sample['out_file_name'][0] + "_eventsflow.png"),
                        ims['im_events'])
        imageio.imwrite(output_path / "visualization" / sample['seq_name'][0] /
                        (sample['out_file_name'][0] + "_events.png"),
                        e_im)
        imageio.imwrite(output_path / "visualization" / sample['seq_name'][0] /
                        (sample['out_file_name'][0] + "_flow.png"),
                        ims['im_color'])


        # create event and colorr flow visualization


        # write to output


def write_flow_submission(flow, parent_path, file_name) :
    _, h,w = flow.shape
    flow_map = np.rint(flow*128 + 2**15)
    flow_map = flow_map.astype(np.uint16).transpose(1,2,0)
    flow_map = np.concatenate((flow_map, np.zeros((h,w,1), dtype=np.uint16)), axis=-1)

    if not os.path.exists(parent_path):
        os.mkdir(parent_path)

    imageio.imwrite(os.path.join(parent_path, file_name), flow_map, format='PNG-FI')

if __name__ == "__main__" :
    main()