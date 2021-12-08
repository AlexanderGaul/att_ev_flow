import numpy as np

import torch

import matplotlib
import matplotlib.pyplot as plt

import argparse

from model import EventTransformer
from dsec import DSEC

from plot import save_plot_flow, create_event_picture


matplotlib.use('Agg')

parser = argparse.ArgumentParser()
parser.add_argument('--output_path', type=str)


def main() :
    args = parser.parse_args()

    save_figures = False
    if hasattr(args, 'output_path') :
        save_figures = False
        save_figures_train = False
        output_path = args.output_path
        print(output_path)

    epochs = 20
    lr=0.0001

    print("lr: " + str(lr))

    print_freq = 1
    val_freq = 200
    model_freq = 1000
    vis_freq = 500


    torch.backends.cudnn.benchmark = True
    print("Benchmark: " + str(torch.backends.cudnn.benchmark))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    train_set = DSEC(seqs=[0], frames=[list(range(20))])
    val_set = DSEC(seqs=[1], frames=[[0]])

    print(len(train_set))

    train_loader = torch.utils.data.DataLoader(train_set, num_workers=8, shuffle=True, pin_memory=False)

    # TODO: add data loader

    model = EventTransformer()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    train_loss_history = []
    val_loss_history = []

    LFunc = torch.nn.L1Loss()

    for it in range(epochs) :
        loss_all = 0
        loss_batch = torch.zeros(1).to(device)

        for _, (event_data, query_locs, flows, frame_i) in enumerate(train_loader) : #range(len(data_gt['davis']['left']['flow_dist'])) :
            optimizer.zero_grad()

            pred = model(torch.FloatTensor(event_data).to(device),
                         torch.FloatTensor(query_locs).to(device))

            loss = LFunc(pred, torch.FloatTensor(flows).squeeze(0).to(device))

            loss.backward()
            optimizer.step()

            loss_all += loss.item()

            if save_figures_train and it % vis_freq == 0 :
                save_plot_flow(
                    output_path + "/train_it" + str(it) + "_frame_" + train_set.get_frame_name(frame_i) + ".png",
                    create_event_picture(event_data[0, :].numpy()),
                    query_locs.detach().cpu()[0, :].numpy(),
                    pred.detach().cpu().numpy(),
                    flow_res=(480, 640))

                if it == 0 :
                    save_plot_flow(
                        output_path + "/train_gt_frame" + train_set.get_frame_name(frame_i) + ".png",
                        create_event_picture(event_data[0, :].numpy()),
                        query_locs.detach().cpu()[0, :].numpy(),
                        flows.detach().cpu()[0, :].numpy(),
                        flow_res=(480, 640))

        scheduler.step()

        train_loss_history.append(loss_all / len(train_set))


        if it % print_freq == 0 :
            print("Iteration: " + str(it) + ", training loss : " + str(loss_all / len(train_set)))


        if it % val_freq == 0 or it % vis_freq == 0 :
            model.eval()
            val_loss_all = 0

            for _, (event_data, query_locs, flows, val_frame_i) in enumerate(val_set) :
                pred = model(torch.FloatTensor(event_data).to(device),
                             torch.FloatTensor(query_locs).to(device))

                loss_val = LFunc(pred, torch.FloatTensor(flows).to(device))

                val_loss_all += loss_val.item()

                if save_figures and it % vis_freq == 0 :
                    save_plot_flow(
                        output_path + "/val_it" + str(it) + "_frame" + val_set.get_frame_name(val_frame_i) + ".png",
                        create_event_picture(event_data),
                        query_locs,
                        pred.detach().cpu().numpy(),
                        flow_res=(480, 640))

                    if it == 0 :
                        save_plot_flow(
                            output_path + "/val_gt_frame" + val_set.get_frame_name(val_frame_i) + ".png",
                            create_event_picture(event_data),
                            query_locs, flows,
                            flow_res=(480, 640))

                        # TODO: print frame
            print("Iteration: " + str(it) + ", validation loss : " + str(val_loss_all / len(val_set)))
            val_loss_history.append(val_loss_all / len(val_set))

            if save_figures :
                plt.plot(np.arange(0, len(train_loss_history)), train_loss_history, label="train")
                plt.plot(np.arange(0, len(val_loss_history) * val_freq, val_freq), val_loss_history, label="val")
                plt.legend(loc='upper right')
                plt.savefig(output_path + "/loss_history.png",
                            pad_inches=0.)
                plt.close()

            model.train()

        # scheduler.step()
        #if it % model_freq == 0 :
        #    torch.save(model.state_dict(), "/storage/user/gaul/gaul/thesis/models/overfit_5_it" + str(it))


if __name__ == "__main__" :
    main()
