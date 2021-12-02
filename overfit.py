import numpy as np

import torch

import matplotlib
import matplotlib.pyplot as plt

import argparse

from model import EventTransformer
from dataset import MVSEC, TartanAir

from plot import save_plot_flow


matplotlib.use('Agg')

parser = argparse.ArgumentParser()
parser.add_argument('--output_path', type=str)


def main() :
    args = parser.parse_args()

    save_figures = False
    if hasattr(args, 'output_path') :
        save_figures = True
        output_path = args.output_path
        print(output_path)

    # output_path = "/storage/user/gaul/gaul/thesis/documents/figures/test"

    # Settings
    training_frames = list(range(100, 110))
    val_frames = [400]

    epochs = 1000
    lr=0.0001

    print_freq = 10
    val_freq = 100
    model_freq = 1000
    vis_freq = 500


    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #dataset_train = MVSEC('../data/MVSEC/indoor_flying/', training_frames, 0.25)
    #dataset_val = MVSEC('../data/MVSEC/indoor_flying/', val_frames, 0.25)

    dataset_train = TartanAir('/storage/user/gaul/gaul/thesis/data/TartanAir/', training_frames, 0.033)
    dataset_val = TartanAir('/storage/user/gaul/gaul/thesis/data/TartanAir/', val_frames, 0.033)

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

        for frame_i, (event_data, query_locs, flows) in enumerate(dataset_train) : #range(len(data_gt['davis']['left']['flow_dist'])) :
            optimizer.zero_grad()

            pred = model(event_data.to(device), query_locs.to(device))

            loss = LFunc(pred, flows.to(device))

            loss.backward()
            optimizer.step()


            loss_all += loss.item()

            if save_figures and it % vis_freq == 0 :
                save_plot_flow(
                    output_path + "train_it" + str(it) + "_frame" + str(training_frames[frame_i]) + ".png",
                    dataset_train.images[frame_i],
                    query_locs.detach().cpu().numpy(),
                    pred.detach().cpu().numpy(),
                    flow_res=(260, 346))

                if it == 0 :
                    save_plot_flow(
                        output_path + "/train_gt_frame" + str(training_frames[frame_i]) + ".png",
                        dataset_train.images[frame_i],
                        query_locs.detach().cpu().numpy(),
                        flows.detach().cpu().numpy(),
                        flow_res=(260, 346))

        scheduler.step()

        train_loss_history.append(loss_all / len(dataset_train))


        if it % print_freq == 0 :
            print("Iteration: " + str(it) + ", training loss : " + str(loss_all / len(training_frames)))


        if it % val_freq == 0 or it % vis_freq == 0 :
            model.eval()
            val_loss_all = 0

            for val_frame_i, (event_data, query_locs, flows) in enumerate(dataset_val) :

                pred = model(event_data.to(device), query_locs.to(device))
                loss_val = LFunc(pred, flows.to(device))

                val_loss_all += loss_val.item()

                # TODO: print frame
                print("Iteration: " + str(it) + ", validation loss : " + str(loss_val.item() / len(dataset_val)))

                if save_figures and it % vis_freq == 0 :
                    save_plot_flow(
                        output_path + "val_it" + str(it) + "_frame" + str(val_frames[val_frame_i]) + ".png",
                        dataset_val.images[val_frame_i],
                        query_locs.detach().cpu().numpy(),
                        pred.detach().cpu().numpy(),
                        flow_res=(260, 346))

                    if it == 0 :
                        save_plot_flow(
                            output_path + "/val_gt_frame" + str(val_frames[val_frame_i]) + ".png",
                            dataset_val.images[val_frame_i],
                            query_locs.detach().cpu().numpy(),
                            flows.detach().cpu().numpy(),
                            flow_res=(260, 346))

            val_loss_history.append(val_loss_all)

            if save_figures :
                plt.plot(np.arange(0, it+1), train_loss_history, label="train")
                plt.plot(np.arange(0, it+1, val_freq), val_loss_history, label="val")
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
