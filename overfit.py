import numpy as np

import torch

import matplotlib
import matplotlib.pyplot as plt

from model import EventTransformer
from dataset import MVSEC, TartanAir

from plot import plot_flow


matplotlib.use('Agg')



folder = "test_040/"
print(folder)
#rint("depth 12, ..., 100ms")

training_frames = list(range(100, 110))
val_frames = [410, 420, 430]

#dataset_train = MVSEC('../data/MVSEC/indoor_flying/', training_frames, 0.25)
#dataset_val = MVSEC('../data/MVSEC/indoor_flying/', val_frames, 0.25)

dataset_train = TartanAir('/storage/user/gaul/gaul/thesis/data/TartanAir/', training_frames, 0.033)
dataset_val = TartanAir('/storage/user/gaul/gaul/thesis/data/TartanAir/', val_frames, 0.033)


model = EventTransformer()

# TODO: if available
#torch.backends.cudnn.benchmark = True
device = torch.device("cuda")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

train_loss_history = []
val_loss_history = []

print_freq = 10
val_freq = 100
model_freq = 1000
vis_freq = 200
epochs = 2000

LFunc = torch.nn.L1Loss()

for event_data, _, _ in dataset_train :
    print(len(event_data))

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
        
        if it % vis_freq == 0 :
            plot_flow(dataset_train.images[frame_i], 
                      query_locs.detach().cpu().numpy(), 
                      pred.detach().cpu().numpy(),
                      flow_res=(260, 346))
            
            plt.savefig("/storage/user/gaul/gaul/thesis/documents/figures/" + 
                        folder + "train_it" + str(it) + "_frame" + str(training_frames[frame_i]) + ".png",
                        pad_inches=0.)
            plt.close()
            
            if it == 0 :
                plot_flow(dataset_train.images[frame_i], 
                          query_locs.detach().cpu().numpy(), 
                          flows.detach().cpu().numpy(),
                          flow_res=(260, 346))
            
                plt.savefig("/storage/user/gaul/gaul/thesis/documents/figures/" + folder + "/train_gt_frame" + str(training_frames[frame_i]) +  ".png",
                            pad_inches=0.)
                plt.close()
        
    
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
        
            if it % vis_freq == 0 :
                plot_flow(dataset_val.images[val_frame_i], 
                          query_locs.detach().cpu().numpy(), 
                          pred.detach().cpu().numpy(),
                          flow_res=(260, 346))
                
                plt.savefig("/storage/user/gaul/gaul/thesis/documents/figures/" + 
                            folder + "val_it" + str(it) + "_frame" + str(val_frames[val_frame_i]) +  ".png",
                            pad_inches=0.)
                plt.close()
                
                if it == 0 :
                    plot_flow(dataset_val.images[val_frame_i], 
                              query_locs.detach().cpu().numpy(), 
                              flows.detach().cpu().numpy(),
                              flow_res=(260, 346))
                
                    plt.savefig("/storage/user/gaul/gaul/thesis/documents/figures/" + 
                                folder + "/val_gt_frame" + str(val_frames[val_frame_i]) +  ".png",
                                pad_inches=0.)
                    plt.close()
                
        val_loss_history.append(val_loss_all)
        
        
        model.train()

    # scheduler.step()
    #if it % model_freq == 0 :
    #    torch.save(model.state_dict(), "/storage/user/gaul/gaul/thesis/models/overfit_5_it" + str(it))

plt.plot(np.arange(0, epochs), train_loss_history, label="train")
plt.plot(np.arange(0, epochs, val_freq), val_loss_history, label="val")
plt.legend(loc='upper right')
plt.savefig("/storage/user/gaul/gaul/thesis/documents/figures/" + folder + "/loss_history.png",
            pad_inches=0.)
plt.close()



