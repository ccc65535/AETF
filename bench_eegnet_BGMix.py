
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import joblib

import os,copy

import sys
sys.path.append(r"../metabci")



from metabci.brainda.datasets import Wang2016  # type: ignore

from metabci.brainda.algorithms.decomposition import (  # type: ignore
    generate_filterbank, generate_cca_references)
from metabci.brainda.algorithms.utils.model_selection import (  # type: ignore
    set_random_seeds,
    generate_kfold_indices, match_kfold_indices,
    generate_loo_indices, match_loo_indices)


from sklearn.metrics import confusion_matrix, balanced_accuracy_score,accuracy_score

from baseline.eegnet import EEGNet  
from Mixup import BGMix


import torch, skorch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skorch.classifier import NeuralNetClassifier
from skorch.helper import predefined_split
from skorch.callbacks import (LRScheduler, EpochScoring, Checkpoint, Callback,
                              TrainEndCheckpoint, LoadInitState, EarlyStopping)

from torch.utils.data import TensorDataset,DataLoader

from util import *


device_id = 0
device = torch.device("cuda:{:d}".format(device_id) if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
print("Total available GPU devices: {}".format(torch.cuda.device_count()))
print("Current pytorch device: {}".format(device))


x_dtype, y_dtype = torch.float, torch.long

model_name = 'eegnet-ssvep'

n_bands = 3
n_harmonics = 5

workspace_folder='./demo'

force_update = True
save_folder = os.path.join(
    workspace_folder, 
    'dl-methods')

SEED=42

#########################
dataset=Wang2016()
delay=0.14

dataset_channels= ['PZ', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'O1', 'OZ', 'O2']


srate = 250
# durations = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
durations = [.2,.3,.4,.5]
# durations = [.25,.35,.45,.55]

first_run=False
if first_run:

    # dataset.download_all(
    #     path='D:\\yuejin\\data', # save folder
    #     force_update=False, # re-download even if the data exist
    #     proxies=None, # add proxy if you need, the same as the Request package
    #     verbose=None
    # )

    make_indice(dataset)
os.makedirs(save_folder, exist_ok=True)
        
dataset_events = sorted(list(dataset.events.keys()))
freqs = [dataset.get_freq(event) for event in dataset_events]
phases = [dataset.get_phase(event) for event in dataset_events]

indice_file = "{:s}-loo-{:d}class-indices.joblib".format(
    dataset.dataset_code, len(dataset_events))
indices = joblib.load(
    os.path.join('indices', indice_file))['indices']


loo = len(indices[1][dataset_events[0]])


X, y, meta = get_ssvep_data(
    dataset, srate, dataset_channels, 1.1, dataset_events, 
    delay=delay, 
    data_hook=data_hook)
labels = np.unique(y)
print("Dataset: {} Size: {}".format(dataset.dataset_code, X.shape))
_, n_channels, n_samples = X.shape
n_classes = len(labels)

min_f, max_f = np.min(freqs), np.max(freqs)
wp = [[min_f*i, max_f*i] for i in range(1, 6)]
ws= [[min_f*i-2, max_f*i+2] for i in range(1, 6)]
aug_filterbank = generate_filterbank(wp, ws, srate, order=4, rp=1)

log_file_name='./record/log-eegnet-'+time.strftime('%m-%d-%H-%M')
log_file=open(log_file_name,mode='w+',buffering=-1)

res_file_name='./record/res-eegnet-'+time.strftime('%m-%d-%H-%M')+'.xlsx'
res_file=pd.ExcelWriter(res_file_name)

for duration in durations:
    os.makedirs(save_folder, exist_ok=True)
    file_name = make_file(
        dataset, model_name, dataset_channels, srate, duration, dataset_events, 
        n_bands=n_bands)
    save_file = os.path.join(save_folder, file_name)
    if not force_update and os.path.exists(save_file):
        continue
    
    mov_size=2
    # gap=5

    loo_global_accs = []
    loo_global_model_states = []
    loo_fine_tuning_accs = []
    for k in range(loo):
    # for k in [3]:

        set_random_seeds(SEED)
        
        filterX, filterY = np.copy(X[..., :int(srate*duration)]), np.copy(y)
        filterX = filterX - np.mean(filterX, axis=-1, keepdims=True)
        
        train_ind, validate_ind, test_ind = match_loo_indices(
            k, meta, indices)
        trainX, trainY, trainMeta = filterX[train_ind], filterY[train_ind], meta.iloc[train_ind]
        validateX, validateY, validateMeta = filterX[validate_ind], filterY[validate_ind], meta.iloc[validate_ind]
        testX, testY, testMeta = filterX[test_ind], filterY[test_ind], meta.iloc[test_ind]


        ###


        trainX, validateX, testX = generate_tensors(
            trainX, validateX, testX, dtype=x_dtype)
        trainY, validateY, testY = generate_tensors(
            trainY, validateY, testY, dtype=y_dtype)

        ######



        batch_size,max_epochs,lr = 256,600,1e-3

        
        all_model=EEGNet(
                n_channels, int(srate*duration), n_classes,
                time_kernel=(96, (1, int(srate*duration)), (1, 1)), 
                D=1,
                separa_kernel=(96, (1, 16), (1, 1)),
                dropout_rate=0.2,
                fc_norm_rate=1)

        

        net=NeuralNetClassifier(
            module=all_model,
            criterion=nn.CrossEntropyLoss,
            optimizer=optim.Adam,
            optimizer__weight_decay=0,
            lr=lr,
            max_epochs = max_epochs,
            batch_size = batch_size,
            # Shuffle training data on each epoch
            iterator_train__shuffle=True,
            device=device,

            callbacks=[
                    ('train_acc', EpochScoring('accuracy', 
                                                name='train_acc', 
                                                on_train=True, 
                                                lower_is_better=False)),
                    ('lr_scheduler', LRScheduler('CosineAnnealingLR', T_max=300 - 1)),
                    ('estoper', EarlyStopping(patience=50)),
                    ('checkpoint', Checkpoint(dirname="checkpoints/{:s}".format(str(id(all_model))))),
            ],  

        )


        net.train_split = predefined_split(
                skorch.dataset.Dataset(
                    X=validateX.to(device),
                    y=validateY.to(device)
                    )
        )
            
        save_path='./model_save/eegnet-fold-'+str(k)+'-'+str(duration)+'s-bench.pth'
        tunning=True
        new_test=True
        # new_test = False
        if tunning:
            if new_test:
                net = net.fit(
                    X= trainX.to(device), 
                    y=trainY.to(device)
                    )
                torch.save(all_model.state_dict(),save_path)
            else:
                net.initialize()
                all_model.load_state_dict(torch.load(save_path))
                print('load all sub model')

        else:
            net = net.fit(
                    X= trainX.to(device), 
                    y=trainY.to(device)
            )


        loo_global_model_states.append(
            copy.deepcopy(net.module.state_dict()))

        ## testing
        sub_accs = []
        for sub_id in dataset.subjects:
            sub_test_mask = (testMeta['subject']==sub_id).to_numpy()
            pred_labels = net.predict(X=testX[sub_test_mask])
            true_labels = testY[sub_test_mask].numpy()
            sub_acc = balanced_accuracy_score(pred_labels, true_labels)
            sub_accs.append(sub_acc)
        loo_global_accs.append(sub_accs)




        last_save=0
        ## fine-tuning
        sub_accs = []
        for sub_id in dataset.subjects:
        # for sub_id in [2,9,10]:
            print(f'fold{k},sub{sub_id}')
            sub_train_mask = (trainMeta['subject']==sub_id).to_numpy()
            sub_valid_mask = (validateMeta['subject']==sub_id).to_numpy()
            sub_test_mask = (testMeta['subject']==sub_id).to_numpy()
            
            sub_trainX, sub_trainY = trainX[sub_train_mask], trainY[sub_train_mask]
            sub_validateX, sub_validateY = validateX[sub_valid_mask], validateY[sub_valid_mask]
            sub_testX, sub_testY = testX[sub_test_mask], testY[sub_test_mask]


            
            
            ###
            used_X=torch.cat((sub_trainX,sub_validateX),dim=0)
            used_Y=np.concatenate((sub_trainY,sub_validateY),axis=0)
            used_Y_zip=np.array(list(zip(used_Y,used_Y)))
            #

            sub_trainX,sub_trainY=BGMix.generate_data(used_X.numpy(),used_Y,nt=[2 for i in range(40)]+[3 for i in range(40)]+[4 for i in range(40)])
            sub_validateX,sub_validateY=used_X.numpy(),used_Y
     
            
            sub_testX=sub_testX.to(device)
            # sub_testY=torch.tensor(sub_testY,dtype=torch.long).to(device)



            batch_size,max_epochs,lr = 128,600,5e-4
            # lamda=0.8
            patience=5

            train_dataset=TensorDataset(
                torch.tensor(sub_trainX,dtype=x_dtype),
                torch.tensor(sub_trainY,dtype=torch.long),
                
            )
            data_loader=DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True
            
            )  

                        
            valid_dataset=TensorDataset(
                torch.tensor(sub_validateX,dtype=x_dtype),
                torch.tensor(sub_validateY,dtype=torch.long),
            )
            valid_data_loader=DataLoader(
                valid_dataset,
                batch_size=batch_size,
                shuffle=True
            )
            

            sub_model=EEGNet(
                n_channels, int(srate*duration), n_classes,
                time_kernel=(96, (1, int(srate*duration)), (1, 1)), 
                D=1,
                separa_kernel=(96, (1, 16), (1, 1)),
                dropout_rate=0.2,
                fc_norm_rate=1).to(device)



            sub_model.load_state_dict(
                copy.deepcopy(loo_global_model_states[k]))

            loss_fun=nn.CrossEntropyLoss()
            optimizer=optim.Adam(sub_model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.98)

            max_acc=0
            min_loss=np.inf
            save_path='./model_save/naki-eegnet-'+str(duration)+'s-fold-'+str(k)+'-sub-'+str(sub_id)+'.pth'

            for epoch in range(max_epochs):

                # if epoch>50:
                #     patience=5
                sub_model.train()
                total_loss=0
                for i, data in enumerate(data_loader):

                    lamda = np.random.beta(8,2)

                    x,l=data
                    x,l=x.to(device),l.to(device)
                    # sub_model.train()
                    
                    out=sub_model(x)
                    loss=lamda*loss_fun(out,l[:,0])+(1-lamda)*loss_fun(out,l[:,1])
                    # loss=loss_fun(out,l)
                    
                    total_loss+=loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(sub_model.parameters(), 0.5)
                    optimizer.step()

                sub_model.eval()
                with torch.no_grad():
                    valid_loss=0
                    valid_pred_labels=[]
                    valid_true_labels=[]
                    for i, data in enumerate(valid_data_loader):
                        torch.cuda.empty_cache()
                        # sub_model.eval()

                        lamda = np.random.beta(8,2)

                        x,l=data
                        x,l=x.to(device),l.to(device)

                        out=sub_model(x)
                        
                        loss=loss_fun(out,l)
                        # loss=lamda*loss_fun(out,l[:,0])+(1-lamda)*loss_fun(out,l[:,1])

                        valid_loss+=loss
                        valid_pred_labels+=list(out.argmax(axis=1).detach().cpu().numpy())


                        valid_true_labels+=list(l.detach().cpu().numpy())

                    del x,l
    
                    valid_acc = balanced_accuracy_score(valid_pred_labels, valid_true_labels)

                    sub_model.eval()
                    output = sub_model(sub_testX)
                    pred_labels=output.argmax(axis=1).detach().cpu().numpy()

                    true_labels = sub_testY.numpy()
                    sub_acc = balanced_accuracy_score(pred_labels, true_labels)

                    valid_loss/=len(valid_data_loader)

                    print(f'epoch:{epoch},train loss:{total_loss/(len(data_loader)):.3f},valid loss:{valid_loss:.4f},valid acc:{valid_acc:.3f},test acc:{sub_acc:.3f}')

                    if (valid_loss+5e-4<min_loss):
                    # if (valid_loss+1e-4<min_loss) or(valid_acc>max_acc):
                        torch.save(sub_model.state_dict(),save_path)

                        print('save model.')
                        max_acc=valid_acc
                        last_save=epoch

                        min_loss=valid_loss
                        

                    if last_save+patience<epoch:
                        print('early stop')
                        break

            state_dict = torch.load(save_path, map_location=device)
            sub_model.load_state_dict(state_dict)
            sub_model.eval()

            output = sub_model(sub_testX)
            pred_labels=output.argmax(axis=1).detach().cpu().numpy()

            true_labels = sub_testY.numpy()
            sub_acc = balanced_accuracy_score(pred_labels, true_labels)
            sub_accs.append(sub_acc)

        loo_fine_tuning_accs.append(sub_accs)

    global_sub_accs = np.array(loo_global_accs).T
    ft_sub_accs = np.array(loo_fine_tuning_accs).T

    joblib.dump(
        {'global_sub_accs': global_sub_accs, 'ft_sub_accs': ft_sub_accs}, save_file)
    torch.save(loo_global_model_states, save_file.replace('joblib', 'pt'))
    print("Processing {:s} with {:s}... Global Acc:{:.4f}".format(
        dataset.dataset_code, model_name, np.mean(global_sub_accs)))
    print("Processing {:s} with {:s}... Fine-Tuning Acc:{:.4f}".format(
        dataset.dataset_code, model_name, np.mean(ft_sub_accs)))

    pd.DataFrame(ft_sub_accs).to_excel(excel_writer=res_file,sheet_name=str(duration), index=False, header=False)
    res_file._save()