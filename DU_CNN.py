import mat73

import os

import torch

from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from utils.cosine_annealing_with_warmup import CosineAnnealingWarmUpRestarts


import matplotlib.pyplot as plt
import numpy as np

from models.network_model_batch import Networkn

# class MyLoss(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx,y,y_pred,y2,y2_pred):
#         ctx.save_for_backward(y,y_pred,y2,y2_pred)
#         return (y_pred-y).pow(2).sum()+0.05*(y2_pred-y2).pow(2).sum()
#     @staticmethod
#     def backward(ctx,grad_ouput):
#         yy,yy_pred,yy2,yy2_pred=ctx.saved_tensors
#         grad_input=torch.neg(2.0*(yy_pred-yy)+2.0*(yy2_pred-yy2))
#         return grad_input, None




def main():
    Load_Data=0
    USE_CUDA = torch.cuda.is_available()
    print(USE_CUDA)
    device = torch.device('cuda:0' if USE_CUDA else 'cpu')
    print('학습을 진행하는 기기:',device)


    num_epochs = 101
    batch_size = 128
    learning_rate = 1e-5


    #dataset = MNIST('./data', transform=img_transform)
    lastevalloss=[]

    matfile=mat73.loadmat('./data/rawdataM.mat')
    wv=matfile['xframe']
    wv=np.transpose(wv)
    shortdata=matfile['val']
    longdata=matfile['vallong']
    matfile=mat73.loadmat('./data/poddata.mat')
    V=matfile['V']
    Dvbd=matfile['DvBoundary']
    snapshot_mean=matfile['Snapshot_mean']
    trainsnapshotmean=torch.from_numpy(np.array(snapshot_mean)).float().cuda()
    trainV=torch.from_numpy(np.array(V)).float().cuda()
    trainDvbd=torch.from_numpy(np.array(Dvbd)).float().cuda()
    randomstate=[2,5]#,8
    #layerss=[2,3,5,7,9,11,13,15,17,19]

    #layerss=[2,3,5,7,9,11]
    layerss=[7]#[2,3,5,7,11,15] [2,3,5,7,11,15]     2,3,5,7,11,15   7,14,28,56,112,224
    kernelss=[7]#[3,7,15,25,45]     3,5,7,9,11,15,21 [7,9,11,15]      3,15,45 29,45,57,115     
    channelss=[16]  #2,4,8,16,32,48,'64 [8,16,32]
    downsampless=[16]    #[32,16,8,4,2,1]   [1,2,4,8,16,32]            1,16
    lossnum=[0] # lossfunction

    for downsample in downsampless:    
        for nlayer in layerss:
            if not os.path.exists('./{}layers'.format(nlayer)):
                os.mkdir('./{}layers'.format(nlayer))

            for nkernel in kernelss:
                for nchannel in channelss:
                    if not os.path.exists('./{}layers/{}kernel_{}channel_down{}'.format(nlayer,nkernel,nchannel,downsample)):
                        os.mkdir('./{}layers/{}kernel_{}channel_down{}'.format(nlayer,nkernel,nchannel,downsample))
                        
                    for ijij in lossnum:  
                        
                        if not os.path.exists('./{}layers/{}kernel_{}channel_down{}/test{}'.format(nlayer,nkernel,nchannel,downsample,ijij)):
                            os.mkdir('./{}layers/{}kernel_{}channel_down{}/test{}'.format(nlayer,nkernel,nchannel,downsample,ijij))
                            os.mkdir('./{}layers/{}kernel_{}channel_down{}/test{}/tut_img'.format(nlayer,nkernel,nchannel,downsample,ijij))
                        FPATH='./{}layers/{}kernel_{}channel_down{}/test{}'.format(nlayer,nkernel,nchannel,downsample,ijij)
                        



                        trainnum, testnum=train_test_split(range(50), test_size=0.1, random_state=2)
                        print(nkernel)
                        print(nchannel)
                        print(trainnum)
                        print(testnum)


                        tmpimg=np.array([])
                        tmplabel=np.array([])
                        train=np.array([])
                        train_label=np.array([])
                        test=np.array([])
                        test_label=np.array([])

                        trainshow=[]
                        trainshow_label=[]

                        j=0
                        for i in trainnum:
                            data1=shortdata[i][0]
                            data2=longdata[i][0]
                            trainshow.append(data1[:,2])
                            trainshow_label.append(data2[:,2])

                            if j==0 :
                                train=np.repeat(data1,data2.shape[1],1)
                                train_label=np.tile(data2,[1,data1.shape[1]])
                            else:
                                tmpimg=np.repeat(data1,data2.shape[1],1)
                                tmplabel=np.tile(data2,[1,data1.shape[1]])       
                                train=np.concatenate([train,tmpimg],1)
                                train_label=np.concatenate([train_label,tmplabel],1)
                            j+=1

                        trainshow=torch.from_numpy(np.array(trainshow)).float()
                        trainshow_label=torch.from_numpy(np.array(trainshow_label)).float()


                        testshow=[]
                        testshow_label=[]
                        j=0
                        for i in testnum:
                            data1=shortdata[i][0]
                            data2=longdata[i][0]
                            testshow.append(data1[:,2])
                            testshow_label.append(data2[:,2])
                            if j==0 :
                                test=np.repeat(data1,data2.shape[1],1)
                                test_label=np.tile(data2,[1,data1.shape[1]])
                            else:
                                tmpimg=np.repeat(data1,data2.shape[1],1)
                                tmplabel=np.tile(data2,[1,data1.shape[1]])       
                                test=np.concatenate([test,tmpimg],1)
                                test_label=np.concatenate([test_label,tmplabel],1)
                            j+=1

                        testshow=torch.from_numpy(np.array(testshow)).float()
                        testshow_label=torch.from_numpy(np.array(testshow_label)).float()
                    
                        train=torch.from_numpy(np.transpose(train)).float()
                        train_label=torch.from_numpy(np.transpose(train_label)).float()
                        traindataset=TensorDataset(train,train_label)
                        test=torch.from_numpy(np.transpose(test)).float()
                        test_label=torch.from_numpy(np.transpose(test_label)).float()
                        testdataset=TensorDataset(test,test_label)


                        traindataloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True, num_workers=6,persistent_workers=True)
                        
                        testloader = DataLoader(testdataset, batch_size=32, shuffle=False)

                        #criterion = nn.L1Loss()
                        criterion = nn.MSELoss()
                        
                        
                        model = Networkn(nlayer, downsample,nkernel,nchannel,in_nc=1 , out_nc=1, act_mode='BR').cuda()
                        
                        optimizer = torch.optim.Adam(
                            model.parameters(), lr=learning_rate, weight_decay=1e-6)
                        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,eta_min=1e-6,T_max=20)
                        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=50, T_mult=1, eta_max=0.005,  T_up=10, gamma=0.1)

                        PATH='./TUT.pth'
                        if os.path.isfile(PATH) and Load_Data == 1:
                            checkpoint = torch.load(PATH)
                            model.load_state_dict(checkpoint['model_state_dict'])
                            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                            epoch = checkpoint['epoch']
                            loss = checkpoint['loss']
                            print("load parameter")




                        loss_ = [] # train loss를 저장할 리스트.
                        evalloss_ = [] # test loss를 저장할 리스트.
                        
                        
                        Timecal = [] # calculation time
                        for epoch in range(num_epochs):
                            start = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            start.record() 
                            running_loss = 0.0 
                            model.train()
                            for data,data_label in traindataloader:
                                randint=torch.from_numpy(np.random.uniform(0.8, 1.2, size=(data.shape[0],1))).float().cuda()
                                spec = data
                                spec = Variable(spec).cuda()*randint
                                label = data_label.cuda()*randint
                                # ===================forward=====================
                                
                                output = model(spec.unsqueeze(1))
                                if ijij == 0:
                                    loss =criterion(output.squeeze(), label)
                                elif ijij == 1:
                                    ouputpca=torch.matmul((output.squeeze()).sub(randint*trainsnapshotmean),trainV)
                                    labelpca=torch.matmul((label).sub(randint*trainsnapshotmean),trainV)
                                    loss =0.9*criterion(output.squeeze(), label) + 0.1*criterion(ouputpca, labelpca)
                                elif ijij == 2:
                                    ouputpca=torch.matmul((output.squeeze()).sub(randint*trainsnapshotmean),trainV)
                                    labelpca=torch.matmul((label).sub(randint*trainsnapshotmean),trainV)
                                    normouputpca=(ouputpca-trainDvbd[:,1].T)/(trainDvbd[:,0].T-trainDvbd[:,1].T).repeat(labelpca.shape[0],1)
                                    normlabelpca=(labelpca-trainDvbd[:,1].T)/(trainDvbd[:,0].T-trainDvbd[:,1].T).repeat(labelpca.shape[0],1)
                                    loss =0.9*criterion(output.squeeze(), label) + 0.1*criterion(normouputpca, normlabelpca)
                                elif ijij == 3:
                                    ouputpca=torch.matmul((output.squeeze()).sub(randint*trainsnapshotmean),trainV)
                                    labelpca=torch.matmul((label).sub(randint*trainsnapshotmean),trainV)
                                    loss = criterion(ouputpca, labelpca)
                                elif ijij == 4:
                                    ouputpca=torch.matmul((output.squeeze()).sub(randint*trainsnapshotmean),trainV)
                                    labelpca=torch.matmul((label).sub(randint*trainsnapshotmean),trainV)
                                    normouputpca=(ouputpca-trainDvbd[:,1].T)/(trainDvbd[:,0].T-trainDvbd[:,1].T).repeat(labelpca.shape[0],1)
                                    normlabelpca=(labelpca-trainDvbd[:,1].T)/(trainDvbd[:,0].T-trainDvbd[:,1].T).repeat(labelpca.shape[0],1)
                                    loss = criterion(normouputpca, normlabelpca)

                                #plt.plot(torch.matmul(ouputpca,trainV.T)[0,:].cpu().detach()+(randint*trainsnapshotmean)[0,:].cpu().detach())

                                #plt.plot(train[0])
                                #plt.plot(model(train[0].unsqueeze(0).unsqueeze(0).cuda()).cpu().detach().numpy().squeeze(0).squeeze(0))

                                # ===================backward====================
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()
                                running_loss += loss.item()*data.shape[0]
                            scheduler.step()
                            #print(scheduler.get_last_lr())
                            
                            
                            # ===================log========================
                            end.record() # Waits for everything to finish running 
                            loss_.append(running_loss/len(traindataloader.sampler))
                            
                            torch.cuda.synchronize()
                            print()

                            print('epoch [{}/{}], loss:{:.4e}, time {:.3f} s {}'.format(epoch + 1, num_epochs, loss_[-1], start.elapsed_time(end)/1000,scheduler.get_lr()[0]))
                            Timecal.append(start.elapsed_time(end)/1000)

                            if epoch % 10 == 0:
                                for i in range(len(testnum)):
                                    plt.subplot(len(testnum), 4, 4*i+1)                # nrows=2, ncols=1, index=1
                                    plt.plot(wv,testshow[i])

                                    if i ==0: plt.title('input')
                                    if i ==len(testnum)-1: 
                                        plt.xticks(visible=True)
                                        plt.xlabel('wavelength (nm)') 
                                    else: plt.xticks(visible=False)


                                    #plt.xlabel('wavelength (nm)')
                                    plt.subplot(len(testnum), 4, 4*i+2)                # nrows=2, ncols=1, index=2
                                    plt.plot( wv,model(testshow.unsqueeze(1).cuda())[i].cpu().detach().squeeze())
                                    if i ==0: plt.title('output')
                                    if i ==len(testnum)-1: 
                                        plt.xlabel('wavelength (nm)')
                                        plt.xticks(visible=True)
                                    else: plt.xticks(visible=False)


                                    plt.subplot(len(testnum), 4, 4*i+3)                # nrows=2, ncols=1, index=2
                                    plt.plot(wv, testshow_label[i])

                                    if i ==0: plt.title('real output')
                                    if i ==len(testnum)-1: 
                                        plt.xlabel('wavelength (nm)')
                                        plt.xticks(visible=True)
                                    else: plt.xticks(visible=False)

                                    plt.subplot(len(testnum), 4, 4*i+4)                # nrows=2, ncols=1, index=2
                                    plt.plot(wv, (model(testshow.unsqueeze(1).cuda())[i].cpu().detach().squeeze()-testshow_label[i])/testshow_label[i])
                                    plt.ylim([-0.2, 0.2])
                                    if i ==0: plt.title('Normalized difference')
                                    if i ==len(testnum)-1: 
                                        plt.xlabel('wavelength (nm)')
                                        plt.xticks(visible=True)
                                    else: plt.xticks(visible=False)

                                plt.tight_layout()
                                plt.savefig(FPATH+'/tut_img/image_{}.png'.format(epoch))
                                plt.close('all')
                                plt.clf()

                            if epoch % 10 == 0:
                                model.eval()
                                eval_loss = 0.
                                for evaldata,eval_label in testloader:
                                    spec = evaldata
                                    label = eval_label
                                    if USE_CUDA:
                                        spec = spec.cuda()
                                        label = label.cuda()
                                    with torch.no_grad():
                                        out = model(spec.unsqueeze(1))
                                        loss = criterion(out.squeeze(), label)
                                    eval_loss += loss.item()*evaldata.shape[0]
                                evalloss_.append(eval_loss/len(testloader.sampler))
                                print(f'Test Loss: {evalloss_[-1]:.4e}\n') 

                        torch.save({
                                    'epoch': epoch,
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'loss': loss_,
                                    'testloss': evalloss_,
                                    'testnum': testnum, 'trainnum':trainnum,'Timecal':Timecal
                                    }, FPATH+'/tut.pth')


                        plt.figure(figsize=(20,10))
                        for i in range(len(testnum)):
                            plt.subplot(len(testnum), 4, 4*i+1)                # nrows=2, ncols=1, index=1
                            plt.plot(wv,testshow[i])

                            if i ==0: plt.title('input')
                            if i ==len(testnum)-1: 
                                plt.xticks(visible=True)
                                plt.xlabel('wavelength (nm)') 
                            else: plt.xticks(visible=False)


                            #plt.xlabel('wavelength (nm)')
                            plt.subplot(len(testnum), 4, 4*i+2)                # nrows=2, ncols=1, index=2
                            plt.plot( wv,model(testshow.unsqueeze(1).cuda())[i].cpu().detach().squeeze())
                            if i ==0: plt.title('output')
                            if i ==len(testnum)-1: 
                                plt.xlabel('wavelength (nm)')
                                plt.xticks(visible=True)
                            else: plt.xticks(visible=False)


                            plt.subplot(len(testnum), 4, 4*i+3)                # nrows=2, ncols=1, index=2
                            plt.plot(wv, testshow_label[i])

                            if i ==0: plt.title('real output')
                            if i ==len(testnum)-1: 
                                plt.xlabel('wavelength (nm)')
                                plt.xticks(visible=True)
                            else: plt.xticks(visible=False)

                            plt.subplot(len(testnum), 4, 4*i+4)                # nrows=2, ncols=1, index=2
                            plt.plot(wv, (model(testshow.unsqueeze(1).cuda())[i].cpu().detach().squeeze()-testshow_label[i])/testshow_label[i])
                            plt.ylim([-0.2, 0.2])
                            if i ==0: plt.title('Normalized difference')
                            if i ==len(testnum)-1: 
                                plt.xlabel('wavelength (nm)')
                                plt.xticks(visible=True)
                            else: plt.xticks(visible=False)

                        plt.tight_layout()
                        plt.savefig(FPATH+'/result.png',bbox_inches='tight')
                        plt.close('all')
                        plt.clf()

                        plt.figure(2)
                        plt.plot(range(1,len(loss_)+1),loss_)
                        #  plt.plot(loss_)
                        plt.title('Loss')
                        plt.xlabel('epoch')
                        plt.plot(range(1,len(evalloss_)*10+1,10),evalloss_)
                        plt.legend(['train loss','test loss'])
                        plt.yscale("log")
                        plt.savefig(FPATH+'/loss.png',bbox_inches='tight')
                        
                        plt.close('all')
                        plt.clf()



                        torch.cuda.empty_cache()

                        lastevalloss.append(evalloss_[-1])
                        f = open('./lastloss.txt', 'w')
                        f.write('  '.join(str(e) for e in lastevalloss))
                        f.close()


if __name__ == '__main__':
    main()






