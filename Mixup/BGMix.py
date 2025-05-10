import numpy as np
import random


def mix(trials_a,trials_b,nt=2):
    # template_a=np.mean(trials_a,axis=0,keepdims=True).repeat(nt,0)*np.power(nt,.5)
    # template_b=np.mean(trials_b,axis=0,keepdims=True).repeat(nt,0)*np.power(nt,.5)

    template_a=np.mean(trials_a,axis=0,keepdims=True).repeat(nt,0)
    template_b=np.mean(trials_b,axis=0,keepdims=True).repeat(nt,0)

    bg_a=trials_a-template_a
    bg_b=trials_b-template_b

    gen_a=template_a+bg_b
    gen_b=template_b+bg_a

    return gen_a,gen_b
    

def block_mix(block_a,block_b,nt=2):
    nTrials,nChan,nSamp=block_a.shape

    rnd_ind=[i for i in range(nTrials)]
    
    random.shuffle(rnd_ind)
    block_a=block_a[rnd_ind,:,:]

    random.shuffle(rnd_ind)
    block_b=block_b[rnd_ind,:,:]

    build_block_a=np.zeros([nTrials,nChan,nSamp])
    build_block_b=np.zeros([nTrials,nChan,nSamp])

    for ns in range(0,nTrials,nt):
        trials=[i%nTrials for i in range(ns,ns+nt)]
        

        trials_a=block_a[trials,:,:]
        trials_b=block_b[trials,:,:]

        gen_a,gen_b=mix(trials_a,trials_b,nt)

        build_block_a[trials,:,:]=gen_a
        build_block_b[trials,:,:]=gen_b
  

    return build_block_a,build_block_b


def generate_data(X,y,nt=[2,2,2]):

    # X: all data for 1 subject
    labels=np.unique(y)

    gen_X=[]
    gen_y=[]

    # iter all label
    for y1 in labels:
        # get data with label y1
        ind=np.where(y==y1)
        block1=X[ind]

        for k in nt:
            # get another label
            y2=random.sample(list(labels),1)[0]
            while y2==y1:
                y2=random.sample(list(labels),1)[0]

            # get data with label y2
            ind_=np.where(y==y2)
            block2=X[ind_]

            gen_block,_=block_mix(block1,block2,k)

            for i in range(gen_block.shape[0]):
                gen_X.append(gen_block[i])
                gen_y.append((y1,y2))
                
    return gen_X,gen_y


def generate_subs_data(X,y,subs,meta,nt=[2,2,2]):

    gen_X,gen_y=[],[]
    for sub in subs:
        sub_mask = (meta['subject']==sub).to_numpy()
        sub_X=X[sub_mask]
        sub_y=y[sub_mask]

        sub_gen_X,sub_gen_y=generate_data(sub_X,sub_y,nt)

        gen_X+=sub_gen_X
        gen_y+=sub_gen_y

    return gen_X,gen_y