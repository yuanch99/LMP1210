from PIL import Image
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import tensorflow as tf

path = '/hpf/largeprojects/tabori/users/yuan/lmp1210/data/unet/'

def load_image( path ) :
    img = Image.open( path )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

def save_image( npdata, outpath ) :
    img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
    img.save( outfilename )
    return None
    
def load_train():
    figure_list = glob.glob(path+'train_val/figure/train/*.jpg')
    resize=256
    figure=[]
    label=[]
    for fname in figure_list:
        figure.append(np.array(Image.open(fname).resize((resize,resize), Image.ANTIALIAS)))
    label_list = glob.glob(path+'train_val/label/train/*.png')
    for fname in label_list:
        label.append(np.array(Image.open(fname).resize((resize,resize), Image.ANTIALIAS)))
    return np.array(figure),np.array(label)

def load_val():
    figure_list = glob.glob(path+'train_val/figure/val/*.jpg')
    resize=256
    figure=[]
    label=[]
    for fname in figure_list:
        figure.append(np.array(Image.open(fname).resize((resize,resize), Image.ANTIALIAS)))
    label_list = glob.glob(path+'train_val/label/val/*.png')
    for fname in label_list:
        label.append(np.array(Image.open(fname).resize((resize,resize), Image.ANTIALIAS)))
    return np.array(figure),np.array(label)

def load_test():
    resize=256
    figure=[]
    label=[]
    for i in range(1,54):
        figure.append(np.array(Image.open(f'/hpf/largeprojects/tabori/users/yuan/lmp1210/data/unet/test/JPEGImages/{i}.jpg').resize((resize,resize), Image.ANTIALIAS)))
    for i in range(1,54):
        label.append(np.array(Image.open(f'/hpf/largeprojects/tabori/users/yuan/lmp1210/data/unet/test/SegmentationClassPNG/{i}.png').resize((resize,resize), Image.ANTIALIAS).convert('RGB')))
    return np.array(figure),np.array(label)

def load_test2():
    figure_list = glob.glob(path+'test/*.tif')
    resize=256
    figure=[]
    for fname in figure_list:
        figure.append(np.array(Image.open(fname).resize((resize,resize), Image.ANTIALIAS)))
    return np.array(figure)

def load_data():
    resize=256
    figure=[]
    label=[]
    for i in range(1,152):
        figure.append(np.array(Image.open(f'/hpf/largeprojects/tabori/users/yuan/lmp1210/data/unet/train_val/JPEGImages/{i}.jpg').resize((resize,resize), Image.ANTIALIAS)))
    for i in range(1,152):
        label.append(np.array(Image.open(f'/hpf/largeprojects/tabori/users/yuan/lmp1210/data/unet/train_val/SegmentationClassPNG/{i}.png').resize((resize,resize), Image.ANTIALIAS).convert('RGB')))
    return np.array(figure),np.array(label)

def load_data2():
    resize=256
    figure=[]
    label=[]
    for i in range(1,152):
        figure.append(np.array(Image.open(f'/hpf/largeprojects/tabori/users/yuan/lmp1210/data/train_val/JPEGImages/{i}.jpg').resize((resize,resize), Image.ANTIALIAS)))
    for i in range(1,152):
        label.append(np.array(Image.open(f'/hpf/largeprojects/tabori/users/yuan/lmp1210/data/train_val/SegmentationClassPNG/{i}.png').resize((resize,resize), Image.ANTIALIAS)))
    return np.array(figure),np.array(label)


def display_sample(display_list,name):
    """Show side-by-side an input image,
    the ground truth and the prediction.
    """
    plt.figure(figsize=(18, 18))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.savefig(name,dpi=300)

def threshold(fig_raw,nci,gci):
    fig=fig_raw.copy()
    fig0=fig[...,0].flatten()
    fig1=fig[...,1].flatten()
    fig2=fig[...,2].flatten()
    for i in range(len(fig0)):
        if fig0[i]<nci:
            fig0[i]=0
        elif fig0[i]>nci:
            fig0[i]=1
        if fig1[i]<gci:
            fig1[i]=0
        elif fig1[i]>gci:
            fig1[i]=1
        if (fig0[i]>=nci or fig1[i]>=gci):
            fig2[i]=0
        if (fig0[i]<nci and fig1[i]<gci):
            fig2[i]=1
    fig[...,0]=np.reshape(fig0,np.shape(fig[...,0]))
    fig[...,1]=np.reshape(fig1,np.shape(fig[...,0]))
    fig[...,2]=np.reshape(fig2,np.shape(fig[...,0]))
    return fig
