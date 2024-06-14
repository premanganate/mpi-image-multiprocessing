import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import time
import shutil

start=time.time()

def load_data():    
    clean_fp='Ground_truth'
    noisy_fp='Noisy_folder'
    clean_img_paths=[]
    noisy_img_paths=[]
    clean_dir=os.listdir(clean_fp)
    noisy_dir=os.listdir(noisy_fp)

    for i in range(0,300):
        clean=os.path.join(clean_fp,clean_dir[i])
        noisy=os.path.join(noisy_fp,noisy_dir[i])
        clean_img_paths.append(clean)
        noisy_img_paths.append(noisy)
    return clean_img_paths,noisy_img_paths

def img_processing(clean,noisy):    
    clean_img=cv2.imread(clean) #expected clean image
    noisy_img=cv2.imread(noisy) #input noisy image
    cleaned_img=cv2.medianBlur(noisy_img,5) #actual cleaned image
    blurred_img=cv2.GaussianBlur(cleaned_img,(5,5),0)
    edges=cv2.Canny(blurred_img,50,150)
    edge_img=cv2.add(edges,127)
    return noisy_img,clean_img,cleaned_img,edge_img
    

def display_img(noisy,clean,cleaned,edges,rank): 
        os.makedirs("mpi_res",exist_ok=True)
        for i in range(len(noisy)):
                plt.figure(figsize=(10,5))
                plt.subplot(1,4,1)
                plt.axis("off")
                plt.imshow(noisy[i])
                plt.title("Input Noisy Image")

                plt.subplot(1,4,2)
                plt.axis("off")
                plt.imshow(clean[i])
                plt.title("Expected Cleaned Image")

                plt.subplot(1,4,3)
                plt.axis("off")
                plt.imshow(cleaned[i])
                plt.title("Actual Cleaned Image")

                plt.subplot(1,4,4)
                plt.axis("off")
                plt.imshow(edges[i])
                plt.title("Extracted edges")
                save_path="mpi_res/"+str(rank)+"_result_"+str(i)+'.png'
                plt.savefig(save_path) 
                plt.close()

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()
total=300
workload=total//size
start_index=rank*workload
if rank==(size-1):
    end_index=300
else:
    end_index=start_index+workload
clean_img_paths, noisy_img_paths = load_data()
clean_paths,noisy_paths=clean_img_paths[start_index:end_index],noisy_img_paths[start_index:end_index]
noisy_img=[]
clean_img=[]
cleaned_img=[]
edge_img=[]
for i in range(len(clean_paths)):
    noisy,clean,cleaned,edges=img_processing(clean_paths[i],noisy_paths[i])
    noisy_img.append(noisy)
    clean_img.append(clean)
    cleaned_img.append(cleaned)
    edge_img.append(edges)
display_img(noisy_img,clean_img,cleaned_img,edge_img,rank)
end=time.time()
exec_time=end-start
exec=comm.gather(exec_time,root=0)
if rank==0:
    print(exec)
    for i in range(len(exec)):
        print("rank ",i,": ",exec[i])
    print("average time: ",(sum(exec)/size))
    print("max time: ",(max(exec)))
MPI.Finalize()