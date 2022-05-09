import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz
import cv2
import os.path
import matplotlib
from math import sqrt, pi
import winsound
import time

from Feature_ARPES_generation_utils import get_spectrum_and_label, get_label, add_bg_noise, bilayer_graphene, gap, resize_energies

save_png = True
train_or_test = 'apply'
folder_name = 'all_12/'
folder_name_X = folder_name+train_or_test+"/X"
#The path for saving:
save_path_X = "D:/Machine_learning/Generated_training_data/ARPES/Feature_extraction/"+folder_name_X+"/txt/"
save_path_jpeg_X = "D:/Machine_learning/Generated_training_data/ARPES/Feature_extraction/"+folder_name_X+"/jpeg/"
folder_name_Y = folder_name+train_or_test+"/Y"
#The path for saving:
save_path_Y = "D:/Machine_learning/Generated_training_data/ARPES/Feature_extraction/"+folder_name_Y+"/txt/"
save_path_jpeg_Y = "D:/Machine_learning/Generated_training_data/ARPES/Feature_extraction/"+folder_name_Y+"/jpeg/"
#I determine reso in and E
k_reso = 128
E_reso = 128
#number of bands
num_bands_min_to_delete, num_bands_max_to_delete = 0,4
label = 0
#number of spectra to make
num_spectra = 20
#parameters limits
t0_min, t0_max = 1, 25      # [eV] nearest neighbour hopping
t1_min, t1_max, t3_min, t3_max, t4_min, t4_max = 0.5, 5, 0.5, 5, 0.5, 5
U_min, U_max = 0, 3
p1x_min, p2x_min, p1y_min, p2y_min = -20, -20, -20, -20
p1x_max, p2x_max, p1y_max, p2y_max = 20, 20, 20, 20
#HWHM of the gaussian for the Voigt, in pixels
alpha_min, alpha_max, gamma_min, gamma_max = 1,10,1,10
#For the added noise and curvature
E_c_min, E_c_max, k_c_min, k_c_max = 0, E_reso, 0, k_reso
a_b_ratio_min, a_b_ratio_max = 0, 2
amp_min, amp_max = 0,1
noise_min, noise_max = 0.0, 0.05
kT_min, kT_max = E_reso/1, E_reso/1
#squeeze factor
squeeze_min, squeeze_max = 1, 2
#thickness gives the thickness of the bare band
# 3, 5, 7
thick = 3
#here I choose if I want the Fermi edge randomly inverted
#if set to 0, make 50% of the spectra with Fermi edge inverted.
#if set to 1, Fermi edge always in the same direction
#that is for the excited states.
#I actually don't use it now because I use high temperature and 
inver = 1
#the minimum of the Gaussian decay length for the Gaussian that is multiplied by the band(to vary the band intensity)
#for the 1 band case (excited) I used k_reso/20.
g_decay_min = int(k_reso/5)





tic = time.time()
for i in range(num_spectra):
    #I set the parameters
    print("I am making spectrum number ", i)
    t0 = np.random.uniform(low = t0_min, high = t0_max)
    t1 = np.random.uniform(low = t1_min, high = t1_max)
    t3 = np.random.uniform(low = t3_min, high = t3_max)
    t4 = np.random.uniform(low = t4_min, high = t4_max)
    U = np.random.uniform(low = U_min, high = U_max)

    p1x = np.random.uniform(low = p1x_min, high = p1x_max)
    p1y = np.random.uniform(low = p1y_min, high = p1y_max)
    p2x = np.random.uniform(low = p2x_min, high = p2x_max)
    p2y = np.random.uniform(low = p2y_min, high = p2y_max)
    p1 = [p1x, p1y]
    p2 = [p2x, p2y]

    alpha = np.random.uniform(low = alpha_min, high = alpha_max)
    gamma = np.random.uniform(low = gamma_min, high = gamma_max)

    E_c = np.random.uniform(low = E_c_min, high = E_c_max)
    k_c = np.random.uniform(low = k_c_min, high = k_c_max)
    hyper = np.random.randint(0, high = 1 + 1)
    inv = np.random.randint(0, high = 1 + 1)
    a_b_ratio = np.random.uniform(low = 1/a_b_ratio_max, high = a_b_ratio_max)
    amp = np.random.uniform(low = amp_min, high = amp_max)
    noise = np.random.uniform(low = noise_min, high = noise_max)
    
    kT = np.random.uniform(low = kT_min, high = kT_max)

    squeeze = np.random.uniform(low = squeeze_min, high = squeeze_max)
    model = pb.Model(
        bilayer_graphene(t0 = t0, t1 = t1, t3 = t3, t4 = t4), 
        pb.translational_symmetry(),
        gap(delta = U)
    )
    solver = pb.solver.lapack(model)


    parameters_string_X = "t0_"+str(np.round(t0,2)).zfill(3)+"t1_"+str(np.round(t1,2)).zfill(3)+\
        "_t3_"+str(np.round(t3,2)).zfill(3)+"_t4_"+str(np.round(t4,2)).zfill(3)+"_V_"+\
            "_U_"+str(np.round(t3,2)).zfill(3)+"_k_reso_"+str(k_reso)+"_E_reso_"+str(E_reso)+\
                "_p1x_"+str(np.round(p1[0],2)).zfill(3)+"_p1y_"+str(np.round(p1[1],2)).zfill(3)+\
                    "_p2x_"+str(np.round(p2[0],2)).zfill(3)+"_p1y_"+str(np.round(p2[1],2)).zfill(3)

    dist = sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    step = dist/k_reso
    bands = solver.calc_bands(p1, p2, step=step)
    #bands.plot(point_labels=['K1','K2'])

    energies = getattr(bands,'energy')
    # print (getattr(bands,'energy')) 
    # print (getattr(bands,'k_path')) 
    # plt.plot(energies)
    # plt.show()
    # print('energies[1,:] before resize:',energies[1,:])
    #I suppress 1, 2, 3 or 4 bands
    num_bands_to_delete = np.random.randint(num_bands_min_to_delete, high = num_bands_max_to_delete + 1)
    energies_resized = resize_energies(energies, num_bands_to_delete)
    #I resize the energies to have the right size 
    if num_bands_to_delete != 4:
        energies = cv2.resize(energies_resized, dsize=(energies_resized.shape[1], k_reso), interpolation=cv2.INTER_CUBIC)
    # plt.plot(energies)
    # plt.show()
    # print('energies[1,:] after resize:',energies[1,:])

    if num_bands_to_delete != 4:
        spectrum, label_Y = get_spectrum_and_label(energies, 
                                                E_reso = E_reso, 
                                                k_reso = k_reso, 
                                                alpha = alpha, 
                                                gamma = gamma,
                                                kT = kT, 
                                                squeeze = squeeze,
                                                thick = thick, 
                                                inver = inver, 
                                                g_decay_min = g_decay_min)
    else:
        spectrum, label_Y = np.zeros([E_reso, k_reso]), np.zeros([E_reso, k_reso])

    #label_Y = get_label(energies, E_reso = E_reso, k_reso = k_reso)

    # im = plt.imshow(spectrum, origin='lower')
    # cbar = plt.colorbar(im)
    # plt.show()
    # im = plt.imshow(label_Y, origin='lower')
    # plt.show()

    spectrum_w_bg = add_bg_noise(spectrum, E_reso, k_reso, E_c = E_c, k_c = k_c, hyper=hyper,
                                    inv=inv, a_b_ratio=a_b_ratio, amp=amp, noise = noise)
    # im = plt.imshow(spectrum_w_bg, origin='lower')
    # cbar = plt.colorbar(im)
    # plt.show()

    #then I save the image
    base_name = "ARPES_X_label_"+str(label)+"_MLG_"+ parameters_string_X 
    name_of_file = base_name
    completeName = os.path.join(save_path_X, name_of_file+".txt")
    file = open(completeName,'wb')
    np.savetxt(file, spectrum_w_bg, header='label='+str(label),delimiter='\t')
    file.close()
    #then save the png
    if save_png == True:
        completeName = os.path.join(save_path_jpeg_X, name_of_file+".png")
        matplotlib.image.imsave(completeName, spectrum_w_bg, vmin = 0, vmax = 1)
    ########################
    #Then I save the Y label
    base_name = "ARPES_Y_label_"+str(label)+"_MLG_"+parameters_string_X 
    name_of_file = base_name
    completeName = os.path.join(save_path_Y, name_of_file+".txt")
    file = open(completeName,'wb')
    np.savetxt(file, label_Y, header='label='+str(label),delimiter='\t')
    file.close()
    #then save the png
    if save_png == True:
        completeName = os.path.join(save_path_jpeg_Y, name_of_file+".png")
        matplotlib.image.imsave(completeName, label_Y, vmin = 0, vmax = 1)
    label += 1


toc = time.time()
winsound.Beep(800, 1000)
print("Time for computing all this stuff:")
print(toc-tic)