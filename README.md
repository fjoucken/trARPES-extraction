# trARPES-extraction
## ML model for extracting electronic bands (or excitonic signals) from noisy time- and angle-resolved photoemission spectra (trARPES)

It is adapted from the paper by [Peng et al., Review of Scientific Instruments 91, 033905 (2020)](https://aip.scitation.org/doi/full/10.1063/1.5132586) to deal with trARPES spectra, which are much noisier than equilibrium ARPES spectra.

The simulated ARPES spectra are obtained from a bilayer graphene lattice into which the hopping parameters are changed randomly, between values very far from the actual values for bilayer graphene so that the bands produced are very diverse. The E-k bands are computed with the [pybinding package](https://docs.pybinding.site/en/stable/). The bands are obtained from two points chosen randomly within the first Brillouin zone.

There are two files for producing simulated ARPES spectra (Feature_ARPES_generation_main.py and Feature_ARPES_generation_utils.py) and two files for training the CNN  model (Feature_ARPES_CNN_main.py and Feature_ARPES_CNN_utils.py).

Typically, a few thousands training spectra are enough to train a model. The models are UNet type. You can choose the type in Feature_ARPES_CNN_main.py (`model_num`).

I have applied this technique to trARPES on semiconductor. In this case, excited states in or close to the conduction band are separated from the valence band and I have trained two different models: one for the valence band bare band extraction and one for the excited state signal extraction.

### Valence band model

For the valence band model, here are the parameters I used for the making of the training data:

```
#Resolution in k and E of the spectra. Must be the same.
k_reso = 128
E_reso = 128
#number of bands to delete
num_bands_min_to_delete, num_bands_max_to_delete = 0,4
label = 0
#number of spectra to make
num_spectra = 10000
#parameters for the tight-binding
t0_min, t0_max = 1, 25      # [eV] nearest neighbour hopping
t1_min, t1_max, t3_min, t3_max, t4_min, t4_max = 0.5, 5, 0.5, 5, 0.5, 5
U_min, U_max = 0, 3
#The limits in the Brioullin zones for cutting to get the spectra
p1x_min, p2x_min, p1y_min, p2y_min = -20, -20, -20, -20
p1x_max, p2x_max, p1y_max, p2y_max = 20, 20, 20, 20
#HWHM of the gaussian for the Voigt, in pixels
alpha_min, alpha_max, gamma_min, gamma_max = 1,10,1,10
#For the added noise and curvature
#E_c and k_c is the position of the background hyperbole
E_c_min, E_c_max, k_c_min, k_c_max = 0, E_reso, 0, k_reso
#ratio between a and b of the hyperbole/parabole
a_b_ratio_min, a_b_ratio_max = 0, 2
#amplitude of the hyperbole/parabole
amp_min, amp_max = 0,1
#amplitude of the noise
noise_min, noise_max = 0.0, 0.05
#I didn't use it but it sets the temperature (used if one applies the Fermi Dirac statistics)
kT_min, kT_max = E_reso/1, E_reso/1
#If squeeze factor is 1, the bands take the whole y range. If it is 2, they take only half the range.
squeeze_min, squeeze_max = 1, 2
#Thickness gives the thickness of the bare band, in pixels.
#the thicker the bare band, the easier the model can be fitted.
# can be 3, 5, 7
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
```

For the model, I used the following parameters:

```
#pixels is the number of pixels. 128 is working fine.
pixels = 128
#num of filters in the CNN
num_filters = 4
#size of the kernel in the CNN
kernel_size = 3
#Choose the model
model_num = 3
#Number of epochs
no_epochs = 150
#batch size
batch_size = 20
#drop out rate
drop_out_rate = 0.0
#learning rate
learning_rate = 0.0001
#Loss should be MAE. BCE works as well but less well
loss = "MAE"   #BCE is binary cross entropy, "dice" for dice
```
The loss evolution during training was: 

![alt text](loss_vs_epoch_VB.jpg)

And here is how the model performs on simulated ARPES spectra: 

![alt text](Results_on_simu_VB.jpg)

### Excited states model

For the excited states, the parameters that has worked best for the making of simulated spectra were:
```
#Resolution in k and E of the spectra. Must be the same.
k_reso = 128
E_reso = 128
#number of bands to delete
num_bands_min_to_delete, num_bands_max_to_delete = 2,4
label = 0
#number of spectra to make
num_spectra = 10000
#parameters for the tight-binding
t0_min, t0_max = 1, 25      # [eV] nearest neighbour hopping
t1_min, t1_max, t3_min, t3_max, t4_min, t4_max = 0.5, 5, 0.5, 5, 0.5, 5
U_min, U_max = 0, 3
#The limits in the Brioullin zones for cutting to get the spectra
p1x_min, p2x_min, p1y_min, p2y_min = -20, -20, -20, -20
p1x_max, p2x_max, p1y_max, p2y_max = 20, 20, 20, 20
#HWHM of the gaussian for the Voigt, in pixels
alpha_min, alpha_max, gamma_min, gamma_max = 5,15,5,15
#For the added noise and curvature
#E_c and k_c is the position of the background hyperbole
E_c_min, E_c_max, k_c_min, k_c_max = 0, E_reso, 0, k_reso
#ratio between a and b of the hyperbole/parabole
a_b_ratio_min, a_b_ratio_max = 0, 2
#amplitude of the hyperbole/parabole
amp_min, amp_max = 0,1
#amplitude of the noise
noise_min, noise_max = 0.0, 0.05
#I didn't use it but it sets the temperature (used if one applies the Fermi Dirac statistics)
kT_min, kT_max = E_reso/1, E_reso/1
#If squeeze factor is 1, the bands take the whole y range. If it is 2, they take only half the range.
squeeze_min, squeeze_max = 1, 2
#Thickness gives the thickness of the bare band, in pixels.
#the thicker the bare band, the easier the model can be fitted.
# can be 3, 5, 7
thick = 3
#here I choose if I want the Fermi edge randomly inverted
#if set to 0, make 50% of the spectra with Fermi edge inverted.
#if set to 1, Fermi edge always in the same direction
#that is for the excited states.
#I actually don't use it now because I use high temperature and 
inver = 1
#the minimum of the Gaussian decay length for the Gaussian that is multiplied by the band(to vary the band intensity)
#for the 1 band case (excited) I used k_reso/20.
g_decay_min = int(k_reso/20)
```

And the model used for the excited states was:
```
#pixels is the number of pixels. 128 is working fine.
pixels = 128
#num of filters in the CNN
num_filters = 8
#size of the kernel in the CNN
kernel_size = 5
#Choose the model
model_num = 4
#Number of epochs
no_epochs = 150
#batch size
batch_size = 20
#drop out rate
drop_out_rate = 0.0
#learning rate
learning_rate = 0.0001
#Loss should be MAE. BCE works as well but less well
loss = "MAE"   #BCE is binary cross entropy, "dice" for dice
```

The loss during training looked like this:

![alt text](loss_vs_epoch_excited.jpg)

And here is how the model performs on simulated ARPES spectra: 

![alt text](Results_on_simu_excited.jpg)

Note that the main differences between the data used for the valence band and the excited state were:
1. The width of the spectrum (Voigt profile) was larger for the excited states.
2. Only 0 to 2 bands were used for the excited states (0 to 4 for the valence band)
3. The width of the bare band (label) was larger for the excited states (5 pixels vs 3 pixels for the valence band)

### Results on experimental trARPES data

To apply the models to experimental data, you must first extract the regions of interest in the spectra you want to analyze and then apply the corresponding model to the regions of interest. 
In the figure below, the models are applied to an experimental trARPES spectrum. The spectra has been obtained on bulk MoSe~2. Details on the experimental setup can be found [here](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/10102/101020I/Ultrafast-extreme-ultraviolet-ARPES-studies-of-electronic-dynamics-in-two/10.1117/12.2251249.full?SSO=1) and [here](https://aip.scitation.org/doi/full/10.1063/1.5079677). The trARPES spectrum is on the green scale while the bare bands extracted from the excited states is in red and the bare bands extracted from the valence band is in blue: 

![alt text](Fit_exp_data.jpg)


