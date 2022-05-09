import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz

from math import sqrt, pi


a = 0.24595   # [nm] unit cell length
a_cc = 0.142  # [nm] carbon-carbon distance

#angle
theta = 0*180/pi
#I define the basis vectors
a1 = np.array([a, 0, 0])
a2 = np.array([a/2, a/2 * sqrt(3), 0])
sub_A1 = np.array([0, a_cc, 0])
sub_B1 = np.array([0, 0, 0])
sub_A2 = np.array([0, 0, -0.335])
sub_B2 = np.array([0, -a_cc, -0.335])

#I rotate the lattice vectors by theta
rot_matrix = np.array([[np.cos(theta), -np.sin(theta),0],
            [np.sin(theta), np.cos(theta),0],
            [0, 0, 1]])
a1 = np.matmul(rot_matrix, a1)
a2 = np.matmul(rot_matrix, a2)
sub_A1 = np.matmul(rot_matrix, sub_A1)
sub_B1 = np.matmul(rot_matrix, sub_B1)
sub_A2 = np.matmul(rot_matrix, sub_A2)
sub_B2 = np.matmul(rot_matrix, sub_B2)

# t0 = 3.16      # [eV] nearest neighbour hopping
# t1 = 0.42
# t3 = 0.38
# t4 = 0.14
def bilayer_graphene(t0, t1, t3, t4):
    
    lat = pb.Lattice(a1,
                    a2)
    lat.add_sublattices(('A1', sub_A1),
                        ('B1', sub_B1),
                        ('A2', sub_A2),
                        ('B2', sub_B2))
    lat.add_hoppings(
        # inside the main cell
        ([0,  0], 'A1', 'B1', t0),
        ([0,  0], 'A2', 'B2', t0),
        ([0,  0], 'A2', 'B1', t1),
        # between neighboring cells
        ([1, -1], 'B1', 'A1', t0),
        ([0, -1], 'B1', 'A1', t0),
        ([1, -1], 'B2', 'A2', t0),
        ([0, -1], 'B2', 'A2', t0),
        #t3
        ([0,  -1], 'B2', 'A1', t3),
        ([1,  -1], 'B2', 'A1', t3),
        ([1,  -2], 'B2', 'A1', t3),
        #t4
        ([0,  0], 'A2', 'A1', t4),
        ([0,  -1], 'A2', 'A1', t4),
        ([1,  -1], 'A2', 'A1', t4),
    )
    # lat.plot_brillouin_zone()
    # plt.show()
    return lat
'''
plt.rcParams['figure.figsize'] = [7, 7]
lattice = bilayer_graphene()
lattice.plot()
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()
'''
def gap(delta):
    """Break sublattice symmetry with opposite A and B onsite energy"""
    @pb.onsite_energy_modifier
    def potential(energy, sub_id):
        energy[sub_id == 'A1'] += delta/2
        energy[sub_id == 'B1'] += delta/2
        energy[sub_id == 'A2'] -= delta/2
        energy[sub_id == 'B2'] -= delta/2
        return energy
    return potential


def get_spectrum_and_label(energies, E_reso, k_reso, alpha, gamma, kT, squeeze, thick, inver, g_decay_min):
    spectrum = np.zeros([E_reso, k_reso])
    #each band will be weighted by an exponential along the k direction
    #the HWHM of this exponential is given here, in pixels
    num_bands = energies.shape[1]
    g_decays = np.random.randint(g_decay_min, k_reso, size=num_bands)
    g_centers = np.random.randint(1, k_reso, size=num_bands)
    #I rescale the enegies. 
    energy_scale = np.arange(0,E_reso)
    #first i rescale the energies between [0,E_reso] so everything is expressed in pixels
    #if squeezing factor = 2, it means the bands will occupy only half the space
    #I choose randomly the offset
    E_start = np.random.randint(0, E_reso - E_reso/squeeze + 1)
    energies = E_start + normalize(energies, E_reso/squeeze-1)
    #now I choose E_F so that it is within the bands (+/-2 pixels)
    E_F = np.random.randint(E_start - 2, E_start + E_reso/squeeze-1 + 2)

    for i in range(energies.shape[0]):
        for j in range(energies.shape[1]):
            #spectrum[:,i] +=imS/((energy_scale - energies[i,j])**2 + imS**2)
            spectrum[:,i] += V(energy_scale - energies[i,j], alpha = alpha, gamma = gamma) * G(i - g_centers[j],g_decays[j])
    #now I get the label, weighted by the same exponential
    label = np.zeros([E_reso, k_reso])
    for i in range(energies.shape[0]):
        for j in range(energies.shape[1]):
            energy = int(np.floor(energies[i,j]))
            label[energy,i] = 1 * G(i - g_centers[j],g_decays[j])
            if thick >= 3:
                if (energy + 1) < E_reso:
                    label[energy + 1,i] = 1 * G(i - g_centers[j],g_decays[j])
            if thick >= 5:
                if (energy + 2) < E_reso:
                    label[energy + 2,i] = 1 * G(i - g_centers[j],g_decays[j])
            if thick >= 7:
                if (energy + 3) < E_reso:
                    label[energy + 3,i] = 1 * G(i - g_centers[j],g_decays[j])
            if thick >= 3:
                if (energy - 1) > -1:
                    label[energy - 1,i] = 1 * G(i - g_centers[j],g_decays[j])
            if thick >= 5:
                if (energy - 2) > -2:
                    label[energy - 2,i] = 1 * G(i - g_centers[j],g_decays[j])
            if thick >= 7:
                if (energy - 3) > -3:
                    label[energy - 3,i] = 1 * G(i - g_centers[j],g_decays[j])
    #I normalize the spectra
    spectrum = normalize(spectrum, 1)
    label = normalize(label, 1)
    #Here I apply FD statistic
    #spectrum /= (np.exp((energy_scale[:, np.newaxis]-E_F)/kT)+1)
    #label /= (np.exp((energy_scale[:, np.newaxis]-E_F)/kT)+1)
    #I rotate the spectra and label by 180 so that 
    #I choose randomly if I apply it in one direction of the energy axis or the other
    #for this, I switch or not randomly the spectrum
    inv = np.random.randint(inver, high = 1 + 1)
    if inv == 1:
        spectrum = np.rot90(spectrum, k=2)
        label = np.rot90(label, k=2)
    #I normalize the spectrum to 1, same for label
    return spectrum, label

def add_bg_noise(spectrum, E_reso, k_reso, E_c, k_c, hyper, inv, a_b_ratio, amp, noise):
    #this function adds a paraboloid to an STM image to simulate the bumps seen in experimental images.
    #E_reso and k_reso are the number of pixels in x and y directions
    #E_c and k_c are the center of the paraboloid
    #a_b_ratio defines the curvature ratio between x and y.
    #if hyper = 1, it is a hyperbolic paraboloid https://en.wikipedia.org/wiki/Paraboloid
    #if inv = -1, it is upside-down
    #amp defines the amplitude of the backgroun (note spectrum is supposed to be normalized)
    x = np.arange(-k_reso/2, k_reso/2, 1)
    y = np.arange(-E_reso/2, E_reso/2, 1)
    xx, yy = np.meshgrid(x, y)
    a = 1
    b = a * a_b_ratio
    z = (-1)**inv * (((xx - k_c)/a)**2 + (-1)**hyper * ((yy - E_c)/b)**2)
    z = normalize(z, amp)
    spectrum += z
    return add_noise(spectrum, noise, 1)

def add_noise(spectrum, noise_amplitude, amplitude):
    max_ = np.max(spectrum)
    min_ = np.min(spectrum)
    scaled_spectrum = amplitude * ((spectrum - min_)/(max_ - min_))
    noise = np. random. normal(0, noise_amplitude, scaled_spectrum.shape)
    noisy_spectrum = scaled_spectrum + noise
    return normalize(noisy_spectrum, amplitude)

def V(x, alpha, gamma):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component HWHM alpha.

    """
    sigma = alpha / np.sqrt(2 * np.log(2))

    return np.real(wofz((x + 1j*gamma)/sigma/np.sqrt(2))) / sigma /np.sqrt(2*np.pi)

def G(x, alpha):
    """ Return Gaussian line shape at x with HWHM alpha """
    return np.sqrt(np.log(2) / np.pi) / alpha * np.exp(-(x / alpha)**2 * np.log(2))

def get_label(energies, E_reso, k_reso):
    label = np.zeros([E_reso, k_reso])
    #first i rescale the energies between [0,E_reso] so everything is expressed in pixels
    energies = normalize(energies, E_reso-1)
    for i in range(energies.shape[0]):
        for j in range(energies.shape[1]):
            energy = int(np.floor(energies[i,j]))
            label[energy,i] = 1
            if (energy + 1) < E_reso:
                label[energy + 1,i] = 1
            if (energy - 1) > -1:
                label[energy - 1,i] = 1
    return label

def normalize(array, new_max):
    max_ = np.max(array)
    min_ = np.min(array)
    array = new_max*((array - min_)/(max_ - min_))
    return array

def resize_energies(energies, num_bands_to_delete):
    # print("num_bands_to_delete:", num_bands_to_delete)
    for i in range(num_bands_to_delete):
        energies= np.delete(energies, np.random.randint(0, high = 3 + 1 - i), 1)
        #print('size energies:', energies.shape)
    return energies
