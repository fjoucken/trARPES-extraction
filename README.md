# trARPES-extraction
## ML model for extracting electronic bands (or excitonic signals) from noisy time- and angle-resolved photoemission spectra (trARPES)

It is adapted from the paper by Peng et al. [Review of Scientific Instruments 91, 033905 (2020)](https://aip.scitation.org/doi/full/10.1063/1.5132586) to deal with trARPES spectra, which are much noisier than equilibrium ARPES spectra.

There are two files for producing simulated ARPES spectra () and two files for training the CNN.

The simulated ARPES spectra are obtained from a bilayer graphene lattice into which the hopping parameters are changed randomly, between values very far from the actual values for bilayer graphene so that the bands produced are very diverse. The E-k bands are computed with the [pybinding package] (https://docs.pybinding.site/en/stable/). The bands are obtained from two points chosen randomly within the first Brillouin zone.

![alt text](loss_vs_epoch.jpg)
