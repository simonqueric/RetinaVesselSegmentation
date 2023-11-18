# Retina Vessel Segmentation based on Frangi vesselness filter and UNet architecture.

I propose the following pipeline in order to perform vessel segmentation :
1) A preprocessing step which consists to enhance vessels structures with a Frangi filter.
2) A UNet architecture to learn a segmentation model. 

### This repo contains :
- A notebook about retina vessel enhancement with Frangi vesselness filter. 
- Python files to train a UNet network for performing segmentation.

### Future work

I plan to train a UNet model with a topological loss (see https://github.com/nstucki/Betti-matching/). It could avoid to have some holes or disconnected vessels in the segmentation.

### References : 
- [1] Z. Jadoon et al., Retinal Blood Vessels Segmentation using ISODATA and High Boost Filter
- [2]  A. Longo et al., Assessment of hessian-based Frangi vesselness filter in optoacoustic imaging, disponible ici : https://mediatum.ub.tum.de/doc/1600558/1600558.pdf
- [3] U-Net: Convolutional Networks for Biomedical Image Segmentation
- [4] https://github.com/nikhilroxtomar/Retina-Blood-Vessel-Segmentation-in-PyTorch/tree/main
