# TESS-Prediction

Prototype for predicting the probability of visually detectable oscillations in frequency power spectra of red giants. The function TESS_Prediction_Main.py writes a file with the output probabilities and predicted frequency at maximum power (nu_max). The main columns of interest in the output file are the following:

'file' - Input power density filename  
'prob' - Averaged confidence score that oscillations are present. Ranges from 0 to 1  
'prob_std' - Deviation of 'prob'  
'det' - 0 if no oscillations; 1 if oscillations are detected  
'numax' - estimated frequency at maximum power in muHz  
'numax_std' - deviation of 'numax'  

___
**To run the script download this folder and run the following command within the folder:**

python TESS_Prediction_Main.py --psd_folder 'your_psd_folder_here' --output_file "name_of_the_output_file"    

___
I have ran this code using Keras Version 2.2.4 and Pytorch Version 1.1.0.    

Note that these pre-trained networks are only a prototype as I'm continuously working on improving them to better predict on TESS data. I will also be working on a full Pytorch implementation. 
