import numpy as np
import keras.backend as K
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))
from keras.models import load_model

from scipy.interpolate import interp1d
from scipy.stats import binned_statistic
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import pandas as pd
import os, keras, csv, torch
import matplotlib.pyplot as plt


class MDN(nn.Module):
    """A mixture density network layer
    The input maps to the parameters of a MoG probability distribution, where
    each Gaussian has O dimensions and diagonal covariance.
    Arguments:
        in_features (int): the number of dimensions in the input
        out_features (int): the number of dimensions in the output
        num_gaussians (int): the number of Gaussians per output dimensions
    Input:
        minibatch (BxD): B is the batch size and D is the number of input
            dimensions.
    Output:
        (pi, sigma, mu) (BxG, BxGxO, BxGxO): B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions for each
            Gaussian. Pi is a multinomial distribution of the Gaussians. Sigma
            is the standard deviation of each Gaussian. Mu is the mean of each
            Gaussian.
    """

    def __init__(self, in_features, out_features, num_gaussians, temperature=1):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.temperature = temperature
        self.pi = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_gaussians),
        )
        self.sigma = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, out_features * num_gaussians)
        )

        self.mu = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, out_features * num_gaussians),
            nn.Softplus(threshold=5)
        )
        self.elu = nn.ELU()

    def forward(self, minibatch):
        pi = self.pi(minibatch)
        pi = F.softmax(pi / self.temperature, dim=1)
        sigma = self.elu(self.sigma(minibatch)) + 1
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)
        mu = self.mu(minibatch)
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        return pi, sigma, mu

class SLOSH_Regressor(nn.Module):
    def __init__(self, num_gaussians):
        super(SLOSH_Regressor, self).__init__()

        self.conv1 = nn.Conv2d(1, 4, kernel_size=5, padding=2) # same padding 2P = K-1
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1) # same padding 2P = K-1
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1) # same padding 2P = K-1
        self.pool1 = nn.MaxPool2d(2,2)
        self.pool2 = nn.MaxPool2d(2,2)
        self.pool3 = nn.MaxPool2d(2,2)
        self.drop1 = nn.Dropout(0.5)
        self.linear1 = nn.Linear(16*16*16, 128)
        self.mdn = MDN(in_features=128, out_features=1, num_gaussians=num_gaussians)

    def print_instance_name(self):
        print (self.__class__.__name__)

    def forward(self, input_image):
        conv1 = F.leaky_relu(self.conv1(input_image.unsqueeze(1)), negative_slope=0.1) # (N, C, H, W)
        conv1 = self.pool1(conv1)
        conv2 = F.leaky_relu(self.conv2(conv1), negative_slope=0.1)
        conv2 = self.pool2(conv2)
        conv3 = F.leaky_relu(self.conv3(conv2), negative_slope=0.1)
        conv3 = self.pool3(conv3)
        conv3 = self.drop1(conv3)

        linear1 = F.relu(self.linear1(conv3.view(conv3.size()[0], -1)))
        pi, sigma, mu = self.mdn(linear1)
        return pi, sigma, mu

def weighted_mean_squared_error(y_true, y_pred):
    return K.mean((K.square(y_pred - y_true))*K.square(y_true-64), axis=-1)

def dist_mu(pi, mu):
    """Calculate the mean of a mixture.
    """
    if pi.size() != mu.size():
        pi = pi.unsqueeze(2)
    return torch.sum(pi*mu, dim=1)

def dist_var_npy(pi, mu, mixture_mu, sigma):
    """Calculate the second moment (variance) of a bimodal distribution
    mu is the tensor while mixture_mu is the mean of the entire mixture as calculated by dist_mu
    """
    if pi.shape != mu.shape:
        pi = np.expand_dims(pi, 2)
    if mixture_mu.shape != mu.shape:
        mixture_mu = np.expand_dims(mixture_mu, -1)
    delta_square =(mu-mixture_mu)* (mu-mixture_mu)
    summation = sigma*sigma + delta_square
    return np.sum(pi*summation, 1)

def aleatoric_loss(y_true, pred_var): #here pred_var should be [prediction, variance], y_true is true numax
    y_pred = pred_var[:,0]
    log_var = pred_var[:, 1]
    loss = (K.abs(y_true - y_pred)/y_true)*(K.exp(-log_var)) + log_var
    return K.mean(loss, -1) # K.mean loss?

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def ps_to_array(freq, power, nbins=128, supersample=1,
                minfreq=3., maxfreq=283., minpow=3., maxpow=3e7):
    """
    Produce 2D array representation of power spectrum that is similar to Marc Hon's 2D images
    Written by Keaton Bell (bell@mps.mpg.de)
    This should be faster and more precise than writing plots to images
    Returns nbin x nbins image-like representation of the data
    freq and power are from power spectrum
    min/max freqs/powers define the array edges in same units as input spectrum
    if supersample == 1, result is strictly black and white (1s and 0s)
    if supersample > 1, returns grayscale image represented spectrum "image" density
    """
    # make sure integer inputs are integers
    nbins = int(nbins)
    supersample = int(supersample)
    # Set up array for output
    output = np.zeros((nbins, nbins))
    if supersample > 1:  # SUPERSAMPLE
        # Call yourself and flip orientation again
        supersampled =  ps_to_array(freq, power, nbins=nbins * supersample, supersample=1,
                                        minfreq=minfreq, maxfreq=maxfreq, minpow=minpow, maxpow=maxpow)[::-1]
        for i in range(supersample):
            for j in range(supersample):
                output += supersampled[i::supersample, j::supersample]
        output = output / (supersample ** 2.)
    else:  # don't supersample
        # Do everything in log space
        logfreq = np.log10(freq)
        logpower = np.log10(power)
        minlogfreq = np.log10(minfreq)
        maxlogfreq = np.log10(maxfreq)
        minlogpow = np.log10(minpow)
        maxlogpow = np.log10(maxpow)

        # Define bins

        xbinedges = np.linspace(np.log10(minfreq), np.log10(maxfreq), nbins + 1)
        xbinwidth = xbinedges[1] - xbinedges[0]
        ybinedges = np.linspace(np.log10(minpow), np.log10(maxpow), nbins + 1)
        ybinwidth = ybinedges[1] - ybinedges[0]

        # resample at/near edges of bins and at original frequencies

        smalloffset = xbinwidth / (10. * supersample)  # to get included in lower-freq bin
        interpps = interp1d(logfreq, logpower, fill_value=(0,0), bounds_error=False)
        poweratedges = interpps(xbinedges)
        logfreqsamples = np.concatenate((logfreq, xbinedges, xbinedges - smalloffset))
        powersamples = np.concatenate((logpower, poweratedges, poweratedges))

        sort = np.argsort(logfreqsamples)
        logfreqsamples = logfreqsamples[sort]
        powersamples = powersamples[sort]

        # Get maximum and minimum of power in each frequency bin
        maxpow = binned_statistic(logfreqsamples, powersamples, statistic='max', bins=xbinedges)[0]
        minpow = binned_statistic(logfreqsamples, powersamples, statistic='min', bins=xbinedges)[0]
        # Convert to indices of binned power

        # Fix to fall within power range
        minpowinds = np.floor((minpow - minlogpow) / ybinwidth)
        minpowinds = squeeze(minpowinds, 0, nbins).astype('int')
        maxpowinds = np.ceil((maxpow - minlogpow) / ybinwidth)
        maxpowinds = squeeze(maxpowinds, 0, nbins).astype('int')

        # populate output array
        for i in range(nbins):
            output[minpowinds[i]:maxpowinds[i], i] = 1.
            if maxpowinds[i] - minpowinds[i] != np.sum(output[minpowinds[i]:maxpowinds[i], i]):
                print(i, "!!!!!!")
                print(minpowinds[i])
                print(maxpowinds[i])
    # return result, flipped to match orientation of Marc's images
    return output[::-1]


def predict_from_psd_mdn():
    K.set_learning_phase(1)

    import time as timer

    psd_folder = '/home/z3384751/Desktop/Data/TESS/Sector1-2_TASOC/Internal_S02/S02/PSD_trial2/PSD_pt4/'
    output_file = '/home/z3384751/Desktop/Data/TESS/Sector1-2_TASOC/Internal_S02/S02/Classification_Results_OTF_FIXED_Mag13_50percent_MDN_numax_pt4.dat'
    model_check = load_model(
        '/home/z3384751/PycharmProjects/K2Detection/ClassifyModels/2D classifier/28d_RG_classification/Bell_Array_OnTheFly/FIXED_Classifier-1-0.9870-Testing-Max-50percent13Mag-Sampling.h5',
        custom_objects={'precision_m': precision_m, 'recall_m': recall_m})
    model_list = [model_check]

    model_reg = SLOSH_Regressor(num_gaussians=4)
    saved_model_dict = '/home/z3384751/PycharmProjects/K2Detection/ClassifyModels/2D classifier/28d_RG_numax/Bell_Array/OTF/V6-UpperMag13-OTF_FIXED_TrainMore-CRPS_4yrUncertainty-MDN_WITH_Drop-50percentAug_to_13_Softplus-MAPE:3.86-MAE:2.34'
    model_reg.load_state_dict(torch.load(saved_model_dict))
    model_reg.cuda()
    model_reg.eval()

    nb_files = 0
    nb_iter = 10
    count = 0

    for _ in os.listdir(psd_folder):
        nb_files += 1

    tic_vec = []

    if os.path.exists(output_file):
        existing_df = pd.read_table(output_file, delim_whitespace=True, header=0)
        existing_file = existing_df['File'].values
        existing_tic = existing_file
    else:
        existing_tic = []

    for filename in os.listdir(psd_folder):
        start_time = timer.time()
        tic_id = os.path.splitext(filename)[0]
        if tic_id in tic_vec:
            count += 1
            print('TIC already written!')
            continue
        if tic_id in existing_tic:
            count += 1
            print('TIC already written!')
            continue
        data = np.load(os.path.join(psd_folder, filename))
        freq = data['freq']
        pow = data['pow']
        try:
            nb_points = data['N']
            contamination = data['contamination']
        except:
            nb_points, contamination = -99, -99
        im = ps_to_array(freq, pow)
        im = im.reshape(1, 128, 128, 1)

        class_pred_vec = np.zeros((nb_iter, len(model_list)))

        for i in range(nb_iter):
            for j in range(len(model_list)):
                class_pred = model_list[j].predict(im)
                class_pred_vec[i, j] = class_pred[:, 1]
        with torch.no_grad():
            pi, sigma, mu = model_reg(torch.from_numpy(im.copy().squeeze(-1)).float().cuda())
            pred_numax = dist_mu(pi, mu).data.cpu().numpy().squeeze()
            numax_pred_var = dist_var_npy(pi=pi.data.cpu().numpy(), mu=mu.data.cpu().numpy(),
                                          mixture_mu=pred_numax,
                                          sigma=sigma.data.cpu().numpy()).squeeze()
            pred_numax_sigma = np.sqrt(numax_pred_var)

        class_mc_average = np.mean(class_pred_vec, axis=0)  # average over mc iterations
        class_mc_aleatoric = np.mean(class_pred_vec * (1 - class_pred_vec), axis=0)
        class_mc_kwon_std = np.sqrt(np.var(class_pred_vec, axis=0) + class_mc_aleatoric)
        class_mc_std = np.std(class_pred_vec, axis=0)
        class_model_average = np.round(np.mean(class_mc_average), 3)  # average over models in ensemble
        class_model_std = 0
        class_model_kwon_std = 0
        class_model_std_ltv = np.sqrt(
            np.var(class_mc_average) + np.mean(np.var(class_pred_vec, axis=0)))  # LTV : Law of Total Variance
        class_model_std_ltv = np.round(class_model_std_ltv, 3)
        class_model_kwon_std_ltv = np.sqrt(
            np.var(class_mc_average) + np.mean(np.var(class_pred_vec, axis=0) + class_mc_aleatoric))
        class_model_kwon_std_ltv = np.round(class_model_kwon_std_ltv, 3)

        for k in range(len(class_mc_std)):  # modify this part to add law of total variance (LTV)
            class_model_std += (class_mc_std[k]) ** 2
            class_model_kwon_std += (class_mc_kwon_std[k]) ** 2
        class_model_std = np.round(np.sqrt(class_model_std), 3)
        class_model_kwon_std = np.round(np.sqrt(class_model_kwon_std), 3)


        pred_numax = np.round(pred_numax, 3)
        pred_numax_sigma = np.round(pred_numax_sigma, 3)

        threshold = 0.4644  # 28d RG
        if class_model_average >= threshold:
            class_label = 1
        else:
            class_label = 0

        if os.path.exists(output_file):
            with open(output_file, 'a') as out_file:
                writer = csv.writer(out_file, delimiter=' ')
                writer.writerow(
                    [tic_id, class_model_average, class_model_std, class_model_kwon_std, class_model_std_ltv,
                     class_model_kwon_std_ltv, class_label, pred_numax, pred_numax_sigma, nb_points, contamination])
        else:
            with open(output_file, 'a') as out_file:
                writer = csv.writer(out_file, delimiter=' ')
                writer.writerow(
                    ['file', 'prob', 'prob_std', 'pred_std_kwon', 'pred_std_ltv', 'pred_std_kwon_ltv', 'det', 'numax',
                     'numax_std', 'N', 'contamination'])
                writer.writerow(
                    [tic_id, class_model_average, class_model_std, class_model_kwon_std, class_model_std_ltv,
                     class_model_kwon_std_ltv, class_label, pred_numax, pred_numax_sigma, nb_points, contamination])

        tic_vec.append(tic_id)
        count += 1
        end_time = timer.time()
        print('Execution Time: %.3f seconds' % (end_time - start_time))
        print('Progress: %d/%d' % (count, nb_files))
        print('Save folder: ', output_file)


def main():
    predict_from_psd_mdn()

if __name__ == '__main__':
    main()