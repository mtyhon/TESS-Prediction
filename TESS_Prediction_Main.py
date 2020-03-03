import numpy as np
import keras.backend as K
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))
import pandas as pd
import os, keras, csv, torch, argparse
import time as timer
from keras.models import load_model
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--psd_folder", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--threshold", type=float, default=0.5) # threshold above which we say it's oscillating
parser.add_argument("--nb_mc_iter", type=int, default=10) # number of MC Dropout iterations for uncertainties
config = parser.parse_args()

def predict_from_psd_mdn(config):
    package_dir = os.path.dirname(os.path.abspath(__file__))
    K.set_learning_phase(1)

    psd_folder = config.psd_folder
    output_file = config.output_file
    model_check = load_model(package_dir +'/models/classifier/Keras_Classifier_Aug13.h5',
        custom_objects={'precision_m': precision_m, 'recall_m': recall_m})
    model_list = [model_check]

    model_reg = SLOSH_Regressor(num_gaussians=4)
    saved_model_dict = package_dir + '/models/regressor/MDN_CRPS_Softplus_Aug13.torchmodel'
    model_reg.load_state_dict(torch.load(saved_model_dict))
    model_reg.cuda()
    model_reg.eval()


    nb_files = 0
    nb_iter = config.nb_mc_iter
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

        threshold = config.threshold
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

def main(config):
    predict_from_psd_mdn(config)
    print('Done!')

if __name__ == '__main__':
    main(config)