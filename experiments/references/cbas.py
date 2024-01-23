import numpy as np


class CBasReferenceModel:
    def __init__(self, vae: object) -> None:
        self.vae = vae
        self.test_kwargs = [
                   {'weights_type': 'cbas', 'quantile': 1},
        ]






def get_balaji_predictions(preds, Xt):
    """
    Author: SB
    Given a set of predictors built according to the methods in 
    the Balaji Lakshminarayanan paper 'Simple and scalable predictive 
    uncertainty estimation using deep ensembles' (2017), returns the mean and
    variance of the total prediction.
    """
    M = len(preds)
    N = Xt.shape[0]
    means = np.zeros((M, N))
    variances = np.zeros((M, N))
    for m in range(M):
        y_pred = preds[m].predict(Xt)
        means[m, :] = y_pred[:, 0]
        variances[m, :] = np.log(1+np.exp(y_pred[:, 1])) + 1e-6
    mu_star = np.mean(means, axis=0)
    var_star = (1/M) * (np.sum(variances, axis=0) + np.sum(means**2, axis=0)) - mu_star**2
    return mu_star, var_star


def run_experimental_weighted_ml(it, ground_truth, X_train, y_train, repeats=3, parallel_function_evaluations=100, black_box_evaluations=50):
    """
    Author: SB
    Runs the GFP comparative tests on the weighted ML models and FBVAE.
    """
    
    TRAIN_SIZE = 5000
    train_size_str = "%ik" % (TRAIN_SIZE/1000)
    num_models = [1, 5, 20][it]
    RANDOM_STATE = it + 1

    #X_train, y_train, gt_train  = util.get_experimental_X_y(random_state=RANDOM_STATE, train_size=TRAIN_SIZE)
    gt_train = y_train

    vae_suffix = '_%s_%i' % (train_size_str, RANDOM_STATE)
    oracle_suffix = '_%s_%i_%i' % (train_size_str, num_models, RANDOM_STATE)
    
    vae_0 = util.build_vae(latent_dim=20,
                           n_tokens=20,
                           seq_length=X_train.shape[1],
                           enc1_units=50)

    vae_0.encoder_.load_weights("../models/vae_0_encoder_weights%s.h5" % vae_suffix)
    vae_0.decoder_.load_weights("../models/vae_0_decoder_weights%s.h5"% vae_suffix)
    vae_0.vae_.load_weights("../models/vae_0_vae_weights%s.h5"% vae_suffix)
    
    loss = losses.neg_log_likelihood
    keras.utils.get_custom_objects().update({"neg_log_likelihood": loss})
    oracles = [keras.models.load_model("../models/oracle_%i%s.h5" % (i, oracle_suffix)) for i in range(num_models)]
    
    test_kwargs = [
                   {'weights_type': 'cbas', 'quantile': 1},
    ]
    
    base_kwargs = {
        'homoscedastic': False,
        'homo_y_var': 0.01,
        'train_gt_evals':gt_train,
        'samples': parallel_function_evaluations,
        'cutoff':1e-6,
        'it_epochs':10,
        'verbose':True,
        'LD': 20,
        'enc1_units':50,
        'iters': black_box_evaluations,
    }
    
    if num_models==1:
        base_kwargs['homoscedastic'] = True
        base_kwargs['homo_y_var'] = np.mean((util.get_balaji_predictions(oracles, X_train)[0] - y_train)**2)
    
    for k in range(repeats):
        for j in range(len(test_kwargs)):
            test_name = test_kwargs[j]['weights_type']
            suffix = "_%s_%i_%i" % (train_size_str, RANDOM_STATE, k)
            if test_name == 'fbvae':
                if base_kwargs['iters'] > 100:
                    suffix += '_long'
            
                print(suffix)
                kwargs = {}
                kwargs.update(test_kwargs[j])
                kwargs.update(base_kwargs)
                [kwargs.pop(k) for k in ['homoscedastic', 'homo_y_var', 'cutoff', 'it_epochs']]
                test_traj, test_oracle_samples, test_gt_samples, test_max = optimization_algs.fb_opt(np.copy(X_train), oracles, ground_truth, vae_0, **kwargs)
            else:
                if base_kwargs['iters'] > 100:
                    suffix += '_long'
                kwargs = {}
                kwargs.update(test_kwargs[j])
                kwargs.update(base_kwargs)
                test_traj, test_oracle_samples, test_gt_samples, test_max = optimization_algs.weighted_ml_opt(np.copy(X_train), oracles, ground_truth, vae_0, **kwargs)
            np.save('../results/%s_traj%s.npy' %(test_name, suffix), test_traj)
            np.save('../results/%s_oracle_samples%s.npy' % (test_name, suffix), test_oracle_samples)
            np.save('../results/%s_gt_samples%s.npy'%(test_name, suffix), test_gt_samples )

            with open('../results/%s_max%s.json'% (test_name, suffix), 'w') as outfile:
                json.dump(test_max, outfile)
