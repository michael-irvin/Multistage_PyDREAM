import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import numpy as np
from pydream.convergence import Gelman_Rubin
from pydream.core import run_dream
from pydream.parameters import SampledParam
from scipy.stats import norm
from ms_pydream.objective import objective_function
from ms_pydream.utils import load_module

program_settings = json.load(open(sys.argv[1]))
# program_settings = json.load(open("/Users/michaelirvin/Research/Multistage_PyDREAM/examples/mixed_effect_model/calibrator_settings.json"))

model_file = program_settings["model_file"]
model_name = program_settings.get("model_name","example")
model_results_dir = program_settings.get("model_results_dir", "")
max_iterations = program_settings.get("max_iterations", 20)
n_iterations = program_settings.get("n_iterations",10)
burn_in_len = program_settings.get("burn_in_len",5)
n_chains = program_settings.get("n_chains",5)
ncr = program_settings.get("ncr", 25)
gamma_levels = program_settings.get("gamma_levels", 8)
p_gamma_unity = program_settings.get("p_gamma_unity", 0.1)
verbose = program_settings.get("verbose", True)

# Import p1 and p2, and sample_priors from user specified `model.py` file. 
# p1 is the faster coarse grain model. 
# p2 is the slower fine tuned model. 
model = load_module(model_file, "model")
sampled_priors = model.sampled_priors
p1 = model.p1
p2 = model.p2

@objective_function(p1x_ref=-1e20)
def p(x):
    # Run coarse-grain p1(x)  
    p1x = p1(x)

    # Acceptance
    mh_ratio = min(1, p1x/p.p1x_ref)
    accept = np.isfinite(mh_ratio) and np.log(np.random.uniform()) < mh_ratio
    if accept:
        # Run fine-tuned
        p.p1x_ref = p1(x)
        p2x = p2(x) 
        return p2x * min(1, p.p1x_ref/p1x)  # Adjust log ps to account for initial MH on the coarse-grain model. 
    else:
        # Return very small number (not -np.inf), so the DREAM algorithm, rejects the proposal. 
        # Not Nan because if you have successive -np.inf's the MH ratio gets stuck at 1.0 and you get trapped accepting -np.inf.  
        return -1e20 # Very small number
    

model_name = os.path.join(model_results_dir, model_name)
dream_settings = {
    "parameters": sampled_priors,
    "likelihood": p, 
    "niterations": n_iterations,
    "nchains": n_chains,
    "snooker": 0.0, # Set 0.0 probability of snooker update. Snooker updates break detailed balance in the multistage calibration.
    "multitry": False, # multitry also breaks detailed balance in the multistage calibratoin. 
    "nCR": ncr,
    "gamma_levels": gamma_levels,
    "adapt_gamma": True,
    "p_gamma_unity": p_gamma_unity,
    "history_thin": 1, 
    "model_name": model_name,
    "verbose": True, 
    }


# -------- Calibration -------
# Model Inference via PyDREAM
if __name__ == '__main__':
    # Run DREAM sampling.  Documentation of DREAM options is in Dream.py.
    converged = False
    total_iterations = n_iterations
    sampled_params, log_ps = run_dream(
        crossover_burnin=min(n_iterations, burn_in_len),
        **dream_settings
        )

    # Save sampling output (sampled parameter values and their corresponding logps).
    for chain in range(len(sampled_params)):
        np.save(model_name + '_' + str(chain) + '_' + str(total_iterations) + '_' + 'parameters', sampled_params[chain])
        np.save(model_name + '_' + str(chain) + '_' + str(total_iterations) + '_' + 'log_p', log_ps[chain])

    GR = Gelman_Rubin(sampled_params)
    burn_in_len = max(burn_in_len - n_iterations, 0)
    print('At iteration: ', total_iterations, ' GR = ', GR)
    print(f'At iteration: {total_iterations}, {burn_in_len} steps of burn-in remain.')

    np.savetxt(model_name + str(total_iterations) + '.txt', GR)

    if np.all(GR < 1.2) or np.any(GR > 1.2):
        converged = True

    # append sample with a re-run of the pyDream algorithm
    while not converged or (total_iterations < max_iterations):
        old_samples = sampled_params
        starts = [sampled_params[chain][-1, :] for chain in range(n_chains)]

        total_iterations += n_iterations
        sampled_params, log_ps = run_dream(
            start=starts,
            crossover_burnin=min(n_iterations, burn_in_len), 
            **dream_settings
            )

        # Save sampling output (sampled parameter values and their corresponding logps).
        for chain in range(len(sampled_params)):
            np.save(model_name + '_' + str(chain) + '_' + str(total_iterations) + '_' + 'parameters',
                    sampled_params[chain])
            np.save(model_name + '_' + str(chain) + '_' + str(total_iterations) + '_' + 'log_p', log_ps[chain])

        old_samples = [np.concatenate((old_samples[chain], sampled_params[chain])) for chain in range(n_chains)]
        GR = Gelman_Rubin(old_samples)
        burn_in_len = max(burn_in_len - n_iterations, 0)
        print('At iteration: ', total_iterations, ' GR = ', GR)
        print(f'At iteration: {total_iterations}, {burn_in_len} steps of burn-in remain.')

        np.savetxt(model_name + str(total_iterations) + '.txt', GR)

        if np.all(GR < 1.2):
            converged = True
