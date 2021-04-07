show_warning_trace = False
if show_warning_trace:
    import traceback
    import warnings
    import sys

    def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

        log = file if hasattr(file,'write') else sys.stderr
        traceback.print_stack(file=log)
        log.write(warnings.formatwarning(message, category, filename, lineno, line))

    warnings.showwarning = warn_with_traceback

#from experiments.hrl.long_trial_hierarchical import run
#from experiments.hrl.gridsearch_fourrooms import run
#from experiments.hrl.long_trial_fourrooms import run
#from experiments.hrl.long_trial_oneroom import run
#from experiments.hrl.long_trial_hierarchical_tworooms import run
#from experiments.hrl.fully_differentiable_tworooms import run
#from experiments.hrl.transfer_nrooms import run
from experiments.hrl.transfer_nrooms_v2 import run

#while True:
#    q = run()
run()
