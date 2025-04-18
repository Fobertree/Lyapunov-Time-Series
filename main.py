import fast_rqa
from plot import plot_data
from data import get_tkr_data
import json
from datetime import datetime
import os
import time

if __name__ == "__main__":
    tkr = "NVDA"

    raw_X = get_tkr_data(tkr)
    X = raw_X.values.flatten()

    print(X.shape)
    
    dat, impute_mask, best_params, best_score = fast_rqa.optimize(X)

    date = datetime.fromtimestamp(time.time())

    #save data as backup so we can manipulate it in Jupyter notebook if unsatisfied + logging purposes
    # we could use json_numpy to make numpy JSON serializable but I will keep it simple
    
    try:
        out_dict = {
            "data": dat.tolist(),
            "impute_mask" : impute_mask.tolist(),
            "best_params": best_params,
            "best_score": best_score
        }

        with open(os.path.join("JSON", f"{tkr}.json"), "w") as fd:
            json.dump(out_dict, fd)
    except Exception as e:
        print(f"ERROR IN JSON PREP/DUMP: {e}")

    plot_data(dat, "RQA", "RQA Lyapunov Exponents", "teal", f"{tkr}-rqa.png", impute_mask, X)