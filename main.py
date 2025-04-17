import fast_rqa
from plot import plot_data
from data import get_tkr_data

if __name__ == "__main__":
    raw_X = get_tkr_data("DJT")
    X = raw_X.values.flatten()

    print(X.shape, X)
    input("BOB")
    dat, impute_mask = fast_rqa.rqa(X, True)

    plot_data(dat, "RQA", "RQA Lyapunov Exponents", "teal", "rqa.png", impute_mask, X)