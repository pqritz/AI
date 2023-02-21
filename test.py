import numpy as np

def test():
    w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))
    w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))
    b_i_h = np.zeros((20, 1))
    b_h_o = np.zeros((10, 1))

    h_pre = b_i_h + w_i_h @ img
    h = 1 / (1 + np.exp(-h_pre))

    o_pre = b_h_o + w_h_o @ h
    o = 1 / (1 + np.exp(-o_pre))

    print(test)

test()