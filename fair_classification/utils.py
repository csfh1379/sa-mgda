import torch
import torch.nn.functional as F
from torch.autograd import Variable as V


def simplex_projection(t, n_cate):
    u, _ = torch.sort(t, dim=0, descending=True)

    # assume t is sort from large to small t1 >= t2
    u = V(u.cpu(), requires_grad=False)

    num_pos_values = 0  # number of positive values in the projected solution (will compute)
    for l in range(1, n_cate + 1):
        check_value = u[l - 1, 0] + (1 - sum(u[0:l, 0])) / l
        if check_value <= 0:
            num_pos_values = l - 1
            break

    if num_pos_values == 0:
        num_pos_values = n_cate

    lambda_ = (1 - sum(u[0:num_pos_values, 0])) / num_pos_values
    return F.relu(lambda_ + t)
