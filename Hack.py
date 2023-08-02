import torch
import torch.nn as nn


if __name__ == '__main__':
    t1 = torch.ones([1024,1024]).to("cuda")
    t2 = torch.ones([1024,1024]).to("cuda")

    while True:
        t3 = t1 +t2