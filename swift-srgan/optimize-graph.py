""" Utility script to convert to torchscript model"""

import torch
from models import Generator
import argparse


def trace(model):
    x = torch.rand(1, 3, 96, 96)
    traced_model = torch.jit.trace(model, (x))
    return traced_model

def main(opt):

    model = Generator(in_channels=3).to(opt.device)
    ckpt = torch.load(opt.ckpt_path, map_location=opt.device)
    model.load_state_dict(ckpt['model'])
    print("Tracing...")
    traced = trace(model)
    torch.jit.save(traced, opt.save_path)
    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Swift-SRGAN')
    parser.add_argument('--ckpt_path', required=True, type=str, help='path to saved checkpoint')
    parser.add_argument('--save_path', required=True, type=str, help='path to save the jit optimized model')
    parser.add_argument('--device', required=False, default='cuda', type=str, help='device to map the model')
    opt = parser.parse_args()
    main(opt)