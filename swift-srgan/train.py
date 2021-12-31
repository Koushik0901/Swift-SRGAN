import os
import math
import pandas as pd
import torch
import torchvision
from data import TrainDataset, ValDataset, display_transform
from torch.utils.data import DataLoader
from models import Generator, Discriminator
from loss import GeneratorLoss
from metric import ssim
from tqdm import tqdm
import argparse


torch.backends.cudnn.benchmark = True
torch.cuda.manual_seed_all(42)


def main(opt):

    os.makedirs("./results", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = TrainDataset(
        "./dataset/train", crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR
    )
    val_set = ValDataset(
        "./dataset/valid", crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR
    )

    train_loader = DataLoader(
        dataset=train_set,
        num_workers=os.cpu_count(),
        batch_size=opt.batch_size,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(dataset=val_set, num_workers=1, batch_size=1, shuffle=False)

    netG = Generator(upscale_factor=UPSCALE_FACTOR).to(DEVICE)
    print("# generator parameters:", sum(param.numel() for param in netG.parameters()))

    netD = Discriminator().to(DEVICE)
    print(
        "# discriminator parameters:", sum(param.numel() for param in netD.parameters())
    )

    generator_criterion = GeneratorLoss().to(DEVICE)

    optimizerG = torch.optim.AdamW(netG.parameters(), lr=1e-3)
    optimizerD = torch.optim.AdamW(netD.parameters(), lr=1e-3)

    results = {
        "d_loss": [],
        "g_loss": [],
        "d_score": [],
        "g_score": [],
        "psnr": [],
        "ssim": [],
    }

    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader, total=len(train_loader))
        running_results = {
            "batch_sizes": 0,
            "d_loss": 0,
            "g_loss": 0,
            "d_score": 0,
            "g_score": 0,
        }

        netG.train()
        netD.train()
        for lr_img, hr_img in train_bar:
            batch_size = lr_img.size(0)
            running_results["batch_sizes"] += batch_size

            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            hr_img = hr_img.to(DEVICE)
            lr_img = lr_img.to(DEVICE)

            sr_img = netG(lr_img)

            netD.zero_grad()
            real_out = netD(hr_img).mean()
            fake_out = netD(sr_img).mean()
            d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph=True)
            optimizerD.step()

            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            netG.zero_grad()

            sr_img = netG(lr_img)
            fake_out = netD(sr_img).mean()

            g_loss = generator_criterion(fake_out, sr_img, hr_img)
            g_loss.backward()

            optimizerG.step()

            # loss for current after before optimization
            running_results["g_loss"] += g_loss.item() * batch_size
            running_results["d_loss"] += d_loss.item() * batch_size
            running_results["d_score"] += real_out.item() * batch_size
            running_results["g_score"] += fake_out.item() * batch_size

            train_bar.set_description(
                desc="[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f"
                % (
                    epoch,
                    NUM_EPOCHS,
                    running_results["d_loss"] / running_results["batch_sizes"],
                    running_results["g_loss"] / running_results["batch_sizes"],
                    running_results["d_score"] / running_results["batch_sizes"],
                    running_results["g_score"] / running_results["batch_sizes"],
                )
            )

        netG.eval()

        with torch.no_grad():
            val_bar = tqdm(val_loader, total=len(val_loader))
            valing_results = {
                "mse": 0,
                "ssims": 0,
                "psnr": 0,
                "ssim": 0,
                "batch_sizes": 0,
            }
            val_images = []
            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results["batch_sizes"] += batch_size
                lr = val_lr
                hr = val_hr
                if torch.cuda.is_available():
                    lr = lr.cuda()
                    hr = hr.cuda()
                # Forward
                sr = netG(lr)
                # Loss & metrics
                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results["mse"] += batch_mse * batch_size
                batch_ssim = ssim(sr, hr).item()

                valing_results["ssims"] += batch_ssim * batch_size
                valing_results["psnr"] = 10 * math.log10(
                    (hr.max() ** 2)
                    / (valing_results["mse"] / valing_results["batch_sizes"])
                )
                valing_results["ssim"] = (
                    valing_results["ssims"] / valing_results["batch_sizes"]
                )
                val_bar.set_description(
                    desc="[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f"
                    % (valing_results["psnr"], valing_results["ssim"])
                )

                val_images.extend(
                    [
                        display_transform(val_hr_restore.squeeze(0)),
                        display_transform(hr.data.cpu().squeeze(0)),
                        display_transform(sr.data.cpu().squeeze(0)),
                    ]
                )
            val_images = torch.stack(val_images)
            val_images = torch.chunk(val_images, val_images.size(0) // 15)
            val_save_bar = tqdm(val_images, desc="[saving training results]")
            index = 1
            for image in val_save_bar:
                image = torchvision.utils.make_grid(image, nrow=3, padding=5)
                torchvision.utils.save_image(
                    image,
                    out_path + "epoch_%d_index_%d.png" % (epoch, index),
                    padding=5,
                )
                index += 1

        # save model parameters
        netG.train()
        netD.train()
        torch.save(
            {"model": netG.state_dict()},
            f"./checkpoints/netG_{UPSCALE_FACTOR}x_epoch{epoch}.pth.tar",
        )
        torch.save(
            {"model": netD.state_dict()},
            f"./checkpoints/netD_{UPSCALE_FACTOR}x_epoch{epoch}.pth.tar",
        )

        results["d_loss"].append(
            running_results["d_loss"] / running_results["batch_sizes"]
        )
        results["g_loss"].append(
            running_results["g_loss"] / running_results["batch_sizes"]
        )
        results["d_score"].append(
            running_results["d_score"] / running_results["batch_sizes"]
        )
        results["g_score"].append(
            running_results["g_score"] / running_results["batch_sizes"]
        )
        results["psnr"].append(valing_results["psnr"])
        results["ssim"].append(valing_results["ssim"])

        if epoch % 10 == 0 and epoch != 0:
            out_path = "logs/"
            data_frame = pd.DataFrame(
                data={
                    "Loss_D": results["d_loss"],
                    "Loss_G": results["g_loss"],
                    "Score_D": results["d_score"],
                    "Score_G": results["g_score"],
                    "PSNR": results["psnr"],
                    "SSIM": results["ssim"],
                },
                index=range(1, epoch + 1),
            )
            data_frame.to_csv(
                out_path + "ssrgan_" + str(UPSCALE_FACTOR) + "_train_results.csv",
                index_label="Epoch",
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Swift-SRGAN')
    parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8], help='super resolution upscale factor')
    parser.add_argument('--crop_size', default=96, type=int, help='training images crop size')
    parser.add_argument('--num_epochs', default=100, type=int, help='number of epochs to train')
    opt = parser.parse_args()
    main(opt)