# External imports
import torch
import torch.nn as nn
from itertools import cycle
from tqdm import tqdm
from datetime import datetime
import os
import tempfile
from pathlib import Path
import ray.cloudpickle as pickle
from ray.train import Checkpoint
from ray import train

# Internal imports
from src.utils import save_model, get_device


OUTPUT_CHANNELS = 3
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def downsample(in_channels, filters, size, apply_instancenorm=True):
    layers = []

    conv2d = nn.Conv2d(
        in_channels=in_channels,
        out_channels=filters,
        kernel_size=size,
        stride=2,
        padding=1,
        bias=False,
    )
    nn.init.normal_(conv2d.weight, mean=0.0, std=0.02)
    layers.append(conv2d)

    if apply_instancenorm:
        instance_norm = nn.InstanceNorm2d(num_features=filters)
        layers.append(instance_norm)

    layers.append(nn.LeakyReLU())

    return nn.Sequential(*layers)


def upsample(in_channels, filters, size, apply_dropout=False):
    layers = []

    conv2d = nn.ConvTranspose2d(
        in_channels=in_channels,
        out_channels=filters,
        kernel_size=size,
        stride=2,
        padding=1,
        bias=False,
    )
    nn.init.normal_(conv2d.weight, mean=0.0, std=0.02)
    layers.append(conv2d)

    instance_norm = nn.InstanceNorm2d(num_features=filters)
    layers.append(instance_norm)

    if apply_dropout:
        layers.append(nn.Dropout(p=0.5))

    layers.append(nn.LeakyReLU())

    return nn.Sequential(*layers)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.down_stack = nn.ModuleList(
            [
                downsample(in_channels=3, filters=64, size=4, apply_instancenorm=False),
                downsample(in_channels=64, filters=128, size=4),
                downsample(in_channels=128, filters=256, size=4),
                downsample(in_channels=256, filters=512, size=4),
                downsample(in_channels=512, filters=512, size=4),
                downsample(in_channels=512, filters=512, size=4),
                downsample(in_channels=512, filters=512, size=4),
                downsample(
                    in_channels=512, filters=512, size=4, apply_instancenorm=False
                ),  # (bs, 1, 1, 512)
            ]
        )

        # adding dimensions to the input channels because of the skip connections
        self.up_stack = nn.ModuleList(
            [
                upsample(in_channels=512, filters=1024, size=4, apply_dropout=True),
                upsample(
                    in_channels=1024 + 512, filters=1024, size=4, apply_dropout=True
                ),
                upsample(
                    in_channels=1024 + 512, filters=1024, size=4, apply_dropout=True
                ),
                upsample(in_channels=1024 + 512, filters=1024, size=4),
                upsample(in_channels=1024 + 512, filters=512, size=4),
                upsample(in_channels=512 + 256, filters=256, size=4),
                upsample(
                    in_channels=256 + 128, filters=128, size=4
                ),  # (bs, 128, 128, 128)
            ]
        )

        conv2d = nn.ConvTranspose2d(
            in_channels=128 + 64,
            out_channels=OUTPUT_CHANNELS,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        nn.init.normal_(conv2d.weight, mean=0.0, std=0.02)
        self.last = nn.Sequential(conv2d, nn.Tanh())

    def forward(self, x):
        # Downsampling the model
        skips = []
        for down in self.down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            x = torch.cat((x, skip), dim=1)
            # x = torch.cat((x, skip))

        x = self.last(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.down_stack = nn.ModuleList(
            [
                downsample(in_channels=3, filters=64, size=4, apply_instancenorm=False),
                downsample(in_channels=64, filters=128, size=4),
                downsample(in_channels=128, filters=256, size=4),
            ]
        )
        self.zeropad = nn.ZeroPad2d(padding=1)
        self.conv = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=4, bias=False, stride=1
        )
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.02)

        self.instance_norm = nn.InstanceNorm2d(num_features=512)
        self.leaky_relu = nn.LeakyReLU()
        self.last = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1)
        nn.init.normal_(self.last.weight, mean=0.0, std=0.02)

    def forward(self, x):
        for down in self.down_stack:
            x = down(x)

        x = self.zeropad(x)
        x = self.conv(x)
        x = self.instance_norm(x)
        x = self.leaky_relu(x)
        x = self.last(x)

        return x


class CycleGAN(nn.Module):
    def __init__(
        self,
        monet_generator,
        photo_generator,
        monet_discriminator,
        photo_discriminator,
        device,
        lambda_cycle=10,
    ):
        super(CycleGAN, self).__init__()
        self.m_gen = monet_generator
        self.p_gen = photo_generator
        self.m_disc = monet_discriminator
        self.p_disc = photo_discriminator
        self.device = device
        self.lambda_cycle = lambda_cycle

        # define optimizers
        self.m_gen_optimizer = torch.optim.Adam(
            self.m_gen.parameters(), lr=2e-4, betas=(0.5, 0.999)
        )
        self.p_gen_optimizer = torch.optim.Adam(
            self.p_gen.parameters(), lr=2e-4, betas=(0.5, 0.999)
        )
        self.m_disc_optimizer = torch.optim.Adam(
            self.m_disc.parameters(), lr=2e-4, betas=(0.5, 0.999)
        )
        self.p_disc_optimizer = torch.optim.Adam(
            self.p_disc.parameters(), lr=2e-4, betas=(0.5, 0.999)
        )

    def discriminator_loss(self, real, generated):
        real_loss = nn.BCEWithLogitsLoss()(
            real,
            torch.ones_like(real),
        )
        generated_loss = nn.BCEWithLogitsLoss()(generated, torch.zeros_like(generated))

        total_disc_loss = real_loss + generated_loss
        return total_disc_loss * 0.5

    def generator_loss(self, generated):
        return nn.BCEWithLogitsLoss()(generated, torch.ones_like(generated))

    def calc_cycle_loss(self, real_image, cycled_image, LAMBDA):
        loss = torch.mean(torch.abs(real_image - cycled_image))
        return LAMBDA * loss

    def identity_loss(self, real_image, same_image, LAMBDA):
        loss = torch.mean(torch.abs(real_image - same_image))
        return LAMBDA * 0.5 * loss

    def train_step(self, monet_ds, photo_ds):
        for _, (real_monet, real_photo) in enumerate(
            zip(
                tqdm(monet_ds, ncols=100, desc="Training for this epoch"),
                photo_ds,
            )
        ):
            real_monet = real_monet.to(self.device)
            real_photo = real_photo.to(self.device)

            # zero the gradients for batch
            self.m_gen_optimizer.zero_grad()
            self.p_gen_optimizer.zero_grad()
            self.m_disc_optimizer.zero_grad()
            self.p_disc_optimizer.zero_grad()

            # photo to monet back to photo
            fake_monet = self.m_gen(real_photo)
            cycled_photo = self.p_gen(fake_monet)

            # monet to photo back to monet
            fake_photo = self.p_gen(real_monet)
            cycled_monet = self.m_gen(fake_photo)

            # generating itself
            same_monet = self.m_gen(real_monet)
            same_photo = self.p_gen(real_photo)

            # discriminator used to check, inputing real images
            disc_real_monet = self.m_disc(real_monet)
            disc_real_photo = self.p_disc(real_photo)

            # discriminator used to check, inputing fake images
            disc_fake_monet = self.m_disc(fake_monet)
            disc_fake_photo = self.p_disc(fake_photo)

            # evaluate generator loss
            monet_gen_loss = self.generator_loss(disc_fake_monet)
            photo_gen_loss = self.generator_loss(disc_fake_photo)

            # evaluate total cycle consistency loss
            monet_cycle_loss = self.calc_cycle_loss(
                real_monet, cycled_monet, self.lambda_cycle
            )
            photo_cycle_loss = self.calc_cycle_loss(
                real_photo, cycled_photo, self.lambda_cycle
            )
            total_cycle_loss = monet_cycle_loss + photo_cycle_loss

            # evaluate total generator loss
            total_monet_gen_loss = (
                monet_gen_loss
                + total_cycle_loss
                + self.identity_loss(real_monet, same_monet, self.lambda_cycle)
            )
            total_monet_gen_loss.backward(retain_graph=True)

            total_photo_gen_loss = (
                photo_gen_loss
                + total_cycle_loss
                + self.identity_loss(real_photo, same_photo, self.lambda_cycle)
            )
            total_photo_gen_loss.backward(retain_graph=True)

            # evaluate total discriminator loss
            monet_disc_loss = self.discriminator_loss(disc_real_monet, disc_fake_monet)
            monet_disc_loss.backward(retain_graph=True)

            photo_disc_loss = self.discriminator_loss(disc_real_photo, disc_fake_photo)
            photo_disc_loss.backward()

            # Adjust the learning weights
            self.m_gen_optimizer.step()
            self.p_gen_optimizer.step()
            self.m_disc_optimizer.step()
            self.p_disc_optimizer.step()

        return {
            "monet_gen_loss": total_monet_gen_loss.item(),
            "photo_gen_loss": total_photo_gen_loss.item(),
            "monet_disc_loss": monet_disc_loss.item(),
            "photo_disc_loss": photo_disc_loss.item(),
        }

    def train(
        self,
        monet_ds,
        photo_ds,
        start_epoch=0,
        epochs=25,
        save_models=False,
        tune=False,
    ):
        best_loss = float("inf")
        if save_models:
            dt = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_folder = os.path.join(os.getcwd(), "checkpoints", dt)
            if not os.path.exists(out_folder):
                os.mkdir(out_folder)

        for i in range(start_epoch, epochs):
            print("Epoch {:d}".format(i + 1))

            loss = self.train_step(monet_ds, photo_ds)

            print(
                "photo_gen_loss: {:.4f}; monet_gen_loss: {:.4f}; photo_disc_loss: {:.4f}; monet_disc_loss: {:.4f}\n".format(
                    loss["photo_gen_loss"],
                    loss["monet_gen_loss"],
                    loss["photo_disc_loss"],
                    loss["monet_disc_loss"],
                )
            )

            if save_models and loss["monet_gen_loss"] < best_loss:
                best_loss = loss["monet_gen_loss"]
                model_path = os.path.join(out_folder, "best_model.pth")
                save_model(
                    model_path,
                    i,
                    self.m_gen,
                    self.m_gen_optimizer,
                    loss["monet_gen_loss"],
                )

            if tune:
                checkpoint_data = {
                    "epoch": i,
                    "model_state_dict": self.state_dict(),
                    "m_gen_optimizer": self.m_gen_optimizer.state_dict(),
                    "p_gen_optimizer": self.p_gen_optimizer.state_dict(),
                    "m_disc_optimizer": self.m_disc_optimizer.state_dict(),
                    "p_disc_optimizer": self.p_disc_optimizer.state_dict(),
                }

                with tempfile.TemporaryDirectory() as checkpoint_dir:
                    data_path = os.path.join(Path(checkpoint_dir), "data.pkl")
                    with open(data_path, "wb") as fp:
                        pickle.dump(checkpoint_data, fp)

                    checkpoint = Checkpoint.from_directory(checkpoint_dir)
                    train.report(
                        {
                            "photo_gen_loss": loss["photo_gen_loss"],
                            "monet_gen_loss": loss["monet_gen_loss"],
                            "photo_disc_loss": loss["photo_disc_loss"],
                            "monet_disc_loss": loss["monet_disc_loss"],
                        },
                        checkpoint=checkpoint,
                    )

        # Save last epoch checkpoint
        if save_models:
            model_path = os.path.join(out_folder, "final_model.pth")
            save_model(
                model_path,
                epochs,
                self.m_gen,
                self.m_gen_optimizer,
                loss["monet_gen_loss"],
            )

        print("Finished Training")


def get_cycle_gan_model() -> CycleGAN:
    device = get_device()

    # Define model
    monet_generator = Generator()
    photo_generator = Generator()
    monet_discriminator = Discriminator()
    photo_discriminator = Discriminator()

    model = CycleGAN(
        monet_generator=monet_generator,
        monet_discriminator=monet_discriminator,
        photo_generator=photo_generator,
        photo_discriminator=photo_discriminator,
        device=device,
    )
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    model.to(device)
    return model
