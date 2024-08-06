from functools import partial
import os
import tempfile
from pathlib import Path
import torch
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
from typing import Dict, Optional, AnyStr
import sys
from tqdm import tqdm

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

# Internal imports
from src.dataloader import load_data
from src.cycle_gan import get_cycle_gan_model, get_device


def tune_cycle_gan(config: Dict, data_dir: AnyStr, epochs: Optional[int] = 25):
    model = get_cycle_gan_model()

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = os.path.join(Path(checkpoint_dir), "data.pkl")
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            model.load_state_dict(checkpoint_state["model_state_dict"])
            model.m_gen_optimizer.load_state_dict(checkpoint_state["m_gen_optimizer"])
            model.p_gen_optimizer.load_state_dict(checkpoint_state["p_gen_optimizer"])
            model.m_disc_optimizer.load_state_dict(checkpoint_state["m_disc_optimizer"])
            model.p_disc_optimizer.load_state_dict(checkpoint_state["m_disc_optimizer"])
    else:
        start_epoch = 0

    monet_ds, photo_ds = load_data(data_dir=data_dir, batch_size=config["batch_size"])

    # Loop over all epochs
    for i in range(start_epoch, epochs):
        print("Epoch {:d}".format(i + 1))

        # Loop over all samples
        for real_monet, real_photo in zip(monet_ds, photo_ds):
            real_monet = real_monet.to(model.device)
            real_photo = real_photo.to(model.device)

            # zero the gradients for batch
            model.m_gen_optimizer.zero_grad()
            model.p_gen_optimizer.zero_grad()
            model.m_disc_optimizer.zero_grad()
            model.p_disc_optimizer.zero_grad()

            # photo to monet back to photo
            fake_monet = model.m_gen(real_photo)
            cycled_photo = model.p_gen(fake_monet)

            # monet to photo back to monet
            fake_photo = model.p_gen(real_monet)
            cycled_monet = model.m_gen(fake_photo)

            # generating itmodel
            same_monet = model.m_gen(real_monet)
            same_photo = model.p_gen(real_photo)

            # discriminator used to check, inputing real images
            disc_real_monet = model.m_disc(real_monet)
            disc_real_photo = model.p_disc(real_photo)

            # discriminator used to check, inputing fake images
            disc_fake_monet = model.m_disc(fake_monet)
            disc_fake_photo = model.p_disc(fake_photo)

            # evaluate generator loss
            monet_gen_loss = model.generator_loss(disc_fake_monet)
            photo_gen_loss = model.generator_loss(disc_fake_photo)

            # evaluate total cycle consistency loss
            monet_cycle_loss = model.calc_cycle_loss(
                real_monet, cycled_monet, model.lambda_cycle
            )
            photo_cycle_loss = model.calc_cycle_loss(
                real_photo, cycled_photo, model.lambda_cycle
            )
            total_cycle_loss = monet_cycle_loss + photo_cycle_loss

            # evaluate total generator loss
            total_monet_gen_loss = (
                monet_gen_loss
                + total_cycle_loss
                + model.identity_loss(real_monet, same_monet, model.lambda_cycle)
            )
            total_monet_gen_loss.backward(retain_graph=True)

            total_photo_gen_loss = (
                photo_gen_loss
                + total_cycle_loss
                + model.identity_loss(real_photo, same_photo, model.lambda_cycle)
            )
            total_photo_gen_loss.backward(retain_graph=True)

            # evaluate total discriminator loss
            monet_disc_loss = model.discriminator_loss(disc_real_monet, disc_fake_monet)
            monet_disc_loss.backward(retain_graph=True)

            photo_disc_loss = model.discriminator_loss(disc_real_photo, disc_fake_photo)
            photo_disc_loss.backward()

            # Adjust the learning weights
            model.m_gen_optimizer.step()
            model.p_gen_optimizer.step()
            model.m_disc_optimizer.step()
            model.p_disc_optimizer.step()

        print(
            "photo_gen_loss: {:.4f}; monet_gen_loss: {:.4f}; photo_disc_loss: {:.4f}; monet_disc_loss: {:.4f}\n".format(
                total_photo_gen_loss.item(),
                total_monet_gen_loss.item(),
                photo_disc_loss.item(),
                monet_disc_loss.item(),
            )
        )

        checkpoint_data = {
            "epoch": i,
            "model_state_dict": model.state_dict(),
            "m_gen_optimizer": model.m_gen_optimizer.state_dict(),
            "p_gen_optimizer": model.p_gen_optimizer.state_dict(),
            "m_disc_optimizer": model.m_disc_optimizer.state_dict(),
            "p_disc_optimizer": model.p_disc_optimizer.state_dict(),
        }

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = os.path.join(Path(checkpoint_dir), "data.pkl")
            with open(data_path, "wb") as fp:
                pickle.dump(checkpoint_data, fp)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            train.report(
                {
                    "photo_gen_loss": total_photo_gen_loss.item(),
                    "monet_gen_loss": total_monet_gen_loss.item(),
                    "photo_disc_loss": photo_disc_loss.item(),
                    "monet_disc_loss": monet_disc_loss.item(),
                },
                checkpoint=checkpoint,
            )


def main(data_dir: AnyStr, num_samples: int = 10):
    config = {
        "batch_size": tune.choice([1, 2, 4, 8, 16]),
    }

    scheduler = ASHAScheduler(
        metric="monet_gen_loss",
        mode="min",
        grace_period=1,
        reduction_factor=2,
    )

    result = tune.run(
        partial(tune_cycle_gan, data_dir=data_dir),
        resources_per_trial={"cpu": 2, "gpu": 1},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        storage_path="/mnt/AIArch/shinde/personal/monet_style_transfer/ray_results",
    )

    best_trail = result.get_best_trial(
        metric="monet_gen_loss", mode="min", scope="last"
    )
    print(f"Best trail config: {best_trail.config}")
    print(
        "Best trail loss - photo_gen_loss: {:.4f}; monet_gen_loss: {:.4f}; photo_disc_loss: {:.4f}; monet_disc_loss: {:.4f}\n".format(
            best_trail["photo_gen_loss"],
            best_trail["monet_gen_loss"],
            best_trail["photo_disc_loss"],
            best_trail["monet_disc_loss"],
        )
    )

    best_checkpoint = result.get_best_checkpoint(
        trial=best_trail, metric="monet_gen_loss", mode="min"
    )

    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = os.path.join(Path(checkpoint_dir), "data.pkl")
        with open(data_path, "rb") as fp:
            best_checkpoint_data = pickle.load(fp)

        file_path = os.path.join("../checkpoints", "best_ray_model.pth")
        torch.save(
            {"model_state_dict": best_checkpoint_data["model_state_dict"]}, file_path
        )


if __name__ == "__main__":
    data_dir = os.path.join(parent_dir, "data")
    main(data_dir=data_dir)
