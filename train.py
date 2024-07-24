import torch
from src.cycle_gan import Generator, Discriminator, CycleGAN
from src.dataloader import CustomImageDataset


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Data loader
DATA_SIZE = 300
monet_ds = CustomImageDataset("./data/monet_jpg", num_imgs=DATA_SIZE)
monet_dl = torch.utils.data.DataLoader(monet_ds, shuffle=True)

photo_ds = CustomImageDataset("./data/photo_jpg", num_imgs=DATA_SIZE)
photo_dl = torch.utils.data.DataLoader(photo_ds, shuffle=True)

# Define model
monet_generator = Generator()
photo_generator = Generator()
monet_discriminator = Discriminator()
photo_discriminator = Discriminator()

cycle_gan_model = CycleGAN(
    monet_generator=monet_generator,
    monet_discriminator=monet_discriminator,
    photo_generator=photo_generator,
    photo_discriminator=photo_discriminator,
)

cycle_gan_model.to(device=DEVICE)
cycle_gan_model.train(monet_ds=monet_dl, photo_ds=photo_dl, epochs=25, save_models=True)
