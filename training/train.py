import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from nst_app.ml.models import Decoder, VGGEncoder
from nst_app.ml.utils import (
    ImageFolderDataset,
    adaptive_instance_normalization,
    calc_mean_std,
    get_transform,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--content-dir",
        default=str(PROJECT_ROOT / "datasets" / "content"),
        help="Location of the content dataset",
    )
    parser.add_argument(
        "--style-dir",
        default=str(PROJECT_ROOT / "datasets" / "style"),
        help="Location of the style dataset",
    )
    parser.add_argument(
        "--vgg",
        default=str(PROJECT_ROOT / "models" / "vgg_normalised.pth"),
        help="Location of the pretrained VGG weights",
    )
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--final-size", type=int, default=256)
    parser.add_argument("--content-size", type=int, default=512)
    parser.add_argument("--style-size", type=int, default=512)
    parser.add_argument(
        "--crop", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-decay", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--content-weight", type=float, default=1.0)
    parser.add_argument("--style-weight", type=float, default=5.0)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--decoder-path")
    parser.add_argument("--optimizer-path")
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.resume and (not args.decoder_path or not args.optimizer_path):
        raise ValueError(
            "--decoder-path and --optimizer-path are required with --resume"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = PROJECT_ROOT / "training" / "runs" / args.experiment
    save_dir.mkdir(exist_ok=True, parents=True)

    with (save_dir / "args.txt").open("w", encoding="utf-8") as args_file:
        for key, value in vars(args).items():
            args_file.write(f"{key}: {value}\n")

    content_dataset = ImageFolderDataset(
        args.content_dir,
        get_transform(args.content_size, args.crop, args.final_size),
    )
    style_dataset = ImageFolderDataset(
        args.style_dir,
        get_transform(args.style_size, args.crop, args.final_size),
    )
    content_loader = DataLoader(
        content_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    style_loader = DataLoader(
        style_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    print("Number of content batches:", len(content_loader))
    print("Number of style batches:", len(style_loader))

    encoder = VGGEncoder(args.vgg).to(device).eval()
    decoder = Decoder().to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 1.0 / (1.0 + args.lr_decay * epoch),
    )

    if args.resume:
        decoder.load_state_dict(torch.load(args.decoder_path, map_location=device))
        optimizer.load_state_dict(
            torch.load(args.optimizer_path, map_location=device)
        )

    mse_loss = torch.nn.MSELoss()
    print("Training...")

    for epoch in range(args.epochs):
        batch_count = min(len(content_loader), len(style_loader))
        progress = tqdm(
            zip(content_loader, style_loader),
            total=batch_count,
        )
        running_loss = 0.0
        running_content_loss = 0.0
        running_style_loss = 0.0

        for content_batch, style_batch in progress:
            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)

            content_features = encoder(content_batch)
            style_features = encoder(style_batch)
            target = adaptive_instance_normalization(
                content_features[-1], style_features[-1]
            )
            generated = decoder(target)
            generated_features = encoder(generated)

            content_loss = (
                mse_loss(generated_features[-1], target) * args.content_weight
            )
            style_loss = 0.0
            for generated_feature, style_feature in zip(
                generated_features, style_features
            ):
                generated_mean, generated_std = calc_mean_std(generated_feature)
                style_mean, style_std = calc_mean_std(style_feature)
                style_loss += mse_loss(generated_mean, style_mean)
                style_loss += mse_loss(generated_std, style_std)
            style_loss *= args.style_weight

            loss = content_loss + style_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress.set_description(
                f"Loss: {loss.item():.4f}, "
                f"Content: {content_loss.item():.4f}, "
                f"Style: {style_loss.item():.4f}"
            )
            running_loss += loss.item()
            running_content_loss += content_loss.item()
            running_style_loss += style_loss.item()

        scheduler.step()
        running_loss /= batch_count
        running_content_loss /= batch_count
        running_style_loss /= batch_count

        if (epoch + 1) % args.log_interval == 0:
            tqdm.write(
                f"Epoch {epoch + 1}: Loss: {running_loss:.4f}, "
                f"Content: {running_content_loss:.4f}, "
                f"Style: {running_style_loss:.4f}"
            )

        if (epoch + 1) % args.save_interval == 0:
            torch.save(decoder.state_dict(), save_dir / f"decoder_{epoch + 1}.pth")
            torch.save(
                optimizer.state_dict(),
                save_dir / f"optimizer_{epoch + 1}.pth",
            )
            with torch.no_grad():
                output = torch.cat(
                    [content_batch, style_batch, generated], dim=0
                )
                save_image(
                    output,
                    save_dir / f"output_{epoch + 1}.png",
                    nrow=args.batch_size,
                )


if __name__ == "__main__":
    main()
