import argparse

from engine.inferencer import run_inference
from models import SUPPORTED_MODELS
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference for infrared image super-resolution.")
    parser.add_argument("--input", type=str, required=True, help="Input image path or input directory.")
    parser.add_argument("--output_dir", type=str, default="outputs/infer", help="Directory to save inference outputs.")
    parser.add_argument("--model", type=str, default=None, choices=SUPPORTED_MODELS, help="Model name; optional if inferred from --checkpoint.")
    parser.add_argument("--scale", type=int, default=None, help="Upscale factor; optional if inferred from --checkpoint.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path. Default: outputs/checkpoints/{model}_x{scale}_best.pth")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan images when --input is a directory.")
    parser.add_argument("--save_bicubic", action="store_true", help="Also save bicubic-upsampled baseline image.")
    parser.add_argument("--gt", type=str, default=None, help="Optional ground-truth image path or directory.")
    parser.add_argument("--save_visuals", action="store_true", help="When --gt is provided, save LR/SR/HR comparison and SR-HR difference map.")
    parser.add_argument("--log_file", type=str, default="outputs/logs/infer.log", help="Log file path.")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger(name="infer", log_file=args.log_file)
    run_inference(args=args, logger=logger)


if __name__ == "__main__":
    main()
