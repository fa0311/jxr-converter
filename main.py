import argparse
import logging
import os
import traceback
from enum import Enum

import cv2
import imagecodecs
import numpy as np
from colorama import Fore, Style, init
from pydantic import Field

from utils.arg_parser import ArgsParserBase, ArgsParserConfig


class CustomFormatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: f"{Fore.LIGHTBLACK_EX}🐞 [DEBUG] %(message)s{Style.RESET_ALL}",
        logging.INFO: f"{Fore.GREEN}✔ [INFO] %(message)s{Style.RESET_ALL}",
        logging.WARNING: f"{Fore.YELLOW}⚠ [WARN] %(message)s{Style.RESET_ALL}",
        logging.ERROR: f"{Fore.RED}✘ [ERROR] %(message)s{Style.RESET_ALL}",
        logging.CRITICAL: f"{Fore.RED}{Style.BRIGHT}‼ [CRITICAL] %(message)s{Style.RESET_ALL}",
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self._fmt)
        formatter = logging.Formatter(log_fmt)
        formatter.formatException = self.formatException
        formatter.formatStack = self.formatStack
        return formatter.format(record)

    def formatException(self, ei):
        exc_text = "".join(traceback.format_exception(*ei))
        exc_text = " |  " + exc_text.strip().replace("\n", "\n |  ")
        return f"{Fore.LIGHTBLACK_EX}{exc_text}{Style.RESET_ALL}"

    def formatStack(self, stack_info):
        stack_info = " |  " + stack_info.strip().replace("\n", "\n |  ")
        return f"{Fore.LIGHTBLACK_EX}{stack_info}{Style.RESET_ALL}"


class HDRMode(str, Enum):
    gamma = "gamma"
    tonemap = "tonemap"
    keep = "keep"
    aces_filmic = "aces_filmic"
    hable = "hable"


class JXRConvertArgs(ArgsParserBase):
    parser = ArgsParserConfig(
        argparse.ArgumentParser(
            description="Convert JXR to JPEG/PNG (HDR supported)",
            exit_on_error=False,
        )
    )

    input: str = Field(
        **(
            parser.add_argument(
                lambda x: x.add_argument("--input", help="Input JXR file")
            )
        )
    )
    output: str = Field(
        **(
            parser.add_argument(
                lambda x: x.add_argument(
                    "-o", "--output", required=True, help="Output file name (JPEG/PNG)"
                )
            )
        )
    )
    quality: int = Field(
        **(
            parser.add_argument(
                lambda x: x.add_argument(
                    "-q",
                    "--quality",
                    type=int,
                    default=90,
                    help="JPEG quality (1-100, default: 90)",
                )
            )
        )
    )
    hdr_mode: HDRMode = Field(
        **(
            parser.add_argument(
                lambda x: x.add_argument(
                    "-hdr",
                    "--hdr-mode",
                    choices=[e.value for e in HDRMode],
                    default="tonemap",
                    help="HDR processing method",
                )
            )
        ),
    )
    debug: bool = Field(
        **(
            parser.add_argument(
                lambda x: x.add_argument(
                    "--debug", action="store_true", help="Enable debug mode"
                )
            )
        )
    )


GAMMA_CORRECTION = 2.2
TONEMAP_A = 0.18
EPSILON = 1e-6


def compute_auto_exposure(image: np.ndarray, target: float = 1.4) -> float:
    luminance = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
    log_avg = np.exp(np.mean(np.log(luminance + EPSILON)))
    return target / log_avg


def gamma(image: np.ndarray) -> np.ndarray:
    return image ** (1 / GAMMA_CORRECTION)


def tonemap(image: np.ndarray) -> np.ndarray:
    luminance = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
    log_avg_lum = np.exp(np.mean(np.log(luminance + EPSILON)))
    scale = TONEMAP_A / log_avg_lum
    mapped = (scale * image) / (1 + scale * image)
    return gamma(mapped)


def aces_filmic_tonemap(image: np.ndarray) -> np.ndarray:
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return (image * (a * image + b)) / (image * (c * image + d) + e)


def hable_tonemap(image: np.ndarray, exposure: float = 8.0) -> np.ndarray:
    A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
    x = exposure * image
    return (x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F) - E / F


def convert_jxr(args: JXRConvertArgs):
    try:
        image = imagecodecs.imread(args.input)
    except Exception as e:
        logger.error(f"Error: Could not open JXR file → {e}")
        return

    image_np = np.array(image).astype(np.float32)

    if args.hdr_mode == HDRMode.tonemap:
        image_np = tonemap(image_np)
    elif args.hdr_mode == HDRMode.gamma:
        image_np = gamma(image_np)
    elif args.hdr_mode == HDRMode.keep:
        pass
    elif args.hdr_mode == HDRMode.aces_filmic:
        image_np = aces_filmic_tonemap(image_np)
    elif args.hdr_mode == HDRMode.hable:
        image_np = hable_tonemap(image_np)
    else:
        raise ValueError(f"Unknown HDR mode: {args.hdr_mode}")

    image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    ext = os.path.splitext(args.output)[1].lower()

    if ext in [".jpg", ".jpeg"]:
        cv2.imwrite(args.output, image_cv, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
    else:
        cv2.imwrite(args.output, image_cv)
    logger.info(f"Saved as {args.output}")


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    init(autoreset=True)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)

    np.seterr(all="call")
    np.seterrcall(lambda *args: logger.debug(f"numpy warning: {args}", stack_info=True))

    try:
        args = JXRConvertArgs.parse_args()
        ch.setLevel(logging.DEBUG if args.debug else logging.INFO)
        convert_jxr(args)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
