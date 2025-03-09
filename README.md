# JXR to JPEG/PNG Converter

Convert your JXR files to stunning JPEG or PNG formats with advanced HDR processing!

This tool was created to convert screenshots saved in JXR format by NVIDIA (formerly Geforce Experience) to PNG or HDR.

## Features

- **HDR Processing**: Choose from multiple HDR modes including Gamma, Tonemap, ACES Filmic, and Hable.
- **High Quality**: Maintain high image quality with adjustable JPEG quality settings.
- **Easy to Use**: Simple command-line interface for quick conversions.

## Installation

1. Clone the repository:

```sh
git clone https://github.com/fa0311/jxr-converter
cd jxr-converter
pip install -r requirements.txt
```

1. Install the required packages:

```sh
pip install -r requirements.txt
```

## Examples

Convert a JXR file to PNG with default settings:

```sh
python main.py --input input.jxr --output output.png
```

Convert a JXR file to JPEG with high quality:

```sh
python main.py --input input.jxr --output output.jpg --quality 100
```

## Usage

Display help and available options:

```sh
python main.py --help
```

## Advanced HDR Modes

Enhance your images with advanced HDR processing techniques:

- **Gamma Correction**: Apply gamma correction for natural brightness.
- **Tonemapping**: Use tonemapping to compress the dynamic range.
- **ACES Filmic**: Achieve cinematic looks with ACES Filmic tonemapping.
- **Hable**: Utilize Hable tonemapping for a balanced exposure.

```sh
python main.py --input input.jxr --output output.png --hdr-mode tonemap
```
