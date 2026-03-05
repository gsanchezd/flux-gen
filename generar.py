import torch
from diffusers import FluxPipeline
import argparse


def main():
    parser = argparse.ArgumentParser(description="Generate images from text using FLUX.1-schnell")
    parser.add_argument("prompt", nargs="?", default="a cat in space", help="Text prompt")
    parser.add_argument("-o", "--output", default="resultado.png", help="Output file path")
    parser.add_argument("-s", "--steps", type=int, default=4, help="Inference steps")
    args = parser.parse_args()

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")

    image = pipe(args.prompt, num_inference_steps=args.steps).images[0]
    image.save(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
