import torch
from diffusers import Flux2KleinPipeline
import argparse


def main():
    parser = argparse.ArgumentParser(description="Generate images using FLUX.2-klein-4B")
    parser.add_argument("prompt", nargs="?", default="a cat in space", help="Text prompt")
    parser.add_argument("-o", "--output", default="resultado.png", help="Output file path")
    parser.add_argument("-s", "--steps", type=int, default=4, help="Inference steps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4B",
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()

    generator = None
    if args.seed is not None:
        generator = torch.Generator("cpu").manual_seed(args.seed)

    image = pipe(
        prompt=args.prompt,
        height=1024,
        width=1024,
        guidance_scale=1.0,
        num_inference_steps=args.steps,
        generator=generator,
    ).images[0]
    image.save(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
