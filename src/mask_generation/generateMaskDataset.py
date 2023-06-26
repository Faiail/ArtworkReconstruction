import glob
import os
import traceback
from joblib import Parallel, delayed
import torchvision.transforms as transforms
from PIL import Image
from MaskGenerator import MaskGenerator
from tqdm import tqdm

# ----
# Main
# ----
def main(args):

    # Checks on input artworks path
    if not args.indir.endswith('/'):
        args.indir += '/'

    # If not exist, then make the output directory specified
    os.makedirs(args.outdir, exist_ok=True)

    # Get a list of image file names in the specified input directory
    in_files = list(glob.glob(os.path.join(args.indir, '**', f'*.{args.ext}'), recursive=True))

    # ---------------
    # Mask Generation
    # ---------------
    if args.n_jobs == 0:
        # No parallel execution
        generate_masks(in_files, args.indir, args.outdir)
    else:
        # Parallel execution using the configured number of concurrent jobs
        in_files_n = len(in_files)
        chunk_size = in_files_n // args.n_jobs + (1 if in_files_n % args.n_jobs > 0 else 0)
        Parallel(n_jobs=args.n_jobs)(
            delayed(generate_masks)(in_files[start:start + chunk_size], args.indir, args.outdir)
            for start in range(0, len(in_files), chunk_size)
        )



# ----
# Masks Generation
# ----
def generate_masks(src_images, indir, outdir):

    # for every image file name in the give list
    for infile in tqdm(src_images):
        try:
            # Checks on paths
            if not outdir.endswith('/'):
                outdir += '/'

            os.makedirs(os.path.dirname(outdir), exist_ok=True)

            # Get artwork name
            os.chdir(indir)
            infile = os.path.basename(infile)
            artwork_name = os.path.splitext(infile)[0]

            # Get the artwork image
            image = Image.open(infile)

            # Resize the artwork to 512x512x3
            new_dim = (512, 512)
            n_variants = 5
            resize = transforms.Resize(new_dim)
            resized_artwork = resize(image)

            # Save the resized artwork
            os.chdir(outdir)
            resized_artwork.save(f"{artwork_name}_crop.png")

            # Generate a set of masks
            generator = MaskGenerator()
            generated_masks = generator(resized_artwork, n_variants)

            # Save the set of generated masks
            i = 1
            for mask in generated_masks:
                mask.save(f"{artwork_name}_crop_mask{i}.png")
                i += 1

        except KeyboardInterrupt:
            return
        except Exception as ex:
            print(f'Could not make masks for {infile} due to {ex}:\n{traceback.format_exc()}')





# ----
# Utility
# ----
if __name__ == '__main__':
    import argparse

    # Parsing necessary arguments for all mask generation operations
    aparser = argparse.ArgumentParser()
    aparser.add_argument('indir', type=str, help='Path to folder with images')
    aparser.add_argument('outdir', type=str, help='Path to folder to store masks to')
    aparser.add_argument('--n-jobs', type=int, default=0, help='How many processes to use')
    aparser.add_argument('--ext', type=str, default='jpg', help='Input image extension')

    main(aparser.parse_args())