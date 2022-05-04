from neodroidvision.data.synthesis.conversion.image_composition import ImageComposition

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Image Composition")
    parser.add_argument(
        "--input_dir",
        type=str,
        dest="input_dir",
        required=True,
        help=(
            "The input directory. This contains a 'backgrounds' directory of pngs or jpgs, and a 'foregrounds' "
            "directory which contains super category directories (e.g. 'animal', 'vehicle'), each of which contain "
            "category directories (e.g. 'horse', 'bear'). Each category directory contains png images of that item on "
            "a transparent background (e.g. a grizzly bear on a transparent background)."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        dest="output_dir",
        required=True,
        help="The directory where "
        "images, masks, \
                                                        and json files will be placed",
    )
    parser.add_argument(
        "--count",
        type=int,
        dest="count",
        required=True,
        help="number of composed images to create",
    )
    parser.add_argument(
        "--width",
        type=int,
        dest="width",
        required=True,
        help="output image pixel width",
    )
    parser.add_argument(
        "--height",
        type=int,
        dest="height",
        required=True,
        help="output image pixel height",
    )
    parser.add_argument(
        "--output_type", type=str, dest="output_type", help="png or jpg (default)"
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="silent mode; doesn't prompt the user for input, automatically overwrites files",
    )

    config = parser.parse_args()

    image_comp = ImageComposition()
    image_comp(config)
