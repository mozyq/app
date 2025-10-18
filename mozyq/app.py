from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer()


@app.command("normalize")
def mzq_normalize(
    in_folder: Annotated[Path, typer.Argument(
        help="Input folder with images to normalize.")],

    out_folder: Annotated[Path, typer.Argument(
        help="Output folder for normalized images.")],

    min_width: Annotated[int, typer.Option(
        help="Minimum width of input images.")] = 630,

    min_height: Annotated[int, typer.Option(
        help="Minimum height of input images.")] = 630,

    target_width: Annotated[int, typer.Option(
        help="Target width of output images.")] = 630,

    target_height: Annotated[int, typer.Option(
        help="Target height of output images.")] = 630,
):
    """Normalize images in a folder to a specific size and format."""
    try:
        # Validate input folder exists
        if not in_folder.exists():
            typer.echo(
                f"❌ Error: Input folder '{in_folder}' does not exist.", err=True)
            raise typer.Exit(1)

        if not in_folder.is_dir():
            typer.echo(f"❌ Error: '{in_folder}' is not a directory.", err=True)
            raise typer.Exit(1)

        # Check if input folder has any images
        image_extensions = {'.jpg', '.jpeg', '.png',
                            '.bmp', '.tiff', '.tif', '.webp'}
        image_files = [f for f in in_folder.iterdir()
                       if f.is_file() and f.suffix.lower() in image_extensions]

        if not image_files:
            typer.echo(
                f"❌ Error: No supported image files found in '{in_folder}'.", err=True)
            typer.echo(
                f"   Supported formats: {', '.join(image_extensions)}", err=True)
            raise typer.Exit(1)

        typer.echo(
            f"📁 Processing {len(image_files)} images from '{in_folder}'")
        typer.echo(f"📤 Output folder: '{out_folder}'")

        from mozyq.norm import normalize

        normalize(
            in_folder=in_folder,
            out_folder=out_folder,
            min_width=min_width,
            min_height=min_height,
            target_width=target_width,
            target_height=target_height,
        )

        typer.echo("✅ Normalization completed successfully!")

    except ImportError as e:
        typer.echo(
            f"❌ Error: Failed to import required modules: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Error during normalization: {e}", err=True)
        raise typer.Exit(1)


@app.command("json")
def mzq_json(
    master: Annotated[Path, typer.Argument(
        ..., help="The seed image to base the Mozyq video on.")],

    output_json: Annotated[Path, typer.Argument(
        help="Path to output JSON file.")] = Path("mzq.json"),

    width: Annotated[int, typer.Option(
        help="Output width.")] = 630,

    height: Annotated[int, typer.Option(
        help="Output height.")] = 630,

    num_tiles: Annotated[int, typer.Option(
        help="Number of tiles in the grid.")] = 21,

    max_transitions: Annotated[int, typer.Option(
        help="Maximum number of transitions.")] = 10,

):
    """Create a Mozyq JSON file from a folder of images and a seed image."""
    try:
        # Validate master image exists
        if not master.exists():
            typer.echo(
                f"❌ Error: Master image '{master}' does not exist.", err=True)
            raise typer.Exit(1)

        if not master.is_file():
            typer.echo(f"❌ Error: '{master}' is not a file.", err=True)
            raise typer.Exit(1)

        # Validate image format
        image_extensions = {'.jpg', '.jpeg', '.png',
                            '.bmp', '.tiff', '.tif', '.webp'}
        if master.suffix.lower() not in image_extensions:
            typer.echo(
                f"❌ Error: Master image must be one of: {', '.join(image_extensions)}", err=True)
            raise typer.Exit(1)

        # Validate tile folder (parent directory of master)
        tile_folder = master.parent
        if not tile_folder.exists() or not tile_folder.is_dir():
            typer.echo(
                f"❌ Error: Tile folder '{tile_folder}' does not exist or is not a directory.", err=True)
            raise typer.Exit(1)

        # Check if tile folder has enough images
        tile_files = [f for f in tile_folder.iterdir()
                      if f.is_file() and f.suffix.lower() in image_extensions]

        required_tiles = num_tiles * num_tiles
        if len(tile_files) < required_tiles:
            typer.echo(
                f"❌ Error: Need at least {required_tiles} images for {num_tiles}x{num_tiles} grid.", err=True)
            typer.echo(
                f"   Found only {len(tile_files)} images in '{tile_folder}'", err=True)
            raise typer.Exit(1)

        # Validate num_tiles is odd
        if num_tiles % 2 == 0:
            typer.echo(
                f"❌ Error: num_tiles must be odd, got {num_tiles}", err=True)
            raise typer.Exit(1)

        # Validate dimensions are divisible by num_tiles
        if width % num_tiles != 0:
            typer.echo(
                f"❌ Error: width ({width}) must be divisible by num_tiles ({num_tiles})", err=True)
            raise typer.Exit(1)

        if height % num_tiles != 0:
            typer.echo(
                f"❌ Error: height ({height}) must be divisible by num_tiles ({num_tiles})", err=True)
            raise typer.Exit(1)

        typer.echo(f"🖼️  Master image: '{master}'")
        typer.echo(
            f"📁 Tile folder: '{tile_folder}' ({len(tile_files)} images)")
        typer.echo(f"📐 Grid: {num_tiles}x{num_tiles} = {required_tiles} tiles")
        typer.echo(f"📏 Dimensions: {width}x{height}")
        typer.echo(f"🔄 Max transitions: {max_transitions}")

        from mozyq.mzq import gen_mzq_json

        gen_mzq_json(
            master=master,
            tile_folder=tile_folder,
            width=width,
            height=height,
            num_tiles=num_tiles,
            max_transitions=max_transitions,
            output_json=output_json,
        )

        typer.echo(f"✅ Mozyq JSON created successfully: '{output_json}'")

    except ImportError as e:
        typer.echo(
            f"❌ Error: Failed to import required modules: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Error during JSON generation: {e}", err=True)
        raise typer.Exit(1)


@app.command("frames")
def mzq_frames(
    mzq_json: Annotated[Path, typer.Argument(
        help="Path to the Mozyq JSON file generated by the json command.")],

    out_folder: Annotated[Path, typer.Argument(
        help="Output folder for the frames.")],

):
    """Generate frames from a Mozyq JSON file."""
    try:
        # Validate JSON file exists
        if not mzq_json.exists():
            typer.echo(
                f"❌ Error: Mozyq JSON file '{mzq_json}' does not exist.", err=True)
            raise typer.Exit(1)

        if not mzq_json.is_file():
            typer.echo(f"❌ Error: '{mzq_json}' is not a file.", err=True)
            raise typer.Exit(1)

        typer.echo(f"📋 Loading Mozyq data from: '{mzq_json}'")
        typer.echo(f"📤 Output frames to: '{out_folder}'")

        import json

        from mozyq.frame import frames
        from mozyq.mozyq_types import Mozyq

        # Load and validate JSON
        try:
            with open(mzq_json) as f:
                json_data = json.load(f)
        except json.JSONDecodeError as e:
            typer.echo(f"❌ Error: Invalid JSON file: {e}", err=True)
            raise typer.Exit(1)

        if not isinstance(json_data, list):
            typer.echo(
                "❌ Error: JSON file must contain a list of Mozyq objects.", err=True)
            raise typer.Exit(1)

        if not json_data:
            typer.echo("❌ Error: JSON file is empty.", err=True)
            raise typer.Exit(1)

        # Convert to Mozyq objects with proper error handling
        mzqs = []
        for i, mzq_dict in enumerate(json_data):
            try:
                if not isinstance(mzq_dict, dict):
                    raise ValueError(f"Item {i} is not a valid object")

                if 'master' not in mzq_dict or 'tiles' not in mzq_dict:
                    raise ValueError(
                        f"Item {i} missing 'master' or 'tiles' field")

                # Convert string paths to Path objects
                mzq = Mozyq(
                    master=Path(mzq_dict['master']),
                    tiles=[Path(tile) for tile in mzq_dict['tiles']]
                )

                # Validate that files exist
                if not mzq.master.exists():
                    typer.echo(
                        f"⚠️  Warning: Master image '{mzq.master}' not found (item {i})")

                missing_tiles = [
                    tile for tile in mzq.tiles if not tile.exists()]
                if missing_tiles:
                    typer.echo(
                        f"⚠️  Warning: {len(missing_tiles)} tile(s) not found in item {i}")

                mzqs.append(mzq)

            except Exception as e:
                typer.echo(f"❌ Error processing item {i}: {e}", err=True)
                raise typer.Exit(1)

        typer.echo(f"📊 Loaded {len(mzqs)} Mozyq sequences")

        # Generate frames
        frames(mzqs, out_folder)

        typer.echo("✅ Frame generation completed successfully!")

    except ImportError as e:
        typer.echo(
            f"❌ Error: Failed to import required modules: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Error during frame generation: {e}", err=True)
        raise typer.Exit(1)


def main():
    app()
