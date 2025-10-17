
import json
import math
from pathlib import Path


def index_html(input_json: Path) -> str:
    """
    Generate an HTML page displaying a grid of tiles with master image blended overlay.

    Args:
        input_json: Path to JSON file containing master and tiles information

    Returns:
        HTML string containing the grid visualization
    """
    # Load the JSON data
    with open(input_json, 'r') as f:
        data = json.load(f)

    master_image = data['master']
    tiles = data['tiles']

    # Calculate grid dimensions (assuming square grid)
    num_tiles = len(tiles)
    grid_size = int(math.sqrt(num_tiles))

    # If not a perfect square, calculate rectangular grid
    if grid_size * grid_size != num_tiles:
        # Find the best rectangular arrangement
        for rows in range(1, int(math.sqrt(num_tiles)) + 1):
            if num_tiles % rows == 0:
                cols = num_tiles // rows
                grid_size = max(rows, cols)
                grid_rows = rows
                grid_cols = cols
                break
    else:
        grid_rows = grid_cols = grid_size

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mozyq Grid Visualization</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background-color: #000;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }}

        .grid-container {{
            position: relative;
            display: inline-block;
            width: 600px;
            height: 750px;
        }}

        .tiles-grid {{
            display: grid;
            grid-template-columns: repeat({grid_cols}, 1fr);
            grid-template-rows: repeat({grid_rows}, 1fr);
            gap: 0;
            width: 600px;
            height: 750px;
        }}

        .tile {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }}

        .master-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            opacity: 0.5;
        }}
    </style>
</head>
<body>
    <div class="grid-container">
        <div class="tiles-grid">"""

    # Add each tile to the grid
    for i, tile_path in enumerate(tiles):
        html += f"""
            <img src="{tile_path}" alt="Tile {i+1}" class="tile">"""

    html += f"""
        </div>
        <img src="{master_image}" alt="Master Image" class="master-overlay">
    </div>
</body>
</html>"""

    return html
