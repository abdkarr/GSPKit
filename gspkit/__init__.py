from pathlib import Path
from importlib import resources

PROJECT_DIR = str(Path(__file__).parents[1])

STYLE_FILE = Path(resources.files("gspkit"), "styles.txt")