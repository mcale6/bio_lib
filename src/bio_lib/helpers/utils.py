from pathlib import Path
from typing import List
from datetime import datetime

def collect_pdb_files(input_path: Path) -> List[Path]:
    """Collect all PDB files from input path."""
    if input_path.suffix.lower() in ['.pdb', '.ent']:
        return [input_path]
    elif input_path.is_dir():
        # If directory, collect all PDB files
        pdb_files = list(input_path.glob('*.pdb')) + list(input_path.glob('*.ent'))
        if not pdb_files:
            raise ValueError(f"No PDB files found in directory: {input_path}")
        return sorted(pdb_files)
    else:
        raise ValueError(f"Input path must be a PDB file or directory, got: {input_path}")

def format_time(seconds: float) -> str:
    """Format time in seconds to a human-readable string."""
    if seconds < 1:
        return f"{seconds*1000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes}m {seconds:.2f}s"

def setup_output_path(pdb_path: Path, output_dir: Path) -> Path:
    """Setup output directory and generate unique output filename."""
    # Create a subdirectory with current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_subdir = output_dir / timestamp
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    base_name = pdb_path.stem
    return output_subdir / f"{base_name}_results.json"
