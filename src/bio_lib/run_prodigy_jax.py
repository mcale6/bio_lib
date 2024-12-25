#!/usr/bin/env python3
import argparse
from pathlib import Path
import json
import sys
from datetime import datetime

from custom_prodigy_jax import run

def setup_output_path(pdb_path: Path, output_dir: Path) -> Path:
    """Setup output directory and generate output filename."""
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename based on PDB name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = pdb_path.stem
    return output_dir / f"{base_name}_results_{timestamp}.json"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PRODIGY analysis on a protein complex"
    )
    parser.add_argument(
        "pdb_path",
        type=Path,
        help="Path to PDB file"
    )
    parser.add_argument(
        "target_chain",
        type=str,
        default="A",
        help="Chain ID for target protein"
    )
    parser.add_argument(
        "binder_chain",
        type=str,
        default="B",
        help="Chain ID for binder protein"
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=5.5,
        help="Distance cutoff for contacts (Å)"
    )
    parser.add_argument(
        "--acc_threshold",
        type=float,
        default=0.05,
        help="rel_sasa >= acc_threshold is considered accessible (default: 0.05)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory (default: ./results)"
    )
    parser.add_argument(
        "--format",
        choices=["json", "human", "both"],
        default="both",
        help="Output format (default: both)"
    )
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    
    # Check input file
    if not args.pdb_path.exists():
        print(f"Error: PDB file not found: {args.pdb_path}", file=sys.stderr)
        return 1
    
    # Run analysis
    results = run(
        args.pdb_path,
        args.target_chain,
        args.binder_chain,
        acc_threshold=args.acc_threshold,
        cutoff=args.cutoff
    )
    
    # Setup output path
    output_path = setup_output_path(args.pdb_path, args.output_dir)
    
    # Handle output based on format
    if args.format in ["json", "both"]:
        # Use the to_dict method for JSON output
        json_data = results.to_dict()
        output_path.write_text(json.dumps(json_data, indent=2))
        print(f"\nResults saved to: {output_path}")

    if args.format in ["human", "both"]:
        # Use the built-in string representation
        print("\n" + str(results))
        
    return 0
if __name__ == "__main__":
    sys.exit(main())