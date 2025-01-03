#!/usr/bin/env python3
import argparse
from pathlib import Path
import json
import sys
from datetime import datetime
import time
import statistics
from typing import List, Dict
from .custom_prodigy_jax import run

def setup_output_path(pdb_path: Path, output_dir: Path) -> Path:
    """Setup output directory and generate output filename."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = pdb_path.stem
    return output_dir / f"{base_name}_results_{timestamp}.json"

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

def process_structures(
    input_path: Path,
    output_dir: Path,
    target_chain: str = "A",
    binder_chain: str = "B",
    use_jax_class: bool = True,
    acc_threshold: float = 0.05,
    cutoff: float = 5.5,
    output_format: str = "both"
) -> Dict[str, Dict]:
    """Process all PDB files in the input path."""
    pdb_files = collect_pdb_files(input_path)
    all_results = {}
    execution_times = []
    
    print(f"\nProcessing {len(pdb_files)} PDB file(s)...")
    
    for pdb_file in pdb_files:
        try:
            # Time the entire run
            start_time = time.perf_counter()
            results = run(
                pdb_file,
                target_chain,
                binder_chain,
                use_jax_class,
                acc_threshold=acc_threshold,
                cutoff=cutoff
            )
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            execution_times.append(execution_time)
            
            # Add results with timing
            json_data = results.to_dict()
            json_data['execution_time'] = {
                'seconds': execution_time,
                'formatted': format_time(execution_time)
            }
            all_results[pdb_file.stem] = json_data
            
            # Save individual results
            if output_format in ["json", "both"]:
                output_path = setup_output_path(pdb_file, output_dir)
                output_path.write_text(json.dumps(json_data, indent=2))
            
            if output_format in ["human", "both"]:
                print(f"\nResults for {pdb_file.name}:")
                print(str(results))
                print(f"Execution time: {format_time(execution_time)}")
                
        except Exception as e:
            print(f"\nError processing {pdb_file.name}: {str(e)}")
            all_results[pdb_file.stem] = {
                "error": str(e),
                "execution_time": {"error": "Failed to complete processing"}
            }
    
    # Calculate and add timing statistics
    if execution_times:
        timing_stats = {
            'mean': statistics.mean(execution_times),
            'median': statistics.median(execution_times),
            'min': min(execution_times),
            'max': max(execution_times),
            'std': statistics.stdev(execution_times) if len(execution_times) > 1 else 0
        }
        
        print("\n=== Timing Summary ===")
        print(f"Mean: {format_time(timing_stats['mean'])}")
        print(f"Median: {format_time(timing_stats['median'])}")
        print(f"Min: {format_time(timing_stats['min'])}")
        print(f"Max: {format_time(timing_stats['max'])}")
        print(f"Std: {format_time(timing_stats['std'])}")
        
        all_results['_timing_summary'] = timing_stats
    
    # Save combined results
    combined_output = output_dir / f"combined_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    combined_output.write_text(json.dumps(all_results, indent=2))
    print(f"\nCombined results saved to: {combined_output}")
    
    return all_results

def main() -> int:
    parser = argparse.ArgumentParser(description="Run PRODIGY analysis on protein complex(es)")
    parser.add_argument("input_path", type=Path, help="Path to PDB file or directory")
    parser.add_argument("target_chain", type=str, default="A", nargs="?", help="Target chain ID (default: A)")
    parser.add_argument("binder_chain", type=str, default="B", nargs="?", help="Binder chain ID (default: B)")
    parser.add_argument("--use-jax-class", type=bool, default=True, help="Use Custom Jax Class (default: True)")
    parser.add_argument("--cutoff", type=float, default=5.5, help="Distance cutoff (Å) (default: 5.5)")
    parser.add_argument("--acc-threshold", type=float, default=0.05, help="Accessibility threshold (default: 0.05)")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Output directory")
    parser.add_argument("--format", choices=["json", "human", "both"], default="both", help="Output format")
    args = parser.parse_args()
    
    if not args.input_path.exists():
        print(f"Error: Input path not found: {args.input_path}", file=sys.stderr)
        return 1
    
    try:
        process_structures(
            args.input_path,
            args.output_dir,
            args.target_chain,
            args.binder_chain,
            args.use_jax_class,
            args.acc_threshold,
            args.cutoff,
            args.format
        )
        return 0
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())