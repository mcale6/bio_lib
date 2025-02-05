import argparse
from pathlib import Path
import json
import sys
from datetime import datetime
import time
import statistics
from typing import Dict
from bio_lib.custom_prodigy import predict_binding_affinity
from bio_lib.custom_prodigy_jax import predict_binding_affinity_jax
from bio_lib.helpers.utils import collect_pdb_files, format_time, NumpyEncoder

def run(
    input_path: Path,
    output_dir: Path,
    selection: str = "A,B",
    use_jax: bool = False,
    temperature: float = 25.0,
    distance_cutoff: float = 5.5,
    acc_threshold: float = 0.05,
    sphere_points: int = 100,
    output_json: bool = True,
    quiet: bool = False
) -> Dict[str, Dict]:
    """Process all PDB files in the input path."""
    pdb_files = collect_pdb_files(input_path)
    all_results = {}
    execution_times = []
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    for pdb_file in pdb_files:
        try:
            start_time = time.perf_counter()
            
            if use_jax:
                result = predict_binding_affinity_jax(
                    struct_path=pdb_file,
                    selection=selection,
                    temperature=temperature,
                    distance_cutoff=distance_cutoff,
                    acc_threshold=acc_threshold,
                    sphere_points=sphere_points,
                    save_results=False,
                    output_dir=str(run_dir / pdb_file.stem),
                    quiet=quiet
                )
            else:
                result = predict_binding_affinity(
                    struct_path=pdb_file,
                    selection=selection,
                    temperature=temperature,
                    distance_cutoff=distance_cutoff,
                    acc_threshold=acc_threshold,
                    sphere_points=sphere_points,
                    save_results=False,
                    output_dir=str(run_dir / pdb_file.stem),
                    quiet=quiet
                )
            
            result_dict = result.to_dict()

            execution_time = time.perf_counter() - start_time
            execution_times.append(execution_time)
    
            result_dict['execution_time'] = {
                'seconds': execution_time,
                'formatted': format_time(execution_time)
            }
            all_results[pdb_file.stem] = result_dict
            print(f"Execution time: {format_time(execution_time)}")

            if output_json:
                output_path = run_dir / f"{pdb_file.stem}_results.json"
                print(f"Results saved in: {output_path}")
                with open(output_path, 'w') as f:
                    json.dump(result_dict, indent=2, cls=NumpyEncoder, fp=f)
                            
        except Exception as e:
            print(f"\nError processing {pdb_file.name}: {str(e)}")
            all_results[pdb_file.stem] = {
                "error": str(e),
                "execution_time": {"error": "Failed"}
            }

    if execution_times:
        all_results['_timing_summary'] = {
            'mean': statistics.mean(execution_times),
            'median': statistics.median(execution_times),
            'min': min(execution_times),
            'max': max(execution_times),
            'std': statistics.stdev(execution_times) if len(execution_times) > 1 else 0
        }
    
    # Save combined results
    if output_json:
        combined_output = run_dir / "combined_results.json"
        print(f"Combined Results saved in: {combined_output}")
        with open(combined_output, 'w') as f:
            json.dump(all_results, indent=2, cls=NumpyEncoder, fp=f)
    
    return all_results

def main() -> int:
    parser = argparse.ArgumentParser(description="Run PRODIGY analysis on protein complex(es)")
    parser.add_argument("input_path", type=Path, help="Path to PDB file or directory")
    parser.add_argument("--selection", type=str, default="A,B", 
                       help="Chain selection (e.g., 'A,B' Only chain A and B will be considered at the moment)")
    parser.add_argument("--use-jax", action="store_true", default=False, help="Use JAX-based implementation")
    parser.add_argument("--temperature", type=float, default=25.0, help="Temperature in Celsius (default: 25.0)")
    parser.add_argument("--distance-cutoff", type=float, default=5.5, help="Distance cutoff (Å) (default: 5.5)")
    parser.add_argument("--acc-threshold", type=float, default=0.05, help="Accessibility threshold (default: 0.05)")
    parser.add_argument("--sphere-points", type=int, default=100, help="Number of points on sphere for accessibility calculation (default: 100)")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Output directory")
    parser.add_argument("--quiet", default=False, action="store_true", help="Outputs only the predicted affinity value")
    parser.add_argument("--output-json", default=False, action="store_true",  help="Output format results")
    args = parser.parse_args()
    
    if not args.input_path.exists():
        print(f"Error: Input path not found: {args.input_path}", file=sys.stderr)
        return 1
        
    try:
        run(
            input_path=args.input_path,
            output_dir=args.output_dir,
            selection=args.selection,
            use_jax=args.use_jax,
            temperature=args.temperature,
            distance_cutoff=args.distance_cutoff,
            acc_threshold=args.acc_threshold,
            sphere_points=args.sphere_points,
            output_json=args.output_json,
            quiet=args.quiet
        )
        return 0
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())