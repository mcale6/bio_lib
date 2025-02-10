import argparse
from pathlib import Path
import json
import sys
from datetime import datetime
import time
import statistics
from typing import Dict
import jax
from bio_lib.custom_prodigy import predict_binding_affinity
from bio_lib.custom_prodigy_jax import predict_binding_affinity_jax
from bio_lib.helpers.utils import collect_pdb_files, format_time, NumpyEncoder

def clear_mem():
    '''remove all data from current device'''
    device = jax.default_backend()
    backend = jax.lib.xla_bridge.get_backend(device)
    for buf in backend.live_buffers(): 
        buf.delete()

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
    quiet: bool = False,
    benchmark: bool = False
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
            n_runs = 3 if benchmark else 1
            times = []
            results = []
            
            for i in range(n_runs):
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
                
                execution_time = time.perf_counter() - start_time
                times.append(execution_time)
                results.append(result)

                # Clear GPU memory after 3 runs
                if benchmark and use_jax and i == n_runs-1:
                    clear_mem()

            result_dict = results[-1].to_dict()
            execution_times.append(times[-1])
            
            result_dict['execution_time'] = {
                'seconds': times[-1],
                'formatted': format_time(times[-1])
            }
            
            if benchmark:
                result_dict['execution_time'].update({
                    'benchmark_times': times,
                    'benchmark_mean': statistics.mean(times),
                    'benchmark_std': statistics.stdev(times)
                })
            
            all_results[pdb_file.stem] = result_dict
            print(f"Execution time: {format_time(times[-1])}")

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
    parser.add_argument("--benchmark", default=False, action="store_true",  help="Run same prediction 3 times")
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
            quiet=args.quiet,
            benchmark=args.benchmark  # Add this line
        )
        return 0
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())