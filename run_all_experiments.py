"""
Run all experiments at once
"""

import subprocess
import sys
from pathlib import Path

# Experiments to run
EXPERIMENTS = [
    "experiments/exp_01_pairwise_ranking_demo.py",
    "experiments/exp_02_rankboost_demo.py",
    "experiments/exp_03_auc_roc_demo.py",
    "experiments/exp_04_margin_vs_error.py",
]

def run_all_experiments():
    """Run all experiments sequentially."""
    print("\n" + "="*70)
    print("RUNNING ALL EXPERIMENTS")
    print("="*70)
    
    project_root = Path(__file__).parent
    completed = 0
    failed = 0
    
    for i, exp_file in enumerate(EXPERIMENTS, 1):
        exp_path = project_root / exp_file
        exp_name = exp_path.stem
        
        print(f"\n[{i}/{len(EXPERIMENTS)}] Running {exp_name}...")
        print("-" * 70)
        
        try:
            result = subprocess.run(
                [sys.executable, str(exp_path)],
                cwd=str(project_root),
                capture_output=False
            )
            
            if result.returncode == 0:
                print(f"[OK] {exp_name} completed successfully")
                completed += 1
            else:
                print(f"[FAIL] {exp_name} failed with code {result.returncode}")
                failed += 1
                
        except Exception as e:
            print(f"[FAIL] {exp_name} error: {e}")
            failed += 1
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Completed: {completed}/{len(EXPERIMENTS)}")
    print(f"Failed: {failed}/{len(EXPERIMENTS)}")
    
    if failed == 0:
        print("\n[OK] All experiments completed successfully!")
        print(f"Check outputs/figures/ for generated plots")
    else:
        print(f"\n[FAIL] {failed} experiment(s) failed")
    
    print("="*70)


if __name__ == "__main__":
    run_all_experiments()
