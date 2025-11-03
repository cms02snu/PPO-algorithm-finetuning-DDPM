\
import sys, runpy
from pathlib import Path

def main():
    # Execute user's original file as the program, passing *all* argv (except the launcher name).
    impl_file = Path(__file__).resolve().parent.parent / "impl" / "finetune_ddpm_with_ppo_colab.py"
    sys.argv = ["finetune_ddpm_with_ppo_colab.py"] + sys.argv[1:]
    runpy.run_path(str(impl_file), run_name="__main__")

if __name__ == "__main__":
    main()
