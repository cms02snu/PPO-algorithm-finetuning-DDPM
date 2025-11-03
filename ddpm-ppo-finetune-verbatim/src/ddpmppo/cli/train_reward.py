\
import sys, runpy
from pathlib import Path

def main():
    impl_file = Path(__file__).resolve().parent.parent / "impl" / "train_reward_model_v2_colab.py"
    sys.argv = ["train_reward_model_v2_colab.py"] + sys.argv[1:]
    runpy.run_path(str(impl_file), run_name="__main__")

if __name__ == "__main__":
    main()
