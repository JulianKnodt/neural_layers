import experiments
import argparse

experiments = {
  "mnist": experiments.mnist,
  "coco": experiments.coco,
}

def arguments():
  a = argparse.ArgumentParser(description="experiment runner")
  a.add_argument(
    "experiment",
    choices=experiments.keys(),
    help="Which experiment to run"
  )
  return a.parse_args()

def main():
  args = arguments()
  experiments[args.experiment]()

if __name__ == "__main__": main()
