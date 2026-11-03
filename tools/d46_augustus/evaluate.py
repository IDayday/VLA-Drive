"""Original official-protocol evaluator with the AUGUstus native-state loader."""
from tools.d44_native_rl import evaluate as native_evaluator
from starVLA.rl.d46_augustus.deployment import load_policy


if __name__=='__main__':
    native_evaluator.load_policy=load_policy
    native_evaluator.main()
