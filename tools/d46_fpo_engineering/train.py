"""Continue a typed performance execution using the shared native trainer."""
from starVLA.rl.d46_fpo_engineering.runtime import PerformanceExtension
from tools.d44_native_rl.train import main

if __name__ == '__main__':
    main(extension=PerformanceExtension())
