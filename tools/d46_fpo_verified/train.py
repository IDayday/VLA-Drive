"""Independent verified FPO, using the mature shared native training loop."""
from starVLA.rl.d46_fpo_verified.runtime import VerifiedExtension
from tools.d44_native_rl.train import main

if __name__=='__main__':main(extension=VerifiedExtension())
