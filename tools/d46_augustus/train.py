"""AUGUSTUS-inspired D46 policy; shared trainer/atomic recovery, no loop copy."""
from starVLA.rl.d46_augustus.runtime import AugustusExtension
from tools.d44_native_rl.train import main


if __name__=='__main__':main(extension=AugustusExtension())
