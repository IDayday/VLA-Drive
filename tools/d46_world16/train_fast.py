"""Typed same-world execution-upgrade entry; unchanged shared native trainer."""
from starVLA.rl.d46_world16.fast_runtime import FastWorld16Extension
from tools.d44_native_rl.train import main


if __name__=='__main__':main(extension=FastWorld16Extension())
