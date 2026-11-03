"""Strict native world16 continuation; no fresh weights or optimizer restart."""
from starVLA.rl.d46_world16.runtime import World16Extension
from tools.d44_native_rl.train import main

if __name__=='__main__':main(extension=World16Extension())
