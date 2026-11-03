"""Reuse the full native trainer with an explicit qualification-only extension."""
from starVLA.rl.d46_fresh_aux.integration import FreshAuxQualification
from tools.d44_native_rl.train import main

if __name__=='__main__':main(extension=FreshAuxQualification())
