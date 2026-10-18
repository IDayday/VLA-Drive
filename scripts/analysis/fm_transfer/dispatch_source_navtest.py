"""New fixed-stage lock; existing capacity admission, shards and GPU worker."""
from scripts.analysis.fm_transfer.dispatch_final_evaluation import main
from scripts.analysis.fm_transfer.source_process_navtest import SourceNavtestDispatch
if __name__=='__main__':main(runtime=SourceNavtestDispatch())
