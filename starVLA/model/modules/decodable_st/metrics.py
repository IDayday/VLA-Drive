import torch


def occupancy_counts(predicted,occupied,valid):
    p,y,v=predicted.bool(),occupied.bool(),valid.bool()
    return torch.stack(((p&y&v).sum(),(p&~y&v).sum(),(~p&y&v).sum(),(~p&~y&v).sum(),(~v).sum())).cpu()


def summarize_counts(counts):
    tp,fp,fn,tn,unknown=map(int,counts)
    def ratio(n,d): return n/d if d else None
    return {"tp":tp,"fp":fp,"fn":fn,"tn":tn,"unknown":unknown,
        "occupied_iou":ratio(tp,tp+fp+fn),"precision":ratio(tp,tp+fp),"recall":ratio(tp,tp+fn),
        "false_free":ratio(fn,tp+fn),"unknown_fraction":ratio(unknown,tp+fp+fn+tn+unknown)}
