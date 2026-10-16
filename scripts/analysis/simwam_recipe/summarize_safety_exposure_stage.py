"""Compatibility entry for the previously published SAFETY25 report command."""
import argparse
from scripts.analysis.simwam_recipe.summarize_followup_stage import summarize as _summarize


def summarize(update, output):
    return _summarize('R_CANONICAL_X1_SAFETY25', update, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--update', type=int, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    summarize(args.update, args.output)
