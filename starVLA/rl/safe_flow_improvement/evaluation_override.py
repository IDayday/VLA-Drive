"""Explicit user-directed evaluation budget change; never rewrite training data IDs."""
import json
from pathlib import Path

from .contracts import file_sha, fingerprint


def evaluation_seeds(population, override_path=None):
    if override_path is None:
        return population['evaluation_seeds'], None
    record = json.loads(Path(override_path).read_text())
    checksum = record.pop('sha256')
    if fingerprint(record) != checksum:
        raise ValueError('Evaluation override checksum changed')
    if (record['population_identity'] != population['identity_sha256']
            or record['evaluation_seeds'] != [2101]
            or record['training_seeds'] != [42]
            or record['reference_seeds'] != population['reference_seeds']
            or not record['preserve_training_model_optimizer_bank_and_population']):
        raise ValueError('Only the explicit single-seed evaluation override is supported')
    if not set(record['evaluation_seeds']) <= set(population['evaluation_seeds']):
        raise ValueError('Evaluation seed must be from the original locked seed list')
    return record['evaluation_seeds'], dict(path=str(override_path), file_sha256=file_sha(override_path),
                                           identity_sha256=checksum, user_instruction=record['user_instruction'])
