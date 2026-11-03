import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.il_report import publish_bd_periodic as p
from tools.il_report.evidence import fingerprint, PROTOCOL, TOKEN_HASH


def write(path, value, *, sealed=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    if sealed:value=dict(value,sha256=fingerprint(value))
    path.write_text(json.dumps(value));return value


class PublicationTest(unittest.TestCase):
    def point(self, base, architecture='B',updates=64000,*,complete=False,scenes=12146):
        root=base/f'periodic_NAVTEST_ARCH_{architecture}_{updates:06d}_STAMP';label=f'{architecture}_IL_{updates:06d}'
        write(root/'PINNED.json',dict(models={label:dict(architecture_id=architecture,updates=updates)}))
        d=root/label;identity=write(d/'evaluation/identity.json',dict(cohort='NAVTEST',seed=2101,
            qualification=False,token_sha256=TOKEN_HASH,protocol=dict(sha256=PROTOCOL),
            inference='single_candidate_cleanx_K10_shift5'),sealed=True)
        means={k:1.0 for k in ['no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound',
                              'ego_progress','comfort','driving_direction_compliance']};means['score']=.88
        write(d/'SUMMARY.json',dict(status='COMPLETE',cohort='NAVTEST',scenes=scenes,invalid_scenes=0,
            evaluation=str(d/'evaluation'),evaluation_identity_sha256=identity['sha256'],PDMS=88.,means=means),sealed=True)
        write(d/'registration.json',dict(updates=updates))
        if complete:
            for path in [root/'COMPLETE.json',d/'COMPLETE.json']:write(path,dict(status='COMPLETE',all_owned_workers_exited=True))
        return root

    def test_partial_or_unclosed_score_never_enters_published_table(self):
        with tempfile.TemporaryDirectory()as d:
            base=Path(d);self.point(base);self.point(base,'D',52000,complete=True)
            with patch.object(p,'FOUR',base),patch.object(p,'CONTROL',base/'control'):result=p.collect()
            self.assertEqual([(r['architecture'],r['updates'])for r in result['completed']],[('D',52000)])
            self.assertIsNone(result['pending'][0]['PDMS'])
            docs=p.documents(result)
            self.assertTrue(docs['README.md'].startswith('# B / D'))
            self.assertIn('| D | 52000 | 88.0000',docs['README.md'])

    def test_closed_but_wrong_population_fails_instead_of_publishing(self):
        with tempfile.TemporaryDirectory()as d:
            base=Path(d);self.point(base,complete=True,scenes=12000)
            with patch.object(p,'FOUR',base),patch.object(p,'CONTROL',base/'control'):
                with self.assertRaises(ValueError):p.collect()

    def test_nonreport_dirty_files_prevent_any_git_mutation(self):
        fake=lambda checkout,*args: p.BRANCH if args==('branch','--show-current')else'actor.py'
        with patch.object(p,'git',side_effect=fake):
            with self.assertRaisesRegex(ValueError,'Unrelated edits'):p.publish(Path('/unused'),{})


if __name__=='__main__':unittest.main()
