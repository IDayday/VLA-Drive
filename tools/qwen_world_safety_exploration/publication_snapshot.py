"""Export existing RL evidence for repository review; never controls a job."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
from urllib.parse import unquote, urlsplit


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


class PortableHTML(HTMLParser):
    """Retain inline figures; resolve bundled evidence, annotate other NAS links."""
    def __init__(self, original_root, mapping, as_of):
        super().__init__(convert_charrefs=False)
        self.root, self.mapping, self.as_of = original_root, mapping, as_of
        self.parts, self.anchors = [], []
        self.bundled_links = self.external_nas_links = 0

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == 'meta' and values.get('http-equiv', '').lower() == 'refresh':
            return
        if tag == 'a':
            href = values.get('href', '')
            if href and not href.startswith('#') and not urlsplit(href).scheme:
                path = Path(unquote(urlsplit(href).path))
                source = str((path if path.is_absolute() else self.root/path).resolve())
                if source in self.mapping:
                    self.parts.append('<a href="'+escape(self.mapping[source], quote=True)+'">')
                    self.bundled_links += 1
                    self.anchors.append('a')
                else:
                    self.parts.append('<span class="nas-reference" title="'+escape('NAS evidence: '+source, quote=True)+'">')
                    self.external_nas_links += 1
                    self.anchors.append('span')
                return
            self.anchors.append('a')
        self.parts.append(self.get_starttag_text())
        if tag == 'body':
            self.parts.append('<aside style="padding:12px 24px;background:#fff3cd;color:#574518">'
                '<b>仓库静态快照 · '+escape(self.as_of)+'</b>。图表已内嵌；已收录证据可离线打开。'
                '灰色NAS证据文字保留原始路径，不冒充随仓库发布。实时报告仍在NAS固定路径更新。</aside>')

    def handle_endtag(self, tag):
        self.parts.append('</'+(self.anchors.pop() if tag == 'a' and self.anchors else tag)+'>')

    def handle_startendtag(self, tag, attrs):
        self.parts.append(self.get_starttag_text())

    def handle_data(self, data):
        self.parts.append(data)

    def handle_entityref(self, name):
        self.parts.append('&'+name+';')

    def handle_charref(self, name):
        self.parts.append('&#'+name+';')

    def handle_comment(self, value):
        self.parts.append('<!--'+value+'-->')

    def handle_decl(self, value):
        self.parts.append('<!'+value+'>')


def export(report_root, output):
    report_root, output = report_root.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    mapping, inputs = {}, []

    def copy(source, relative):
        source = Path(source)
        if not source.is_file():
            return None
        target = output/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = source.read_bytes()
        target.write_bytes(raw)
        mapping[str(source.resolve())] = str(relative)
        inputs.append(dict(source=str(source), bundled=str(relative), bytes=len(raw),
                           sha256=hashlib.sha256(raw).hexdigest()))
        return str(relative)

    # The existing sole renderer holds this lock exclusively during publication.
    # This brief read lock captures one coherent report generation without restart.
    with (report_root/'reporting/render.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        snapshot = read(report_root/'reporting/report_snapshot.json')
        review_path = report_root/'analysis/REPORT_LATEST_COMPLETE_REVIEW_20261101/SUMMARY.json'
        review = read(review_path)
        document = (report_root/'report.html').read_text()
        copy(review_path, Path('results/latest_complete_review.json'))
        copy(report_root/'reporting/FRONT_PAGE_SUMMARY.json', Path('results/training_status.json'))
        for source in sorted((report_root/'reporting').glob('*.py')):
            copy(source, Path('reporting_source')/source.name)
        for source in sorted((report_root/'analysis/RL_RECIPE_CONFIGURATION_REVIEW_20261102').glob('*.json')):
            copy(source, Path('analysis/recipe_configurations')/source.name)

    points, sources = [], []
    studies = snapshot['editorial_analysis']['standard_studies']
    for study in studies:
        root = Path(study['root']); group = root.name
        copy(root/'RUNS.json', Path('experiments')/group/'RUNS.json')
        for name in ['REGISTRATION.json','EXPERIMENT_DESIGN.json','SCHEDULER_PROTOCOL.json','TRAINING_EXPOSURE.json','SOURCE_FREEZE.json']:
            copy(root/name, Path('experiments')/group/name)
        # Registry as observed by this report, rather than a later live mutation.
        registry = study['registry']
        write(output/'experiments'/group/'RUNS.json', registry)
        for run in registry['runs']:
            config = Path(run['config'])
            assert sha(config) == run['config_sha256'], 'Scientific config differs from registry'
            relative = Path('experiments')/group/'configs'/config.name
            copy(config, relative)
            sources.append(dict(arm=run['id'], study=group, source=run['source'],
                source_commit=run.get('source_commit'), execution_source=run.get('execution_source'),
                execution_migration_receipt=run.get('execution_migration_receipt'),
                bundled_config=str(relative), config_sha256=run['config_sha256'],
                target_updates=run['target_updates'], native_gates=run['gates'],
                evaluation_gates=run.get('evaluation_gates', run['gates']), parent_checkpoint=run['parent_checkpoint']))

    for arm in review['arms']:
        for point in arm['curve']:
            source = Path(point['evidence']); value = read(source)
            assert value['status'] == 'COMPLETE' and value['scenes'] == 12146 and value['invalid_scenes'] == 0
            base = Path('results/evaluations')/point['arm']/f"u{point['updates']:06}"
            relative = copy(source, base/'SUMMARY.json')
            for name in ['SPEC.json','COMPLETE']:
                copy(source.parent/name, base/name)
            identity = source.parent/'identity.json'
            if identity.is_file():
                receipt = base/'IDENTITY_RECEIPT.json'
                write(output/receipt, dict(schema='repository_evaluation_identity_receipt_v1',
                    original_path=str(identity), raw_identity_sha256=sha(identity),
                    evaluation_identity_sha256=value.get('evaluation_identity_sha256'),
                    protocol_sha256=value.get('protocol_sha256'),
                    model_sha256=value.get('model_sha256'),
                    data_manifest_sha256=value.get('data_manifest_sha256'),
                    token_sha256=value.get('token_sha256'), seed=value.get('seed'),
                    note='Full NAS identity payload is not bundled; this receipt preserves its hash without duplicating the complete scene-token manifest.'))
                mapping[str(identity.resolve())] = str(receipt)
            points.append(dict(arm=point['arm'], optimizer_updates=point['updates'],
                PDMS=point['PDMS'], zero_scenes=point.get('zeros'), scenes=12146,
                invalid_scenes=0, evidence=relative,
                kind='native0_initial_function' if point['updates']==0 else 'RL_checkpoint'))
    assert len(points) == review['complete_points']
    for index, pair in enumerate(review['comparisons']):
        copy(pair['evidence'], Path('results/paired')/f"{index:03}_{Path(pair['evidence']).name}")

    for key in ['new_pilot_matched_review','safety_mechanism','safety_process']:
        value = snapshot['editorial_analysis'].get(key)
        if value:
            write(output/'analysis'/f'legacy_{key}.json', value)
    for row in snapshot['results']:
        if row.get('arm') in ('PPO_FULL','PPO_SAFETY_EP') and row.get('publishable_complete') and row.get('cohort') == 'NAVTEST':
            copy(row['evidence'], Path('results/legacy')/row['arm']/f"u{row.get('RL_updates',row.get('updates',0)):06}.json")

    integrated = next(s for s in studies if Path(s['root']).name.startswith('integrated_'))
    base = Path(integrated['root'])
    for subfolder, names in [
        ('analysis/TRICK_REVIEW_20261102',['DIAGNOSTICS.json','NOMINAL_VS_SAMPLED.json','REFINEMENT_PROPOSALS.json','TRAIN_RESCUE_POOL_MANIFEST.json','QUALIFICATION.json']),
        ('operations/USER_10K_2K5K10K_20261102',['REQUEST.json','APPLIED.json','ACTUAL_CONTINUITY_PROOF.json','TARGETED_CHECKS.json']),
        ('operations/REPORTING_RECOVERY_20261102',['MIGRATION_RECEIPT.json','ACTUAL_RECOVERY_PROOF.json'])]:
        for name in names:
            copy(base/subfolder/name, Path('analysis/integrated')/Path(subfolder).name/name)
    campaign = base.parent
    for name in ['HANDOFF.json','ACTUAL_PROGRESS.json','CURRENT_OWNERS.json','PUBLICATION.json','SOURCE_FREEZE.json','SOURCE_FREEZE_V2.json','APPLIED.json','ADOPTION_V2_APPLIED.json']:
        copy(campaign/'operations/EVALUATION_BACKLOG_COMPLETION_20261102'/name, Path('operations/evaluation_completion')/name)

    write(output/'SCIENTIFIC_SOURCE_MAP.json', dict(schema='RL_publication_scientific_source_map_v1',
        runs=sources, notice='Publication HEAD is an integrated review checkout, not a replacement for frozen live scientific/execution sources.'))
    with (output/'results/scores.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(points[0]))
        writer.writeheader();writer.writerows(points)
    portable = PortableHTML(report_root, mapping, snapshot['generated_utc'])
    portable.feed(document);portable.close()
    (output/'report.html').write_text(''.join(portable.parts))
    for item in inputs:
        item['source_sha256']=item.pop('sha256')
        item['bundled_sha256']=sha(output/item['bundled'])
        item['bundled_bytes']=(output/item['bundled']).stat().st_size
    write(output/'PUBLICATION_MANIFEST.json', dict(schema='world_RL_repository_snapshot_v1',
        created_utc=datetime.now(timezone.utc).isoformat(), report_generated_utc=snapshot['generated_utc'],
        source_report=str(report_root/'report.html'), complete_points=len(points),
        paired_comparisons=review['complete_comparisons'], registered_recipes=len(sources),
        highest_complete=review['observed_peak'], input_files=inputs,
        portable_evidence_links=portable.bundled_links, annotated_unbundled_NAS_links=portable.external_nas_links,
        no_model_weights_or_datasets_exported=True, no_training_or_evaluation_mutations=True,
        original_source_branches=dict(science='research/qwen-world-safety-exploration-20261031@fcb2060',
            reporting='research/qwen-world-abcd-rl-best-20261031@84565f2',
            evaluation_operations='ops/qwen-world-eval-completion-20261102@c4e3451')))
    return snapshot, review, points


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--report-root', type=Path, default=Path('/mnt/project/qwen-world-rl-optimization-20261030'))
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args();_,review,points=export(a.report_root,a.output)
    print(json.dumps(dict(status='EXPORTED',directory=str(a.output),complete_points=len(points),
                         comparisons=review['complete_comparisons']),ensure_ascii=False))


if __name__ == '__main__':main()
