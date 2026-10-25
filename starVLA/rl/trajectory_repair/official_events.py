"""Offline read-only observation of accepted official NC/TTC rule events.

The canonical scorer methods execute unchanged. A bounded Python trace reads
their locals at the existing event-time assignment. No alternate collision or
TTC formula is introduced. Use only in isolated offline label-export workers;
this adds Python tracing overhead and is not a training/deployment hot path.
"""
import ast
import inspect
import sys
import textwrap

import numpy as np


def event_assignment_line(method, attribute):
    lines, first = inspect.getsourcelines(method)
    tree = ast.parse(textwrap.dedent(''.join(lines)))
    matches = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Assign)
               for target in node.targets if isinstance(target, ast.Subscript)
               and isinstance(target.value, ast.Attribute) and target.value.attr == attribute]
    if len(matches) != 1:
        raise ValueError('Official event assignment changed; qualify the new scorer explicitly')
    return first + matches[0] - 1


class OfficialEventTrace:
    def __init__(self, scorer_class):
        nc = scorer_class._calculate_no_at_fault_collision
        ttc = scorer_class._calculate_ttc
        self.targets = {
            nc.__code__: ('NC', event_assignment_line(nc, '_collision_time_idcs')),
            ttc.__code__: ('TTC', event_assignment_line(ttc, '_ttc_time_idcs')),
        }
        self.collections = []
        self.current = None
        self.scorer_id = None
        self.active = False

    def _trace(self, frame, event, arg):
        target = self.targets.get(frame.f_code)
        if target is None:
            return None
        kind, line = target
        values = frame.f_locals
        if event == 'call':
            if kind == 'NC':
                self.current = dict(NC=[], TTC=[])
                self.collections.append(self.current)
                self.scorer_id = id(values['self'])
            elif self.current is None or self.scorer_id != id(values['self']):
                raise ValueError('TTC trace has no matching completed scene scorer invocation')
        if event == 'line' and frame.f_lineno == line:
            item = dict(proposal_index=int(values['proposal_idx']), physical_time_index=int(values['time_idx']),
                        object_token=str(values['token']))
            if kind == 'NC':
                collision = values['collision_type']
                item.update(collision_type=getattr(collision, 'name', str(collision)),
                            official_NC_value=float(values['no_at_fault_collision_score']))
            else:
                item.update(projection_offset_index=int(values['future_time_idx']),
                            object_time_index=int(values['current_time_idx']))
            self.current[kind].append(item)
        return self._trace

    def __enter__(self):
        if self.active or sys.gettrace() is not None:
            raise RuntimeError('Offline event export requires an isolated interpreter without another trace')
        self.active = True
        sys.settrace(self._trace)
        return self

    def __exit__(self, *_):
        sys.settrace(None)
        self.active = False

    def attach(self, records):
        if self.active or len(records) != len(self.collections):
            raise ValueError('Expected one unmodified independent official reference/candidate pair per record')
        for record, collection in zip(records, self.collections):
            h = np.asarray(record['simulated_states']).shape[0]
            if h != 41:
                raise ValueError('Expected physical 0.1s/4s simulation window')
            events = {kind: [e for e in entries if e['proposal_index'] == 1]
                      for kind, entries in collection.items()}
            for kind, field in [('NC', 'official_at_fault_collision_by_time'),
                                ('TTC', 'official_ttc_rule_infraction_by_time')]:
                mask = np.zeros(h, dtype=bool)
                for item in events[kind]:
                    if not 0 <= item['physical_time_index'] < h:
                        raise ValueError('Out-of-window official event')
                    mask[item['physical_time_index']] = True
                record[field] = mask
            record['official_event_trace'] = events
            record['event_trace_semantics'] = (
                'Accepted official NC/TTC rule events at each driving-time index; '
                'not geometric overlap duration, minimum TTC, or an edit direction. '
                'TTC object_time_index includes its official projection offset.')
        return records


def score_scene_with_events(job):
    from starVLA.rl.safe_flow_improvement import v1_reward
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    if v1_reward.score_scene.__module__ != v1_reward.__name__:
        raise ValueError('Offline trace exporter requires the original independent-pair scorer entry')
    with OfficialEventTrace(PDMScorer) as trace:
        records = v1_reward.score_scene(job)
    return trace.attach(records)
