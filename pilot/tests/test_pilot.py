"""CPU tests. Fake editor tests validate orchestration, NOT pretrained capability."""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from pilot.common import REPO, read, write, blank_annotation, recover_xyz
from pilot.measure import measure, rule_feedback
from pilot.data import plan, save_motion, translate_features, corrupt
from pilot.backend import parse_motion
from pilot.review import import_review, export_review
from pilot.experiment import refine
from pilot.report import report

CONFIG = read(REPO / 'pilot/configs/pilot.json')

def features(length=40):
    f = np.zeros((length, 263), dtype=np.float32)
    f[:, 3] = 1
    joints = f[:, 4:67].reshape(length, 21, 3)
    joints[:, :, 1] = 1
    for j in [7, 8, 10, 11]:
        joints[:, j-1, 1] = 0
    joints[:, [6, 9], 0] = .1
    joints[:, [7, 10], 0] = -.1
    return f

def reviewed(xyz, key='generated_03/initial'):
    a = blank_annotation(key, xyz, 20, 'A person stands still.')
    a.update(reviewed=True, reviewer='SYNTHETIC_TEST_FIXTURE', units_confirmed=True,
             floor_verified=True, support_reviewed=True, labels_complete=True,
             support=[dict(body_part='left_foot', anchor='toe', start=0, end=len(xyz))],
             required_support=[dict(start=0, end=len(xyz))])
    return a

class Measurements(unittest.TestCase):
    def test_support_displacement_is_not_hidden_by_speed(self):
        xyz = recover_xyz(features())
        xyz[:, 10, 0] += np.arange(len(xyz)) * .01
        m = measure(xyz, 20, reviewed(xyz), CONFIG['thresholds'])
        self.assertAlmostEqual(m['summary']['sliding_mean_m_s'], .2, places=5)
        event = next(e for e in m['events'] if e['defect'] == 'sliding')
        self.assertEqual((event['start'], event['end'], event['body_part']), (0, 39, 'left_foot'))

    def test_known_penetration(self):
        xyz = recover_xyz(features())
        xyz[10:20, 10, 1] = -.04
        m = measure(xyz, 20, reviewed(xyz), CONFIG['thresholds'])
        self.assertAlmostEqual(m['summary']['penetration_max_m'], .04, places=6)
        e = next(e for e in m['events'] if e['defect'] == 'penetration')
        self.assertEqual((e['start'], e['end']), (10, 20))

    def test_jump_and_running_flight_are_not_floating(self):
        xyz = recover_xyz(features())
        xyz[10:25, :, 1] += .3
        a = reviewed(xyz)
        a.update(support=[dict(body_part='left_foot', anchor='toe', start=0, end=10)],
                 required_support=[dict(start=0, end=10), dict(start=25, end=40)],
                 airborne=[dict(start=10, end=25)])
        m = measure(xyz, 20, a, CONFIG['thresholds'])
        self.assertFalse(any(e['defect'] == 'floating' for e in m['events']))

    def test_raised_swing_foot_not_floating(self):
        xyz = recover_xyz(features())
        xyz[:, [8, 11], 1] += .4
        self.assertEqual(measure(xyz, 20, reviewed(xyz), CONFIG['thresholds'])['summary']['floating_duration_s'], 0)

    def test_required_support_float(self):
        xyz = recover_xyz(features())
        xyz[10:20, :, 1] += .2
        m = measure(xyz, 20, reviewed(xyz), CONFIG['thresholds'])
        e = next(e for e in m['events'] if e['defect'] == 'floating')
        self.assertEqual((e['start'], e['end']), (10, 20))

    def test_unknown_support_stays_unknown(self):
        m = measure(recover_xyz(features()), 20, {}, CONFIG['thresholds'])
        self.assertIsNone(m['summary']['sliding_mean_m_s'])
        self.assertEqual(rule_feedback(m)['decision'], 'uncertain')

    def test_review_cannot_conflict_with_airborne(self):
        xyz = recover_xyz(features())
        a = reviewed(xyz)
        a['airborne'] = [dict(start=10, end=20)]
        with self.assertRaises(ValueError):
            measure(xyz, 20, a, CONFIG['thresholds'])

    def test_support_overlap_cannot_double_count(self):
        xyz = recover_xyz(features())
        a = reviewed(xyz)
        a['support'].append(dict(body_part='left_foot', anchor='toe', start=4, end=10))
        with self.assertRaises(ValueError):
            measure(xyz, 20, a, CONFIG['thresholds'])

    def test_stale_review_is_rejected(self):
        xyz = recover_xyz(features())
        a = reviewed(xyz)
        xyz[4, 0, 0] = 1
        with self.assertRaises(ValueError):
            measure(xyz, 20, a, CONFIG['thresholds'])

    def test_coherent_translation_with_turning(self):
        f = features()
        f[:, 0] = .025
        f[:, 1:3] = [.01, .02]
        xyz = recover_xyz(f)
        d = np.zeros((len(f), 3), dtype=np.float32)
        d[5:30, 0] = np.sin(np.linspace(0, np.pi, 25)) * .12
        d[5:30, 1] = np.sin(np.linspace(0, np.pi, 25)) * -.08
        altered = translate_features(f, d)
        np.testing.assert_allclose(recover_xyz(altered), xyz + d[:, None], atol=5e-6)
        np.testing.assert_array_equal(altered[:, 67:193], f[:, 67:193])

    def test_no_reflooring(self):
        xyz = recover_xyz(features())
        xyz[:, :, 1] -= .1
        self.assertAlmostEqual(measure(xyz, 20, reviewed(xyz), CONFIG['thresholds'])['summary']['penetration_max_m'], .1, places=6)

    def test_parser_rejects_fallback_and_invalid_codes(self):
        for value in ('No refinements needed.', '<Motion><Motion_1>', '<Motion>bad</Motion>', '<Motion><Motion_9999></Motion>'):
            with self.assertRaises(ValueError):
                parse_motion(value, 512)
        self.assertEqual(parse_motion('<Motion><Motion_1><Motion_2></Motion>', 512).tolist(), [1, 2])

    def test_plan_grouping(self):
        rows = plan(42)
        self.assertEqual(len(rows), 50)
        self.assertEqual(sum(r['subset'] == 'calibration' for r in rows), 10)
        groups = {}
        for row in rows:
            self.assertEqual(groups.setdefault(row['source_recording'], row['subset']), row['subset'])

class FakeBackend:
    calls = []
    identity = {'backend': 'TEST_FAKE_NOT_IRG'}
    def __init__(self, config):
        pass
    def decode(self, tokens):
        return self.initial_features if tokens.tolist() == [1, 2] else features()
    def semantic(self, caption, tokens, seed):
        return dict(decision='defect', instruction='Test instruction', defects=[]), {'seconds': 0}
    def edit(self, caption, tokens, feedback, seed):
        self.calls.append((tokens.tolist(), seed, feedback['instruction']))
        return np.array([2, 3]), features(), {'seconds': 0}
    def session_record(self):
        return {'identity': self.identity, 'allocated_process_gpu_hours': 0, 'used_gpu_count': 0}

class Workflow(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.work = Path(self.temp.name)
        rows = plan(42)
        row = next(r for r in rows if r['clip_id'] == 'generated_03')
        self.f = features()
        self.f[10:20, 4 + (10-1)*3 + 1] -= .04
        self.xyz = save_motion(self.work, row, self.f, 'initial', [1, 2])
        FakeBackend.initial_features = self.f
        row['status'] = 'ready_for_review'
        write(self.work / 'manifest.json', rows)
        self.config = copy.deepcopy(CONFIG)
        FakeBackend.calls = []
    def tearDown(self):
        self.temp.cleanup()
    def test_no_fabricated_approval_and_resume(self):
        c = refine(self.work, self.config, 'test', FakeBackend)
        self.assertEqual(c['blocked'], 1)
        self.assertEqual(c['complete'], 3)
        self.assertEqual(len(FakeBackend.calls), 1)  # semantic only; rule abstains
        c = refine(self.work, self.config, 'test', FakeBackend)
        self.assertEqual(c['resumed'], 3)
        self.assertEqual(len(FakeBackend.calls), 1)
    def test_matched_inputs_and_unreviewed_outcomes(self):
        a = reviewed(self.xyz)
        event = dict(defect='penetration', body_part='left_foot', start=10, end=20)
        a['labels'] = [event]
        a['feedback'] = dict(approved=True, decision='defect', defects=[event], instruction='TEST FIXTURE approval')
        write(self.work / 'annotations.json', {a['key']: a})
        c = refine(self.work, self.config, 'test', FakeBackend)
        self.assertEqual(c['complete'], 4)
        self.assertEqual([tokens for tokens, _, _ in FakeBackend.calls], [[1, 2]] * 3)
        self.assertEqual([seed for _, seed, _ in FakeBackend.calls], [42] * 3)
        report(self.work, self.config, 'test')
        summaries = read(self.work / 'reports/test/comparison_summary.json')
        self.assertTrue(all(s['reviewed_defective_clips'] == 0 for s in summaries))
    def test_review_export_import_and_stale_rejection(self):
        out = self.work / 'review'
        export_review(self.work, self.config, out)
        payload = read(out / 'annotations_template.json')
        self.assertEqual(import_review(self.work, out / 'annotations_template.json', self.config), 0)
        payload['entries'][0] = reviewed(self.xyz)
        write(out / 'reviewed.json', payload)
        self.assertEqual(import_review(self.work, out / 'reviewed.json', self.config), 1)
        payload['entries'][0]['motion_hash'] = 'stale'
        write(out / 'bad.json', payload)
        with self.assertRaises(ValueError):
            import_review(self.work, out / 'bad.json', self.config)
    def test_second_human_round_waits_for_new_review(self):
        self.config['rounds'] = 2
        self.config['conditions'] = ['human']
        a = reviewed(self.xyz)
        a['feedback'] = dict(approved=True, decision='defect', instruction='TEST ONLY',
                             defects=[dict(defect='penetration', body_part='left_foot', start=10, end=20)])
        write(self.work / 'annotations.json', {a['key']: a})
        c = refine(self.work, self.config, 'test', FakeBackend)
        self.assertEqual(c['complete'], 1)
        self.assertEqual(c['blocked'], 1)
        key = 'test/generated_03/human/round_1'
        first = reviewed(recover_xyz(features()), key)
        first['feedback'] = dict(approved=True, decision='no_defect', defects=[], instruction='')
        write(self.work / 'annotations.json', {a['key']: a, key: first})
        c = refine(self.work, self.config, 'test', FakeBackend)
        self.assertEqual(c['resumed'], 1)
        self.assertEqual(c['complete'], 1)
        self.assertEqual(c['blocked'], 0)

    def test_cached_decoder_mismatch_is_rejected(self):
        FakeBackend.initial_features = features()  # does not match the penetrated input
        c = refine(self.work, self.config, 'test', FakeBackend)
        self.assertEqual(c['errors'], 1)
        self.assertEqual(FakeBackend.calls, [])

    def test_rule_second_round_waits_for_support(self):
        self.config.update(rounds=2, conditions=['rule'])
        a = reviewed(self.xyz)
        write(self.work / 'annotations.json', {a['key']: a})
        c = refine(self.work, self.config, 'test', FakeBackend)
        self.assertEqual(c['complete'], 1)
        self.assertEqual(c['blocked'], 1)
    def test_changed_config_cannot_overwrite_run(self):
        refine(self.work, self.config, 'test', FakeBackend)
        self.config['seed'] += 1
        with self.assertRaises(ValueError):
            refine(self.work, self.config, 'test', FakeBackend)

    def test_corruptions_require_acceptable_review_and_remain_unverified(self):
        rows = read(self.work / 'manifest.json')
        original = next(r for r in rows if r['clip_id'] == 'original_00')
        xyz = save_motion(self.work, original, features(), 'raw')
        write(self.work / 'manifest.json', rows)
        corrupt(self.work)
        child = next(r for r in read(self.work / 'manifest.json') if r['clip_id'] == 'original_00_sliding')
        self.assertFalse(child['paths'])
        ann = reviewed(xyz, 'original_00/raw')
        write(self.work / 'annotations.json', {ann['key']: ann})
        corrupt(self.work)
        child = next(r for r in read(self.work / 'manifest.json') if r['clip_id'] == 'original_00_sliding')
        self.assertIn('raw_263', child['paths'])
        self.assertNotIn('initial_tokens', child['paths'])
        self.assertEqual(child['defect_labels'], [])
        self.assertEqual(child['status'], 'pending_reconstruction')

    def test_pair_export_hides_condition_and_keeps_mapping_separate(self):
        refine(self.work, self.config, 'test', FakeBackend)
        output = self.work / 'pairs'
        count = export_review(self.work, self.config, output, 'pairs', 'test')
        self.assertEqual(count, 3)
        payload = read(output / 'annotations_template.json')
        self.assertTrue(all('/' not in e['key'] for e in payload['entries']))
        self.assertFalse((output / 'pairs_mapping.json').exists())
        self.assertEqual(len(read(self.work / 'private/pairs_mapping.json')), 3)

if __name__ == '__main__':
    unittest.main()
