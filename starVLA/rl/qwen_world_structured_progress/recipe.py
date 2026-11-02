"""One extra teacher-only proposal pool; all original S6 science preserved."""
import copy
from starVLA.rl.qwen_world_verified_quality import recipe as S6
from starVLA.rl.qwen_world_postcot.contracts import fingerprint
from .geometry import DELTAS

ROOT=S6.ROOT
ARM='S7_STRUCTURED_PROGRESS'
SCHEMA='qwen_world_structured_progress_rl_v1'
CONFIG=dict(schema='world_verified_structured_progress_teacher_pool_v1',
    trigger='valid TRAIN AND trueB1 notstrict-high AND mainG16 no strict-high AND extraSDE16 no strict-high',
    families=['GT_path','B1_path'],deltas_at4_m=list(DELTAS),maximum_candidates=16,
    initial_velocity='official no-slip bicycle forward current vx at local heading0; logged vy retained as provenance',
    geometry='C2 clamped xy(cumulative-chord metre parameter), yaw from same tangent; beyondknownpath explicit constant terminal-curvature circular extrapolation',
    progress='s_base(t)+delta*(t/4)^3; new output times0.5..4s; initial position/velocity/addedacceleration unchanged',
    codec='CPU FP32 denorm(norm(synthetic_FP32)); actually scored bytes authoritative',
    GT='sealed NAVTRAIN+original rawlog hashes+official coordinate reconstruction+all8 ILtime gates; invalidGT skips only GT family',
    teacher='NC=DAC=TTC=Comfort1 AND score>=.9 AND gain_vs_trueB1>=.02; choose highest',
    role='typed synthetic offline teacher-only; never SDEchain/oldlogp/PPO/std/exposure or deployment oracle',
    B1_geometry_invalid='skip that family with explicit reason; never omit the main training scene',
    native_replay='saved source validity/family gate and exact scored endpoints fixed through four inners; no new raw-file availability decision or scoring on resume',
    maximum_new_queries_per_trigger=16)
PROTOCOL=dict(schema=SCHEMA,arm=ARM,structured_progress=copy.deepcopy(CONFIG),
    shared_S6=copy.deepcopy(S6.PROTOCOL),initialization='same complete IL70/fresh Adam0/RNG42, no RL checkpoint',
    main_PG='unchanged G16/W3 SafetyEP PPO and global64*16 rawA std',
    counters='separate rank-local/global structured-query ledger; old SDE extraG16 counters unchanged')
PROTOCOL_SHA256=fingerprint(PROTOCOL)


def build_recipe(base):
    S6.validate_recipe(base);cfg=copy.deepcopy(base);r=cfg['research']
    r.update(schema=SCHEMA,arm=ARM,family='structured_progress',structured_progress=copy.deepcopy(PROTOCOL))
    return validate_recipe(cfg)


def validate_recipe(cfg):
    r=cfg['research']
    if (r.get('schema')!=SCHEMA or r.get('arm')!=ARM or r.get('family')!='structured_progress'
            or r.get('structured_progress')!=PROTOCOL):raise ValueError('Exact independent structured-progress identity required')
    old=copy.deepcopy(cfg);q=old['research'];q.pop('structured_progress')
    q.update(schema=S6.SCHEMA,arm=S6.ARM,family='verified_quality')
    S6.validate_recipe(old)
    return cfg
