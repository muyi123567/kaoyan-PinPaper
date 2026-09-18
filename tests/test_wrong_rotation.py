"""
错题轮换 + 按做错次数加权 单元测试

回归两个真实缺陷：

缺陷 A（错题池无轮换）：paper_engine 里 seen 只作用于新题，错题优先池写着"永不排除"，
等于完全没有记忆 —— 选谁纯靠打分，分数相近就反复抽中同几道。实测数二 40 道错题跑 8 轮：
被抽次数分布 {1:15, 2:9, 3:3, 4:3, 5:2}，有题抽 5 次，**8 道一次都没抽到**。

缺陷 B（联考丢设置）：generate_bundle 从没读 exclude_seen / historical_seen_question_ids /
priority_pool_ids / priority_ratio。实测：exclude_seen=True 时 5 轮仍有 30 道新题重复；
错题占比设 100%，首卷 22 题里只出 1 道错题。

修法：轮换打底（本轮未练的错题优先，全练过则清空开新一轮）+ 轮内按 wrong_count 加权。
"""
from __future__ import annotations

import random
import tempfile
from pathlib import Path

import pytest

from core.bank_loader import BankLoader
from core.models import PaperMode, SubjectType
from core.paper_engine import EngineRequest, PaperEngine
from core.state_manager import StateManager


@pytest.fixture(scope="module")
def pool_880():
    lo = BankLoader(subject=SubjectType.MATH_2)
    lo.load()
    return [q for q in lo.questions_by_id.values() if getattr(q, "book", "880") == "880"]


@pytest.fixture
def mgr(tmp_path):
    return StateManager(username="rot_test", subject=SubjectType.MATH_2, base_dir=tmp_path)


def _draw(pool, mgr, seed, ratio=0.5, use_rotation=True):
    """跑一次组卷，返回题号列表。"""
    wrong_ids = mgr.get_active_wrong_question_ids()
    eng = PaperEngine(pool)
    req = EngineRequest(
        title="t", subject=SubjectType.MATH_2, mode=PaperMode.FULL_10_6_6,
        target_chapters=[], seed=seed,
        historical_seen_question_ids=set(mgr.historical_seen_ids),
        candidate_question_pool=pool,
        priority_pool_ids=wrong_ids, priority_ratio=ratio, exclude_seen=True,
        priority_practiced_ids=set(mgr.wrong_rotation_ids) if use_rotation else set(),
        priority_wrong_counts=mgr.get_wrong_counts(wrong_ids),
    )
    return [q.id for q in eng.generate_single_paper(req).questions]


# ---------------------------------------------------------------- 缺陷 A


def test_every_wrong_question_gets_a_turn(pool_880, mgr):
    """核心回归：一轮内每道错题都该轮到，不该有题长期不出现。

    修复前实测 40 道里 8 道一次没抽到；修复后应全部覆盖。
    """
    wrong = [q.id for q in pool_880[:40]]
    mgr.batch_mark_wrong(wrong)

    hits: dict[str, int] = {}
    rng = random.Random(11)
    for _ in range(8):
        got = _draw(pool_880, mgr, rng.randint(1000, 99999))
        mgr.record_paper_generation([q for q in pool_880 if q.id in set(got)])
        mgr.record_wrong_practice(got)
        for qid in got:
            if qid in set(wrong):
                hits[qid] = hits.get(qid, 0) + 1

    never = [q for q in wrong if q not in hits]
    assert not never, f"仍有 {len(never)} 道错题一次都没被抽到，轮换失效"


def test_rotation_ledger_resets_after_full_round(pool_880, mgr):
    """活跃错题全部练过一遍后账本自动清空，开启新一轮。"""
    wrong = [q.id for q in pool_880[:6]]
    mgr.batch_mark_wrong(wrong)

    saw_reset = False
    rng = random.Random(3)
    for _ in range(10):
        got = _draw(pool_880, mgr, rng.randint(1000, 99999), ratio=1.0)
        if mgr.record_wrong_practice(got):
            saw_reset = True
            break
    assert saw_reset, "错题全部练过后账本没有重置，会卡在'无题可轮'"
    assert mgr.wrong_rotation_ids == set(), "重置后账本应为空"


def test_stubborn_questions_drawn_more_often_within_a_round(pool_880):
    """轮内按 wrong_count 加权：错得多的题该更常被抽中（显著高于均匀分配）。"""
    wl = pool_880[:40]
    stubborn = {q.id for q in wl[:5]}

    total = 0
    stub_hits = 0
    for t in range(40):
        with tempfile.TemporaryDirectory() as td:
            m = StateManager(username=f"w{t}", subject=SubjectType.MATH_2, base_dir=Path(td))
            m.batch_mark_wrong([q.id for q in wl])
            for q in wl[:5]:
                m.set_wrong_count(q.id, 5)
            wrong_ids = m.get_active_wrong_question_ids()
            got = _draw(pool_880, m, 1000 + t * 13, ratio=0.5)
            drawn = [i for i in got if i in wrong_ids]
            total += len(drawn)
            stub_hits += sum(1 for i in drawn if i in stubborn)

    assert total > 0
    share = stub_hits / total
    uniform = 5 / 40  # 均匀分配下顽固题应占的比例
    assert share > uniform * 1.3, (
        f"顽固题占比 {share:.1%}，均匀应为 {uniform:.1%}，加权没生效"
    )


def test_new_questions_never_enter_rotation_ledger(pool_880, mgr):
    """账本只登记确实属于错题本的题号，新题不该混进来。"""
    wrong = [q.id for q in pool_880[:5]]
    mgr.batch_mark_wrong(wrong)
    new_ids = [q.id for q in pool_880[100:110]]

    mgr.record_wrong_practice(new_ids)
    assert mgr.wrong_rotation_ids == set(), "新题被错误登记进了错题轮换账本"


def test_unmarked_question_does_not_block_round_end(pool_880, mgr):
    """账本里残留已取消标记的旧题号，不该阻碍轮次结束（issubset 而非等值比较）。"""
    wrong = [q.id for q in pool_880[:4]]
    mgr.batch_mark_wrong(wrong)
    mgr.record_wrong_practice(wrong[:2])
    # 取消标记一道已练的，再练完剩下的
    mgr.remove_wrong_question(wrong[0])
    did_reset = mgr.record_wrong_practice(wrong[2:])
    assert did_reset, "残留题号阻碍了轮次结束"


# ---------------------------------------------------------------- URL 同步


def test_rotation_url_code_roundtrip(pool_880, tmp_path):
    """账本位图 URL 编解码保真（与 seen 同构）。"""
    canon = [q.id for q in pool_880]
    src = StateManager(username="rot_src", subject=SubjectType.MATH_2, base_dir=tmp_path)
    src.batch_mark_wrong(canon[:10])
    src.record_wrong_practice(canon[:4])
    practiced = set(src.wrong_rotation_ids)
    assert practiced, "前置条件：账本得非空"

    code = src.rotation_to_url_code(canon)
    dst = StateManager(username="rot_dst", subject=SubjectType.MATH_2, base_dir=tmp_path)
    n, status = dst.apply_rotation_url_code(code, canon)
    assert status == "ok"
    assert dst.wrong_rotation_ids == practiced and n == len(practiced)


def test_rotation_url_code_merges_not_overwrites(pool_880, tmp_path):
    """跨设备并集合并：手机练的 + 电脑练的应汇总，而非互相冲掉。"""
    canon = [q.id for q in pool_880]
    phone = StateManager(username="rot_phone", subject=SubjectType.MATH_2, base_dir=tmp_path)
    phone.batch_mark_wrong(canon[:10])
    phone.record_wrong_practice(canon[:3])
    code = phone.rotation_to_url_code(canon)

    pc = StateManager(username="rot_pc", subject=SubjectType.MATH_2, base_dir=tmp_path)
    pc.batch_mark_wrong(canon[:10])
    pc.record_wrong_practice(canon[5:8])
    before = set(pc.wrong_rotation_ids)

    pc.apply_rotation_url_code(code, canon)
    assert pc.wrong_rotation_ids == before | set(canon[:3]), "应并集合并"


def test_rotation_url_code_stale_on_bank_change(pool_880, tmp_path):
    """题库签名不匹配时判 stale，不错位恢复。"""
    canon = [q.id for q in pool_880]
    src = StateManager(username="rot_s2", subject=SubjectType.MATH_2, base_dir=tmp_path)
    src.batch_mark_wrong(canon[:6])
    src.record_wrong_practice(canon[:3])
    code = src.rotation_to_url_code(canon)

    dst = StateManager(username="rot_d2", subject=SubjectType.MATH_2, base_dir=tmp_path)
    n, status = dst.apply_rotation_url_code(code, canon[:-1])
    assert status == "stale" and n == 0


def test_rotation_ledger_persists_across_reload(pool_880, tmp_path):
    """账本落盘，重开应用后本轮进度不丢。"""
    canon = [q.id for q in pool_880]
    m = StateManager(username="rot_persist", subject=SubjectType.MATH_2, base_dir=tmp_path)
    m.batch_mark_wrong(canon[:8])
    m.record_wrong_practice(canon[:3])
    expected = set(m.wrong_rotation_ids)

    again = StateManager(username="rot_persist", subject=SubjectType.MATH_2, base_dir=tmp_path)
    assert again.wrong_rotation_ids == expected


# ---------------------------------------------------------------- 缺陷 B


def test_bundle_honors_wrong_ratio(pool_880, mgr):
    """联考套卷必须遵守错题占比。修复前占比拉满首卷也只出 1 道错题。"""
    wrong = [q.id for q in pool_880[:80]]
    mgr.batch_mark_wrong(wrong)
    wrong_ids = mgr.get_active_wrong_question_ids()

    eng = PaperEngine(pool_880)
    req = EngineRequest(
        title="t", subject=SubjectType.MATH_2, mode=PaperMode.BUNDLE_3_PAPERS,
        target_chapters=[], seed=4242, candidate_question_pool=pool_880,
        priority_pool_ids=wrong_ids, priority_ratio=1.0, exclude_seen=True,
        priority_wrong_counts=mgr.get_wrong_counts(wrong_ids),
    )
    bundle = eng.generate_bundle(req, bundle_size=3)
    first = [q.id for q in bundle.papers[0].questions]
    n_wrong = sum(1 for i in first if i in wrong_ids)
    assert n_wrong >= len(first) // 2, (
        f"错题占比设 100%，首卷 {len(first)} 题只出 {n_wrong} 道错题，联考仍在无视 priority_ratio"
    )


def test_bundle_honors_exclude_seen(pool_880, mgr):
    """联考套卷必须遵守 exclude_seen。修复前 5 轮有 30 道新题重复。"""
    seen: set[str] = set()
    repeats = 0
    rng = random.Random(5)
    for _ in range(4):
        eng = PaperEngine(pool_880)
        req = EngineRequest(
            title="t", subject=SubjectType.MATH_2, mode=PaperMode.BUNDLE_3_PAPERS,
            target_chapters=[], seed=rng.randint(1000, 99999),
            historical_seen_question_ids=set(seen),
            candidate_question_pool=pool_880, exclude_seen=True,
        )
        bundle = eng.generate_bundle(req, bundle_size=3)
        got = [q.id for p in bundle.papers for q in p.questions]
        repeats += sum(1 for i in got if i in seen)
        seen |= set(got)

    assert repeats == 0, f"exclude_seen=True 但联考仍重复抽了 {repeats} 道已抽过的题"


def test_bundle_still_dedups_and_covers_chapters(pool_880):
    """护栏：错题抽取插在章节分桶之前，不得破坏三卷不重复与章节覆盖。"""
    eng = PaperEngine(pool_880)
    req = EngineRequest(
        title="t", subject=SubjectType.MATH_2, mode=PaperMode.BUNDLE_3_PAPERS,
        target_chapters=[], seed=777, candidate_question_pool=pool_880,
    )
    bundle = eng.generate_bundle(req, bundle_size=3)
    all_ids = [q.id for p in bundle.papers for q in p.questions]
    assert len(all_ids) == len(set(all_ids)), "三卷之间出现了重复题"
    chapters = {q.chapter for p in bundle.papers for q in p.questions}
    assert len(chapters) >= 8, f"章节覆盖退化到 {len(chapters)} 章"
