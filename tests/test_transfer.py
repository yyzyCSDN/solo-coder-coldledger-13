import unittest
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

from coldledger.transfer import (
    CapacityBook,
    CapacityRow,
    CommitAborted,
    CommitCoordinator,
    Lane,
    ReceivingWindow,
    RejectReason,
    SiteNeed,
    StockLot,
    TransferPlan,
    build_plan,
    rejection_summary,
)
from coldledger.inventory.ledger import InventoryLedger

SHIP = date(2026, 10, 1)


def _win(site, day_offset, qty):
    return ReceivingWindow(site, SHIP + timedelta(days=day_offset), Decimal(str(qty)))


class GlobalPlanTests(unittest.TestCase):
    def test_global_flow_avoids_greedy_gap_sorting(self):
        # A 只能经到货日 day1 窗口接 4；缺口 8 必须联合 C 站补足，而不是把 A 塞爆后才发现接不下
        lots = [
            StockLot("A", "S1", "NEAR", date(2026, 11, 15), Decimal("5")),
            StockLot("A", "S1", "FRESH", date(2027, 6, 1), Decimal("10")),
            StockLot("C", "S1", "CLOT", date(2027, 3, 1), Decimal("20")),
        ]
        needs = [SiteNeed("B", "S1", Decimal("8"), priority=2)]
        lanes = [
            Lane("AB", "A", "B", Decimal("2"), 1, Decimal("6"), risk=0.1),
            Lane("CB", "C", "B", Decimal("1"), 2, Decimal("20")),
        ]
        windows = [_win("B", 1, 4), _win("B", 2, 20)]
        plan = build_plan("P1", lots, needs, lanes, windows,
                          site_safety_stock={("A", "S1"): Decimal("3")},
                          ship_day=SHIP, min_remaining_days=20)
        by_lane = {m.lane_id: m.quantity for m in plan.movements}
        self.assertEqual(by_lane["AB"], Decimal("4"))
        self.assertEqual(by_lane["CB"], Decimal("4"))
        self.assertTrue(plan.is_complete())
        # A 站窗口只过 4，绝不能超过目的站收货容量
        arrivals = [m.arrival_day for m in plan.movements if m.target == "B"]
        day1 = sum(m.quantity for m in plan.movements
                   if m.target == "B" and m.arrival_day == SHIP + timedelta(days=1))
        self.assertLessEqual(day1, Decimal("4"))

    def test_expiry_exposure_prefers_near_expiry_lots(self):
        lots = [
            StockLot("A", "S1", "NEAR", date(2026, 11, 10), Decimal("6")),
            StockLot("A", "S1", "FRESH", date(2027, 8, 1), Decimal("6")),
        ]
        needs = [SiteNeed("B", "S1", Decimal("4"))]
        lanes = [Lane("AB", "A", "B", Decimal("2"), 1, Decimal("20"))]
        plan = build_plan("P2", lots, needs, lanes, [_win("B", 1, 20)],
                          ship_day=SHIP, min_remaining_days=20)
        picks = [p for m in plan.movements for p in m.picks]
        self.assertEqual([p.lot for p in picks], ["NEAR"])

    def test_safety_stock_is_hard_constraint_with_trail(self):
        lots = [StockLot("A", "S1", "L1", date(2027, 5, 1), Decimal("10"))]
        needs = [SiteNeed("B", "S1", Decimal("6"))]
        lanes = [Lane("AB", "A", "B", Decimal("2"), 1, Decimal("20"))]
        plan = build_plan("P3", lots, needs, lanes, [_win("B", 1, 20)],
                          site_safety_stock={("A", "S1"): Decimal("10")}, ship_day=SHIP)
        self.assertEqual(plan.movements, ())
        self.assertEqual(len(plan.unmet), 1)
        reasons = {r.reason for r in plan.rejections}
        self.assertIn(RejectReason.SOURCE_SAFETY_STOCK, reasons)

    def test_shelf_life_rejected_with_remaining_days(self):
        lots = [StockLot("A", "S1", "SHORT", date(2026, 10, 20), Decimal("10"))]
        needs = [SiteNeed("B", "S1", Decimal("4"))]
        lanes = [Lane("AB", "A", "B", Decimal("2"), 1, Decimal("20"))]
        plan = build_plan("P4", lots, needs, lanes, [_win("B", 1, 20)],
                          ship_day=SHIP, min_remaining_days=30)
        self.assertFalse(plan.is_complete())
        self.assertTrue(any(r.reason == RejectReason.SHELF_LIFE for r in plan.rejections))
        self.assertIn("18d", plan.rejections[0].detail)

    def test_receiving_capacity_full_blocks_destination(self):
        lots = [StockLot("A", "S1", "L1", date(2027, 5, 1), Decimal("10"))]
        needs = [SiteNeed("B", "S1", Decimal("6"))]
        lanes = [Lane("AB", "A", "B", Decimal("2"), 1, Decimal("20"))]
        plan = build_plan("P5", lots, needs, lanes, [_win("B", 1, 0)], ship_day=SHIP)
        self.assertFalse(plan.is_complete())
        self.assertIn(RejectReason.NO_WINDOW, {r.reason for r in plan.rejections})
        # 不应再出现误导性的 no_lane
        self.assertNotIn(RejectReason.NO_LANE, {r.reason for r in plan.rejections})

    def test_no_lane_recorded(self):
        lots = [StockLot("Z", "S1", "L1", date(2027, 5, 1), Decimal("5"))]
        needs = [SiteNeed("B", "S1", Decimal("3"))]
        lanes = [Lane("AB", "A", "B", Decimal("2"), 1, Decimal("20"))]
        plan = build_plan("P6", lots, needs, lanes, [_win("B", 1, 20)], ship_day=SHIP)
        self.assertTrue(any(r.reason == RejectReason.NO_LANE and r.source == "Z"
                            for r in plan.rejections))

    def test_alternative_selection_reason_for_unused_cheaper_candidate(self):
        lots = [
            StockLot("A", "S1", "NEAR", date(2026, 11, 15), Decimal("5")),
            StockLot("A", "S1", "FRESH", date(2027, 6, 1), Decimal("10")),
        ]
        needs = [SiteNeed("B", "S1", Decimal("4"))]
        lanes = [Lane("AB", "A", "B", Decimal("2"), 1, Decimal("20"))]
        plan = build_plan("P7", lots, needs, lanes, [_win("B", 1, 20)],
                          ship_day=SHIP, min_remaining_days=20)
        reasons = {(r.lot, r.reason) for r in plan.rejections}
        self.assertIn(("FRESH", RejectReason.ALTERNATIVE_SELECTED), reasons)

    def test_multi_sku_waves_consume_shared_window_capacity(self):
        lots = [
            StockLot("A", "HOT", "L1", date(2027, 5, 1), Decimal("10")),
            StockLot("A", "COLD", "L2", date(2027, 5, 1), Decimal("10")),
        ]
        needs = [
            SiteNeed("B", "HOT", Decimal("10"), priority=5),
            SiteNeed("B", "COLD", Decimal("6"), priority=1),
        ]
        lanes = [Lane("AB", "A", "B", Decimal("2"), 1, Decimal("100"))]
        plan = build_plan("P8", lots, needs, lanes, [_win("B", 1, 10)], ship_day=SHIP)
        skus = {m.sku: m.quantity for m in plan.movements}
        self.assertEqual(skus.get("HOT"), Decimal("10"))
        self.assertNotIn("COLD", skus)
        unmet_sku = {u.sku for u in plan.unmet}
        self.assertIn("COLD", unmet_sku)
        self.assertTrue(any(r.sku == "COLD" and r.reason == RejectReason.NO_WINDOW
                            for r in plan.rejections))

    def test_rejection_summary(self):
        lots = [StockLot("A", "S1", "L1", date(2026, 10, 20), Decimal("10"))]
        needs = [SiteNeed("B", "S1", Decimal("4"))]
        lanes = [Lane("AB", "A", "B", Decimal("2"), 1, Decimal("20"))]
        plan = build_plan("P9", lots, needs, lanes, [_win("B", 1, 20)],
                          ship_day=SHIP, min_remaining_days=30)
        summary = rejection_summary(plan)
        self.assertGreaterEqual(summary["shelf_life"], 1)


class AtomicCommitTests(unittest.TestCase):
    def _books(self, lane_cap="10", win_cap="6"):
        lane = CapacityBook([CapacityRow("AB", Decimal(lane_cap))])
        window = CapacityBook([CapacityRow(f"B@{(SHIP + timedelta(days=1)).isoformat()}",
                                           Decimal(win_cap))])
        return lane, window

    def _plan(self, pid="C1"):
        lots = [StockLot("A", "S1", "L1", date(2027, 5, 1), Decimal("10"))]
        needs = [SiteNeed("B", "S1", Decimal("6"))]
        lanes = [Lane("AB", "A", "B", Decimal("2"), 1, Decimal("10"))]
        return build_plan(pid, lots, needs, lanes, [_win("B", 1, 6)], ship_day=SHIP)

    def test_commit_reserves_all_three_resources(self):
        ledger = InventoryLedger()
        ledger.post("A", "seed", Decimal("10"), "seed")
        lane, window = self._books()
        coord = CommitCoordinator(lane, window, ledger)
        result = coord.commit(self._plan())
        self.assertEqual(result.movement_ids, ("C1:M001",))
        self.assertEqual(ledger.balance("A"), Decimal("4"))
        self.assertEqual(ledger.balance("B"), Decimal("6"))
        self.assertEqual(lane.rows["AB"].available, Decimal("4"))
        self.assertEqual(window.rows[list(window.rows)[0]].available, Decimal("0"))

    def test_commit_is_idempotent(self):
        ledger = InventoryLedger()
        ledger.post("A", "seed", Decimal("10"), "seed")
        coord = CommitCoordinator(*self._books(), ledger)
        coord.commit(self._plan())
        with self.assertRaises(RuntimeError):
            coord.commit(self._plan())

    def test_prepare_failure_changes_nothing(self):
        # 线路与窗口容量都不足以再做一次相同计划
        ledger = InventoryLedger()
        ledger.post("A", "seed", Decimal("10"), "seed")
        lane, window = self._books()
        coord = CommitCoordinator(lane, window, ledger)
        coord.commit(self._plan())
        before_a = ledger.balance("A")
        with self.assertRaises(CommitAborted) as ctx:
            coord.commit(self._plan("C2"))
        resources = {f.resource for f in ctx.exception.failures}
        self.assertIn("lane", resources)
        self.assertIn("window", resources)
        # 库存无任何变化（窗口/线路也没有被半扣）
        self.assertEqual(ledger.balance("A"), before_a)
        self.assertEqual(lane.rows["AB"].available, Decimal("4"))
        self.assertEqual(window.rows[list(window.rows)[0]].available, Decimal("0"))

    def test_unmet_plan_cannot_commit(self):
        ledger = InventoryLedger()
        ledger.post("A", "seed", Decimal("10"), "seed")
        coord = CommitCoordinator(*self._books(), ledger)
        lots = [StockLot("A", "S1", "L1", date(2026, 10, 20), Decimal("10"))]
        needs = [SiteNeed("B", "S1", Decimal("4"))]
        lanes = [Lane("AB", "A", "B", Decimal("2"), 1, Decimal("10"))]
        bad = build_plan("C3", lots, needs, lanes, [_win("B", 1, 6)],
                         ship_day=SHIP, min_remaining_days=30)
        with self.assertRaises(CommitAborted):
            coord.commit(bad)
        self.assertEqual(ledger.balance("A"), Decimal("10"))

    def test_partial_commit_failures_are_compensated(self):
        # 库存账本被篡改到余额不足：线路/窗口已扣后库存失败，必须全部回滚
        ledger = InventoryLedger()
        ledger.post("A", "seed", Decimal("10"), "seed")
        lane, window = self._books()
        coord = CommitCoordinator(lane, window, ledger)
        plan = self._plan("C4")
        # 让 prepare 通过、commit 阶段库存失败：prepare 后、commit 前挪用库存
        orig = CommitCoordinator._commit

        def sabotage(self2, p, actor):
            ledger.post("A", "drain", Decimal("-6"), "drain") if ledger.balance("A") >= 6 else None
            return orig(self2, p, actor)

        CommitCoordinator._commit = sabotage
        try:
            with self.assertRaises(CommitAborted):
                coord.commit(plan)
        finally:
            CommitCoordinator._commit = orig
        # 补偿：线路与窗口容量恢复，A 余额保留被挪用后的现状（回滚只撤销本次提交造成的占用）
        self.assertEqual(lane.rows["AB"].available, Decimal("10"))
        self.assertEqual(window.rows[list(window.rows)[0]].available, Decimal("6"))


if __name__ == "__main__":
    unittest.main()
