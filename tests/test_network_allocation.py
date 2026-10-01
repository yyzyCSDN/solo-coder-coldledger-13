import unittest
from datetime import date
from decimal import Decimal

from coldledger.transfer.network_allocation import (
    Lot, SitePosition, Lane, ReceivingWindow, Weights, Plan,
    plan, explain,
    REASON_NO_LANE, REASON_LANE_DISABLED, REASON_SKU_NOT_QUALIFIED,
    REASON_SHELF_LIFE, REASON_SOURCE_DRY, REASON_SAFETY_LOCKED,
    REASON_LANE_CAPACITY, REASON_NO_WINDOW, REASON_WINDOW_CAPACITY,
    REASON_SATISFIED,
)
from coldledger.transfer.network_commit import (
    StockRecord, LaneRecord, WindowRecord, StockBook, LaneBook, WindowBook,
    TransferCommitter, resolve_lane,
)

D = Decimal
TODAY = date(2026, 10, 1)


class NetworkPlanTests(unittest.TestCase):
    def base_world(self, *, window_cap="40", lane_cap="100"):
        positions = [
            SitePosition("A", "X", on_hand=D("200"), safety_stock=D("20"),
                         forecast_demand=D("30")),
            SitePosition("B", "X", on_hand=D("10"), safety_stock=D("50"),
                         forecast_demand=D("60"), velocity_per_day=D("1"),
                         min_inbound_days=30),
            SitePosition("C", "X", on_hand=D("100"), safety_stock=D("30"),
                         forecast_demand=D("10")),
        ]
        lots = {
            ("A", "X"): [Lot("A1", "X", D("150"), date(2026, 11, 15)),
                         Lot("A2", "X", D("50"), date(2027, 6, 1))],
            ("B", "X"): [Lot("B1", "X", D("10"), date(2027, 1, 1))],
            ("C", "X"): [Lot("C1", "X", D("70"), date(2027, 3, 1))],
        }
        lanes = [
            Lane("A", "B", D("2"), lead_days=2, capacity=D(lane_cap)),
            Lane("C", "B", D("3"), lead_days=5, capacity=D(lane_cap)),
        ]
        windows = [ReceivingWindow("B", date(2026, 10, 1), date(2026, 10, 31),
                                   D(window_cap))]
        return positions, lots, lanes, windows

    def test_plan_is_global_not_gap_ordered_and_respects_receiving_capacity(self):
        # Window takes only 40 while the gap is 100: planning must discover
        # this up front, move exactly 40, and leave a classified residual.
        positions, lots, lanes, windows = self.base_world()
        result = plan(positions, lots, lanes, windows, plan_date=TODAY)

        total = sum((m.quantity for m in result.moves), D("0"))
        self.assertEqual(total, D("40"))
        self.assertEqual(result.inbound("B", "X"), D("40"))
        gap = {(g.target, g.sku): g for g in result.residual_gaps}
        self.assertEqual(gap[("B", "X")].quantity, D("60"))
        # Both candidate lanes keep their reason for the unserved rest.
        reasons = {(r.source, r.reason) for r in result.rejections
                   if r.shortfall > 0}
        self.assertIn(("A", REASON_WINDOW_CAPACITY), reasons)
        self.assertIn(("C", REASON_WINDOW_CAPACITY), reasons)

    def test_cheaper_shorter_lane_wins_among_feasible_candidates(self):
        positions, lots, lanes, windows = self.base_world(
            window_cap="100", lane_cap="200")
        result = plan(positions, lots, lanes, windows, plan_date=TODAY)
        # A: cheaper/faster lane, near-expiry FEFO lot, ample headroom.
        self.assertEqual(len(result.moves), 1)
        move = result.moves[0]
        self.assertEqual((move.source, move.target, move.lot), ("A", "B", "A1"))
        self.assertEqual(move.quantity, D("100"))
        self.assertAlmostEqual(move.factors["lane_cost"], 1 / 3)
        self.assertGreater(move.factors["expiry"], 0.7)

    def test_safety_stock_caps_source_even_when_gap_is_larger(self):
        # C has 100 on hand / 30 safety / 10 forecast => only 60 releasable.
        positions = [
            SitePosition("C", "X", on_hand=D("100"), safety_stock=D("30"),
                         forecast_demand=D("10")),
            SitePosition("D", "X", on_hand=D("0"), safety_stock=D("80"),
                         forecast_demand=D("0"), min_inbound_days=10),
        ]
        lots = {("C", "X"): [Lot("C1", "X", D("100"), date(2027, 1, 1))]}
        lanes = [Lane("C", "D", D("1"), lead_days=1, capacity=D("500"))]
        windows = [ReceivingWindow("D", date(2026, 10, 1),
                                   date(2026, 12, 31), D("500"))]
        result = plan(positions, lots, lanes, windows, plan_date=TODAY)
        self.assertEqual(result.inbound("D", "X"), D("60"))
        self.assertEqual(result.residual_gaps[0].quantity, D("20"))
        self.assertEqual(result.residual_gaps[0].dominant_reason,
                         REASON_SAFETY_LOCKED)

    def test_shelf_life_rejection_is_lot_specific(self):
        positions = [
            SitePosition("A", "X", on_hand=D("100"), safety_stock=D("0")),
            SitePosition("B", "X", on_hand=D("0"), safety_stock=D("10"),
                         min_inbound_days=60),
        ]
        # Arrival 2026-10-03: A1 has 43 days of shelf life (< 60), A2 passes.
        lots = {("A", "X"): [Lot("A1", "X", D("50"), date(2026, 11, 15)),
                             Lot("A2", "X", D("50"), date(2027, 6, 1))]}
        lanes = [Lane("A", "B", D("1"), lead_days=2, capacity=D("100"))]
        windows = [ReceivingWindow("B", date(2026, 10, 1),
                                   date(2026, 10, 31), D("100"))]
        result = plan(positions, lots, lanes, windows, plan_date=TODAY)
        self.assertEqual([m.lot for m in result.moves], ["A2"])
        shelf = [r for r in result.rejections if r.reason == REASON_SHELF_LIFE]
        self.assertTrue(any(r.lot == "A1" for r in shelf))
        self.assertTrue(all("A1" in r.detail for r in shelf if r.lot == "A1"))

    def test_expiry_exposure_prefers_pushing_near_expiry_stock(self):
        # Both lanes feasible and identical cost: the source holding a
        # near-expiry lot should win over the source with long-dated stock.
        positions = [
            SitePosition("A", "X", on_hand=D("100"), safety_stock=D("0")),
            SitePosition("C", "X", on_hand=D("100"), safety_stock=D("0")),
            SitePosition("B", "X", on_hand=D("0"), safety_stock=D("50"),
                         velocity_per_day=D("1")),
        ]
        lots = {
            ("A", "X"): [Lot("A1", "X", D("100"), date(2026, 10, 20))],
            ("C", "X"): [Lot("C1", "X", D("100"), date(2028, 1, 1))],
        }
        lanes = [
            Lane("A", "B", D("1"), lead_days=2, capacity=D("100")),
            Lane("C", "B", D("1"), lead_days=2, capacity=D("100")),
        ]
        windows = [ReceivingWindow("B", date(2026, 10, 1),
                                   date(2026, 12, 31), D("100"))]
        result = plan(positions, lots, lanes, windows, plan_date=TODAY)
        self.assertEqual(result.moves[0].source, "A")
        self.assertGreater(result.moves[0].factors["expiry"], 0.9)

    def test_lane_capacity_is_shared_across_moves(self):
        positions = [
            SitePosition("A", "X", on_hand=D("200"), safety_stock=D("0")),
            SitePosition("B", "X", on_hand=D("0"), safety_stock=D("30")),
            SitePosition("E", "X", on_hand=D("0"), safety_stock=D("30")),
        ]
        lots = {("A", "X"): [Lot("A1", "X", D("200"), date(2027, 1, 1))]}
        lanes = [
            Lane("A", "B", D("1"), lead_days=1, capacity=D("40")),
            Lane("A", "E", D("1"), lead_days=1, capacity=D("40")),
        ]
        windows = [
            ReceivingWindow("B", date(2026, 10, 1), date(2026, 12, 31), D("100")),
            ReceivingWindow("E", date(2026, 10, 1), date(2026, 12, 31), D("10")),
        ]
        result = plan(positions, lots, lanes, windows, plan_date=TODAY)
        # E can only receive 10; B gets 30 through its own lane.
        self.assertEqual(result.inbound("B", "X"), D("30"))
        self.assertEqual(result.inbound("E", "X"), D("10"))
        e_gap = [g for g in result.residual_gaps if g.target == "E"][0]
        self.assertEqual(e_gap.dominant_reason, REASON_WINDOW_CAPACITY)

    def test_rejections_cover_disabled_unqualified_and_missing_lanes(self):
        positions = [
            SitePosition("A", "X", on_hand=D("100"), safety_stock=D("0")),
            SitePosition("C", "Y", on_hand=D("100"), safety_stock=D("0")),
            SitePosition("S", "X", on_hand=D("100"), safety_stock=D("0")),
            SitePosition("B", "X", on_hand=D("0"), safety_stock=D("10")),
        ]
        lots = {
            ("A", "X"): [Lot("A1", "X", D("100"), date(2027, 1, 1))],
            ("C", "Y"): [Lot("C1", "Y", D("100"), date(2027, 1, 1))],
            ("S", "X"): [Lot("S1", "X", D("100"), date(2027, 1, 1))],
        }
        lanes = [
            Lane("A", "B", D("1"), lead_days=1, capacity=D("100"), enabled=False),
            Lane("C", "B", D("1"), lead_days=1, capacity=D("100"), sku="Y"),
        ]
        windows = [ReceivingWindow("B", date(2026, 10, 1),
                                   date(2026, 12, 31), D("100"))]
        result = plan(positions, lots, lanes, windows, plan_date=TODAY)
        codes = {(r.source, r.reason) for r in result.rejections}
        self.assertIn(("A", REASON_LANE_DISABLED), codes)
        self.assertIn(("C", REASON_SKU_NOT_QUALIFIED), codes)
        self.assertIn(("S", REASON_NO_LANE), codes)
        self.assertEqual(result.moves, ())
        self.assertEqual(result.residual_gaps[0].dominant_reason,
                         REASON_LANE_DISABLED)

    def test_no_window_on_arrival_is_recorded(self):
        positions = [
            SitePosition("A", "X", on_hand=D("100"), safety_stock=D("0")),
            SitePosition("B", "X", on_hand=D("0"), safety_stock=D("10")),
        ]
        lots = {("A", "X"): [Lot("A1", "X", D("100"), date(2027, 1, 1))]}
        lanes = [Lane("A", "B", D("1"), lead_days=10, capacity=D("100"))]
        windows = [ReceivingWindow("B", date(2026, 10, 1),
                                   date(2026, 10, 5), D("100"))]
        result = plan(positions, lots, lanes, windows, plan_date=TODAY)
        self.assertEqual(result.moves, ())
        self.assertEqual(result.residual_gaps[0].dominant_reason, REASON_NO_WINDOW)

    def test_plan_is_deterministic(self):
        positions, lots, lanes, windows = self.base_world(window_cap="100")
        first = plan(positions, lots, lanes, windows, plan_date=TODAY)
        second = plan(positions, lots, lanes, windows, plan_date=TODAY)
        self.assertEqual(
            [(m.source, m.target, m.lot, str(m.quantity)) for m in first.moves],
            [(m.source, m.target, m.lot, str(m.quantity)) for m in second.moves],
        )

    def test_explain_lines_cover_moves_rejections_and_gaps(self):
        positions, lots, lanes, windows = self.base_world()
        lines = explain(plan(positions, lots, lanes, windows, plan_date=TODAY))
        self.assertTrue(any(l.startswith("MOVE") for l in lines))
        self.assertTrue(any(l.startswith("REJECT") for l in lines))
        self.assertTrue(any(l.startswith("GAP") for l in lines))


class AtomicCommitTests(unittest.TestCase):
    def _plan_and_books(self):
        positions = [
            SitePosition("A", "X", on_hand=D("100"), safety_stock=D("0")),
            SitePosition("B", "X", on_hand=D("0"), safety_stock=D("40")),
        ]
        lots = {("A", "X"): [Lot("A1", "X", D("100"), date(2027, 1, 1))]}
        lanes = [Lane("A", "B", D("1"), lead_days=2, capacity=D("100"))]
        windows = [ReceivingWindow("B", date(2026, 10, 1),
                                   date(2026, 10, 31), D("100"))]
        result = plan(positions, lots, lanes, windows, plan_date=TODAY)
        stock = StockBook([
            StockRecord("A", "X", D("100"), version=0),
            StockRecord("B", "X", D("0"), version=0),
        ])
        lane_book = LaneBook([
            LaneRecord("A", "B", D("100"), sku=None, version=0),
        ])
        window_book = WindowBook([
            WindowRecord("W0", "B", D("100"), version=0),
        ])
        return result, stock, lane_book, window_book

    def test_commit_applies_stock_lane_and_window_together(self):
        planned, stock, lane_book, window_book = self._plan_and_books()
        committer = TransferCommitter(stock, lane_book, window_book)
        snap = committer.snapshot(planned, window_ids={0: "W0"})
        receipt = committer.commit(planned, "PLAN-1", snap, window_ids={0: "W0"})

        self.assertEqual(receipt.reference, "PLAN-1")
        self.assertEqual(stock.rows[("A", "X")].available, D("60"))
        self.assertEqual(stock.rows[("B", "X")].available, D("40"))
        self.assertEqual(stock.rows[("A", "X")].version, 1)
        self.assertEqual(lane_book.rows[("A", "B", None)].available, D("60"))
        self.assertEqual(window_book.rows["W0"].available, D("60"))
        self.assertEqual(len(receipt.moves), 1)

    def test_version_conflict_aborts_the_whole_commit(self):
        planned, stock, lane_book, window_book = self._plan_and_books()
        committer = TransferCommitter(stock, lane_book, window_book)
        snap = committer.snapshot(planned, window_ids={0: "W0"})
        # Someone else bumps the source row first.
        stock.rows[("A", "X")] = StockRecord("A", "X", D("100"), version=1)
        with self.assertRaises(RuntimeError):
            committer.commit(planned, "PLAN-2", snap, window_ids={0: "W0"})
        # Nothing else moved either.
        self.assertEqual(window_book.rows["W0"].available, D("100"))
        self.assertEqual(lane_book.rows[("A", "B", None)].available, D("100"))
        self.assertNotIn("PLAN-2", committer.committed_references())

    def test_receiving_overrun_aborts_before_any_mutation(self):
        planned, stock, lane_book, window_book = self._plan_and_books()
        # Window capacity shrank to 10 after planning: all-or-nothing.
        window_book.rows["W0"] = WindowRecord("W0", "B", D("10"), version=0)
        committer = TransferCommitter(stock, lane_book, window_book)
        snap = committer.snapshot(planned, window_ids={0: "W0"})
        with self.assertRaises(ValueError):
            committer.commit(planned, "PLAN-3", snap, window_ids={0: "W0"})
        self.assertEqual(stock.rows[("A", "X")].available, D("100"))
        self.assertEqual(stock.rows[("B", "X")].available, D("0"))
        self.assertEqual(lane_book.rows[("A", "B", None)].available, D("100"))

    def test_duplicate_commit_reference_is_rejected(self):
        planned, stock, lane_book, window_book = self._plan_and_books()
        committer = TransferCommitter(stock, lane_book, window_book)
        committer.commit(planned, "ONCE", window_ids={0: "W0"})
        with self.assertRaises(ValueError):
            committer.commit(planned, "ONCE", window_ids={0: "W0"})

    def test_window_site_mismatch_aborts(self):
        planned, stock, lane_book, window_book = self._plan_and_books()
        window_book.rows["W0"] = WindowRecord("W0", "OTHER", D("100"), version=0)
        committer = TransferCommitter(stock, lane_book, window_book)
        with self.assertRaises(ValueError):
            committer.commit(planned, "PLAN-4", window_ids={0: "W0"})

    def test_resolve_lane_prefers_sku_specific_record(self):
        book = LaneBook([
            LaneRecord("A", "B", D("5"), sku="X", version=0),
            LaneRecord("A", "B", D("500"), sku=None, version=0),
        ])
        self.assertEqual(resolve_lane(book, "A", "B", "X").sku, "X")
        self.assertIsNone(resolve_lane(book, "A", "B", "Z").sku)


class EndToEndServiceTests(unittest.TestCase):
    def test_plan_then_atomic_commit_through_service(self):
        from coldledger.api import ColdLedgerService
        svc = ColdLedgerService()
        positions = [
            SitePosition("A", "X", on_hand=D("80"), safety_stock=D("10")),
            SitePosition("B", "X", on_hand=D("0"), safety_stock=D("30"),
                         min_inbound_days=7),
        ]
        lots = {("A", "X"): [Lot("A1", "X", D("80"), date(2027, 1, 1))]}
        lanes = [Lane("A", "B", D("1"), lead_days=1, capacity=D("100"))]
        windows = [ReceivingWindow("B", date(2026, 10, 1),
                                   date(2026, 10, 15), D("100"))]
        planned = svc.plan_transfers(positions, lots, lanes, windows,
                                     plan_date=TODAY)
        self.assertEqual(planned.inbound("B", "X"), D("30"))

        stock = StockBook([StockRecord("A", "X", D("80")),
                           StockRecord("B", "X", D("0"))])
        lane_book = LaneBook([LaneRecord("A", "B", D("100"), sku=None)])
        window_book = WindowBook([WindowRecord("RCV-1", "B", D("100"))])
        committer = svc.transfer_committer(stock, lane_book, window_book)
        receipt = committer.commit(planned, "E2E-1", window_ids={0: "RCV-1"})
        self.assertEqual(stock.rows[("A", "X")].available, D("50"))
        self.assertEqual(stock.rows[("B", "X")].available, D("30"))
        self.assertEqual(len(receipt.moves), 1)

    def test_concurrent_disjoint_commits_both_land(self):
        import threading
        common = dict(plan_date=TODAY)
        plan_x = plan(
            [SitePosition("A", "X", on_hand=D("100"), safety_stock=D("0")),
             SitePosition("D", "X", on_hand=D("0"), safety_stock=D("10"))],
            {("A", "X"): [Lot("A1", "X", D("100"), date(2027, 1, 1))]},
            [Lane("A", "D", D("1"), lead_days=1, capacity=D("100"))],
            [ReceivingWindow("D", date(2026, 10, 1), date(2026, 12, 31), D("100"))],
            **common,
        )
        plan_y = plan(
            [SitePosition("B", "Y", on_hand=D("100"), safety_stock=D("0")),
             SitePosition("E", "Y", on_hand=D("0"), safety_stock=D("10"))],
            {("B", "Y"): [Lot("B1", "Y", D("100"), date(2027, 1, 1))]},
            [Lane("B", "E", D("1"), lead_days=1, capacity=D("100"))],
            [ReceivingWindow("E", date(2026, 10, 1), date(2026, 12, 31), D("100"))],
            **common,
        )
        stock = StockBook([
            StockRecord("A", "X", D("100")), StockRecord("B", "Y", D("100")),
            StockRecord("D", "X", D("0")), StockRecord("E", "Y", D("0")),
        ])
        lane_book = LaneBook([
            LaneRecord("A", "D", D("100"), sku=None),
            LaneRecord("B", "E", D("100"), sku=None),
        ])
        window_book = WindowBook([
            WindowRecord("WD", "D", D("100")),
            WindowRecord("WE", "E", D("100")),
        ])
        committer = TransferCommitter(stock, lane_book, window_book)
        errors: list[BaseException] = []

        def commit(planned, ref, ids):
            try:
                committer.commit(planned, ref, window_ids=ids)
            except BaseException as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [
            threading.Thread(target=commit, args=(plan_x, "CX", {0: "WD"})),
            threading.Thread(target=commit, args=(plan_y, "CY", {0: "WE"})),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
            self.assertFalse(t.is_alive(), "deadlock during concurrent commit")
        self.assertEqual(errors, [])
        self.assertEqual(stock.rows[("D", "X")].available, D("10"))
        self.assertEqual(stock.rows[("E", "Y")].available, D("10"))
        self.assertEqual(stock.rows[("A", "X")].available, D("90"))
        self.assertEqual(stock.rows[("B", "Y")].available, D("90"))


if __name__ == "__main__":
    unittest.main()
