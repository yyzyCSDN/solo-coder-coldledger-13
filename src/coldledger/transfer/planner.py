from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Iterable

from .model import (
    Lane,
    LotPick,
    Movement,
    ReceivingWindow,
    Rejection,
    RejectReason,
    SiteNeed,
    StockLot,
    TransferPlan,
    UnmetGap,
    Weights,
)

# 费用全部放大 SCALE 倍后取整，避免浮点。效期收益为负成本。
SCALE = 10_000
EPS = Decimal("0.000001")


def _q(value: Decimal | float) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def _cost_int(value: Decimal) -> int:
    return int((value * SCALE).to_integral_value())


def _cap_int(value: Decimal) -> int:
    return max(0, int((value * SCALE).to_integral_value()))


@dataclass
class _Edge:
    u: int
    v: int
    cap: int
    cost: int
    flow: int = 0


class _MCMF:
    """连续最短路（SPFA）最小费用流，支持负费用边；本场景网络为 DAG，无负环。"""

    def __init__(self) -> None:
        self.edges: list[_Edge] = []
        self.adj: dict[int, list[int]] = {}

    def node(self) -> int:
        i = len(self.adj)
        self.adj[i] = []
        return i

    def add(self, u: int, v: int, cap: int, cost: int) -> int:
        idx = len(self.edges)
        self.edges.append(_Edge(u, v, cap, cost))
        self.edges.append(_Edge(v, u, 0, -cost))
        self.adj[u].append(idx)
        self.adj[v].append(idx + 1)
        return idx

    def run(self, source: int, sink: int, required: int) -> tuple[int, int]:
        sent = 0
        total_cost = 0
        n = len(self.adj)
        while sent < required:
            dist: list[int | None] = [None] * n
            prev_edge: list[int | None] = [None] * n
            in_queue = [False] * n
            dist[source] = 0
            queue = [source]
            in_queue[source] = True
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                in_queue[u] = False
                du = dist[u]
                for ei in self.adj[u]:
                    e = self.edges[ei]
                    if e.cap - e.flow <= 0:
                        continue
                    nd = du + e.cost
                    if dist[e.v] is None or nd < dist[e.v]:
                        dist[e.v] = nd
                        prev_edge[e.v] = ei
                        if not in_queue[e.v]:
                            queue.append(e.v)
                            in_queue[e.v] = True
            if dist[sink] is None:
                break
            add = required - sent
            cur = sink
            while cur != source:
                ei = prev_edge[cur]
                e = self.edges[ei]
                add = min(add, e.cap - e.flow)
                cur = e.u
            cur = sink
            while cur != source:
                ei = prev_edge[cur]
                self.edges[ei].flow += add
                self.edges[ei ^ 1].flow -= add
                cur = self.edges[ei].u
            sent += add
            total_cost += add * dist[sink]
        return sent, total_cost


@dataclass(frozen=True)
class _LaneContext:
    lane: Lane
    arrival: date
    window_left: Decimal
    lane_left: Decimal


@dataclass
class _WaveResult:
    movements: list[Movement]
    rejections: list[Rejection]
    unmet: list[UnmetGap]
    lane_usage: dict[str, Decimal]
    window_usage: dict[tuple[str, date], Decimal]
    score: Decimal


def _expiry_fraction(expiry: date, ship_day: date, weights: Weights) -> Decimal:
    """效期暴露：距效期越近越该优先调出；0=新鲜，1=已到/过效期。"""
    days = Decimal((expiry - ship_day).days)
    if days <= 0:
        return Decimal("1")
    h = Decimal(weights.expiry_horizon_days)
    return max(Decimal("0"), min(Decimal("1"), Decimal("1") - days / h))


def _priority_factor(priority: int, weights: Weights) -> Decimal:
    return max(Decimal("0"), Decimal("1") - weights.priority_step * Decimal(priority))


def _solve_wave(
    sku: str,
    plan_id: str,
    ship_day: date,
    min_remaining_days: int,
    lots: list[StockLot],
    needs: list[SiteNeed],
    lanes: list[Lane],
    window_residual: dict[tuple[str, date], Decimal],
    lane_residual: dict[str, Decimal],
    weights: Weights,
) -> _WaveResult:
    rejections: list[Rejection] = []
    lane_usage: dict[str, Decimal] = {}
    window_usage: dict[tuple[str, date], Decimal] = {}

    need_by_site = {n.site: n for n in needs}
    lots_by_site: dict[str, list[StockLot]] = {}
    for lot in lots:
        lots_by_site.setdefault(lot.site, []).append(lot)

    # 目的站 -> 可用线路上下文（到货日落在某个有容量的收货窗口内）
    lane_ctxs: dict[str, list[_LaneContext]] = {}
    for lane in lanes:
        if not lane.enabled or lane.target not in need_by_site:
            if lane.enabled and lane.target not in need_by_site:
                for lot in lots_by_site.get(lane.source, []):
                    rejections.append(Rejection(sku, lane.source, lane.target, lot.lot, lane.lane_id,
                                                RejectReason.NO_DEMAND, "destination has no gap for sku"))
            continue
        arrival = ship_day + timedelta(days=lane.lead_days)
        w_left = window_residual.get((lane.target, arrival), Decimal("0"))
        l_left = lane_residual.get(lane.lane_id, lane.capacity)
        if w_left <= 0:
            for lot in lots_by_site.get(lane.source, []):
                rejections.append(Rejection(sku, lane.source, lane.target, lot.lot, lane.lane_id,
                                            RejectReason.NO_WINDOW,
                                            f"no receiving capacity on {arrival.isoformat()}"))
            continue
        if l_left <= 0:
            for lot in lots_by_site.get(lane.source, []):
                rejections.append(Rejection(sku, lane.source, lane.target, lot.lot, lane.lane_id,
                                            RejectReason.LANE_CAPACITY,
                                            "lane exhausted by higher priority wave"))
            continue
        lane_ctxs.setdefault(lane.target, []).append(_LaneContext(lane, arrival, w_left, l_left))

    net = _MCMF()
    source = net.node()
    sink = net.node()

    total_gap = sum((n.gap for n in needs), Decimal("0"))

    # 需求节点 + 缺口未满足支路（缺货惩罚，按站点优先级加权）
    demand_nodes: dict[str, int] = {}
    lot_edge_meta: list[dict] = []
    for n in needs:
        dn = net.node()
        penalty = weights.shortage * _priority_factor(n.priority, weights)
        net.add(source, dn, _cap_int(n.gap), _cost_int(penalty))
        net.add(dn, sink, _cap_int(n.gap), 0)
        demand_nodes[n.site] = dn

    # 共享容量节点：批次 -> 线路(共享 lane capacity) -> 窗口(共享 receiving capacity) -> 需求
    lane_nodes: dict[str, int] = {}
    lane_out_meta: dict[str, int] = {}
    window_nodes: dict[tuple[str, int], int] = {}
    window_out_meta: dict[tuple[str, int], int] = {}

    def lane_node_for(ctx: _LaneContext) -> int:
        if ctx.lane.lane_id not in lane_nodes:
            node = net.node()
            lane_nodes[ctx.lane.lane_id] = node
            wnode = window_node_for(ctx.lane.target, ctx.arrival)
            out_edge = net.add(node, wnode, _cap_int(ctx.lane_left), 0)
            lane_out_meta[ctx.lane.lane_id] = out_edge
        return lane_nodes[ctx.lane.lane_id]

    def window_node_for(target: str, arrival: date) -> int:
        key = (target, arrival.toordinal())
        if key not in window_nodes:
            node = net.node()
            w_left = window_residual[(target, arrival)]
            out_edge = net.add(node, demand_nodes[target], _cap_int(w_left), 0)
            window_nodes[key] = node
            window_out_meta[key] = out_edge
        return window_nodes[key]

    for site in sorted(lots_by_site):
        for lot in sorted(lots_by_site[site], key=lambda x: (x.expiry, x.lot)):
            lot_node = net.node()
            net.add(source, lot_node, _cap_int(lot.quantity), 0)
            feasible = False
            for target in sorted(lane_ctxs):
                for ctx in lane_ctxs[target]:
                    if ctx.lane.source != site:
                        continue
                    remaining_days = (lot.expiry - ctx.arrival).days
                    if not lot.released or lot.quarantined:
                        rejections.append(Rejection(sku, site, target, lot.lot, ctx.lane.lane_id,
                                                    RejectReason.LOT_INELIGIBLE,
                                                    "lot not released or quarantined"))
                        continue
                    if remaining_days < min_remaining_days:
                        rejections.append(Rejection(sku, site, target, lot.lot, ctx.lane.lane_id,
                                                    RejectReason.SHELF_LIFE,
                                                    f"{remaining_days}d remaining < {min_remaining_days}d"))
                        continue
                    feasible = True
                    expiry_benefit = weights.expiry * _expiry_fraction(lot.expiry, ship_day, weights)
                    lane_risk_cost = weights.lane_risk * Decimal(str(ctx.lane.risk))
                    per_unit = weights.lane * ctx.lane.cost_per_unit + lane_risk_cost - expiry_benefit
                    ln = lane_node_for(ctx)
                    le = net.add(lot_node, ln, _cap_int(lot.quantity), _cost_int(per_unit))
                    lot_edge_meta.append({"edge": le, "lot": lot, "ctx": ctx, "target": target})
            if not feasible and not any(
                l.enabled and l.source == site and l.target in need_by_site
                for l in lanes
            ):
                rejections.append(Rejection(sku, site, "-", lot.lot, "-",
                                            RejectReason.NO_LANE,
                                            "no enabled lane from source to any gap site"))

    _, total_cost = net.run(source, sink, _cap_int(total_gap))

    # 共享容量实际用量
    for lid, ei in lane_out_meta.items():
        q = Decimal(net.edges[ei].flow) / Decimal(SCALE)
        if q > 0:
            lane_usage[lid] = q
    for key, ei in window_out_meta.items():
        target, ordinal = key
        q = Decimal(net.edges[ei].flow) / Decimal(SCALE)
        if q > 0:
            window_usage[(target, date.fromordinal(ordinal))] = q

    # 批次边流量聚合成 Movement（同 源/目的/线路/到货日）
    moved: dict[tuple[str, str, str, date], list[tuple[StockLot, Decimal]]] = {}
    site_inflow: dict[str, Decimal] = {}
    for meta in lot_edge_meta:
        flow = net.edges[meta["edge"]].flow
        if flow <= 0:
            continue
        q = Decimal(flow) / Decimal(SCALE)
        lot: StockLot = meta["lot"]
        ctx: _LaneContext = meta["ctx"]
        target = meta["target"]
        moved.setdefault((lot.site, target, ctx.lane.lane_id, ctx.arrival), []).append((lot, q))
        site_inflow[target] = site_inflow.get(target, Decimal("0")) + q

    movements: list[Movement] = []
    for seq, (key, rows) in enumerate(sorted(moved.items()), start=1):
        src, target, lane_id, arrival = key
        qty = sum((q for _, q in rows), Decimal("0"))
        picks = tuple(LotPick(lot.lot, lot.expiry, q)
                      for lot, q in sorted(rows, key=lambda x: (x[0].expiry, x[0].lot)))
        movements.append(Movement(
            movement_id=f"{plan_id}:M{seq:03d}",
            sku=sku, source=src, target=target, lane_id=lane_id,
            quantity=qty, picks=picks, arrival_day=arrival,
        ))

    unmet: list[UnmetGap] = []
    for n in needs:
        remaining = n.gap - site_inflow.get(n.site, Decimal("0"))
        if remaining > EPS:
            unmet.append(UnmetGap(n.site, sku, n.gap, remaining,
                                  RejectReason.SHORTAGE_UNFILLED,
                                  "global flow could not cover gap under lane/window/safety constraints"))

    # 有可行边但零流量的候选：判定真实瓶颈，否则说明全局选了更优组合
    for meta in lot_edge_meta:
        if net.edges[meta["edge"]].flow > 0:
            continue
        lot: StockLot = meta["lot"]
        ctx: _LaneContext = meta["ctx"]
        target = meta["target"]
        gap_left = need_by_site[target].gap - site_inflow.get(target, Decimal("0"))
        lane_full = lane_usage.get(ctx.lane.lane_id, Decimal("0")) + EPS >= ctx.lane_left
        window_full = window_usage.get((target, ctx.arrival), Decimal("0")) + EPS >= ctx.window_left
        if lane_full and gap_left > EPS:
            reason, detail = RejectReason.LANE_CAPACITY, "lane capacity consumed by better-scoring movements"
        elif window_full and gap_left > EPS:
            reason, detail = RejectReason.RECEIVING_CAPACITY, "window consumed by better-scoring movements"
        else:
            reason, detail = RejectReason.ALTERNATIVE_SELECTED, \
                "global minimum-cost flow chose a cheaper lot/lane combination"
        rejections.append(Rejection(sku, lot.site, target, lot.lot, ctx.lane.lane_id, reason, detail))

    return _WaveResult(movements, rejections, unmet, lane_usage, window_usage,
                       Decimal(total_cost) / Decimal(SCALE) / Decimal(SCALE))


def build_plan(
    plan_id: str,
    lots: Iterable[StockLot],
    needs: Iterable[SiteNeed],
    lanes: Iterable[Lane],
    windows: Iterable[ReceivingWindow],
    site_safety_stock: dict[tuple[str, str], Decimal] | None = None,
    ship_day: date | None = None,
    min_remaining_days: int = 30,
    weights: Weights | None = None,
    now: datetime | None = None,
) -> TransferPlan:
    """先做全局方案：多 SKU 按政策波次有序消耗共享的线路/窗口容量；
    波次内由最小费用流对缺口、效期、安全库存、线路成本、收货容量联合优化。"""
    weights = weights or Weights()
    weights.validate()
    ship_day = ship_day or date.today()
    if min_remaining_days < 0:
        raise ValueError("min_remaining_days must be non-negative")
    safety = {k: _q(v) for k, v in (site_safety_stock or {}).items()}

    lot_rows = list(lots)
    need_rows = list(needs)
    lane_rows = list(lanes)
    window_rows = list(windows)
    for lot in lot_rows:
        lot.validate()
    for n in need_rows:
        n.validate()
    for lane in lane_rows:
        lane.validate()
    for w in window_rows:
        w.validate()
    lane_ids = [l.lane_id for l in lane_rows]
    if len(lane_ids) != len(set(lane_ids)):
        raise ValueError("duplicate lane id")
    need_keys = [(n.site, n.sku) for n in need_rows]
    if len(need_keys) != len(set(need_keys)):
        raise ValueError("duplicate need for same site and sku")

    # 站点在手量（仅放行批次）与安全库存扣减后的可调出盈余
    on_hand: dict[tuple[str, str], Decimal] = {}
    for lot in lot_rows:
        if lot.released and not lot.quarantined:
            on_hand[(lot.site, lot.sku)] = on_hand.get((lot.site, lot.sku), Decimal("0")) + lot.quantity
    surplus = {key: max(Decimal("0"), qty - safety.get(key, Decimal("0"))) for key, qty in on_hand.items()}

    window_residual: dict[tuple[str, date], Decimal] = {}
    for w in window_rows:
        window_residual[(w.site, w.day)] = window_residual.get((w.site, w.day), Decimal("0")) + w.capacity

    lane_residual: dict[str, Decimal] = {l.lane_id: l.capacity for l in lane_rows if l.enabled}

    need_by_sku: dict[str, list[SiteNeed]] = {}
    for n in need_rows:
        need_by_sku.setdefault(n.sku, []).append(n)
    lots_by_sku: dict[str, list[StockLot]] = {}
    for lot in lot_rows:
        lots_by_sku.setdefault(lot.sku, []).append(lot)

    # 政策波次：关键站点优先、缺口总量大优先
    wave_order = sorted(
        need_by_sku,
        key=lambda s: (-max(n.priority for n in need_by_sku[s]),
                       -sum(n.gap for n in need_by_sku[s]),
                       s),
    )

    all_movements: list[Movement] = []
    all_rejections: list[Rejection] = []
    all_unmet: list[UnmetGap] = []
    total_score = Decimal("0")

    for sku in wave_order:
        sku_needs = sorted(need_by_sku[sku], key=lambda n: (-n.priority, -n.gap, n.site))

        # 安全库存硬约束：按 FEFO 顺序保留近效期之外的可调出量
        sku_lots: list[StockLot] = []
        for site in sorted({l.site for l in lots_by_sku.get(sku, [])}):
            ordered = sorted((l for l in lots_by_sku[sku] if l.site == site),
                             key=lambda x: (x.expiry, x.lot))
            free = surplus.get((site, sku), Decimal("0"))
            for lot in ordered:
                if not lot.released or lot.quarantined:
                    sku_lots.append(lot)  # 网络层按 LOT_INELIGIBLE 留痕
                    continue
                if free <= 0:
                    for n in sku_needs:
                        all_rejections.append(Rejection(
                            sku, site, n.site, lot.lot, "-",
                            RejectReason.SOURCE_SAFETY_STOCK,
                            f"on-hand locked by safety stock at {site}"))
                    continue
                take = min(lot.quantity, free)
                if take < lot.quantity:
                    sku_lots.append(StockLot(lot.site, lot.sku, lot.lot, lot.expiry, take,
                                             lot.released, lot.quarantined))
                else:
                    sku_lots.append(lot)
                free -= take

        result = _solve_wave(
            sku, plan_id, ship_day, min_remaining_days,
            sku_lots, sku_needs, lane_rows,
            window_residual, lane_residual, weights,
        )
        # 波次原子提交到共享容量账本（先全局算，再顺序扣减）
        for lid, used in result.lane_usage.items():
            lane_residual[lid] = lane_residual.get(lid, Decimal("0")) - used
        for wk, used in result.window_usage.items():
            window_residual[wk] = window_residual.get(wk, Decimal("0")) - used
        all_movements.extend(result.movements)
        all_rejections.extend(result.rejections)
        all_unmet.extend(result.unmet)
        total_score += result.score

    generated = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return TransferPlan(
        plan_id=plan_id,
        ship_day=ship_day,
        min_remaining_days=min_remaining_days,
        movements=tuple(all_movements),
        unmet=tuple(all_unmet),
        rejections=tuple(_dedupe_rejections(all_rejections)),
        total_score=total_score,
        generated_at=generated,
        weights=weights,
    )


def _dedupe_rejections(rows: Iterable[Rejection]) -> list[Rejection]:
    seen: set[tuple] = set()
    out: list[Rejection] = []
    for r in rows:
        key = (r.sku, r.source, r.target, r.lot, r.lane_id, r.reason, r.detail)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def rejections_by_site(plan: TransferPlan, site: str) -> list[Rejection]:
    return [r for r in plan.rejections if r.source == site or r.target == site]


def rejection_summary(plan: TransferPlan) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in plan.rejections:
        out[r.reason.value] = out.get(r.reason.value, 0) + 1
    return dict(sorted(out.items()))
