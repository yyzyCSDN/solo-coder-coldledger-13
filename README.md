# ColdLedger Pharmaceutical Cold-Chain Operations Lab

Dependency-free Python reference implementation for temperature-sensitive pharmaceutical distribution. It models inventory expiry, chain-of-custody, telemetry correction, packaging qualification, dry-ice depletion, freezer allocation, customs holds, serialization, recalls, release decisions, vehicle compartment compatibility, returns, SLA calendars and cross-dock state.

It also plans multi-site stock transfers globally (`coldledger.transfer.network_allocation`): every candidate move is scored jointly on shortfall, expiry exposure, source safety-stock headroom, lane cost/time and receiving-window slack, with lane and receiving capacity reserved during the search. Reasons for every rejected candidate and every residual gap are kept on the plan. The plan is then committed atomically across the stock, lane-capacity and receiving-window books (`coldledger.transfer.network_commit`) under optimistic version checks, fixed lock ordering and idempotent references.

Runtime third-party dependencies: none.
