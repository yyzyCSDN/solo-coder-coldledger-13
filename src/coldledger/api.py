from datetime import timedelta
from .inventory.fefo import reserve
from .quality.release import may_release
from .transport.connections import choose


class ColdLedgerService:
    def reserve_inventory(self,*args,**kwargs): return reserve(*args,**kwargs)
    def release_decision(self,*args,**kwargs): return may_release(*args,**kwargs)
    def next_connection(self,inbound,candidates,minutes): return choose(inbound,candidates,timedelta(minutes=minutes))
    def plan_transfers(self,*args,**kwargs):
        from .transfer.planner import build_plan
        return build_plan(*args,**kwargs)
    def commit_transfers(self,coordinator,plan,actor="planner"):
        return coordinator.commit(plan,actor)
    def health(self): return {'service':'coldledger','status':'ok','runtime_dependencies':0}
