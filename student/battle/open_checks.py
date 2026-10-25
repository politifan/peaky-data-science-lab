import future,events,history,monitor
def check():
    tests=[('future',lambda:future.audit_features({'decision_time':10,'features':[{'name':'receipt','available_at':11}]}),['receipt']),('events',lambda:events.aggregate_events([{'event_id':'x','version':1,'amount':2},{'event_id':'x','version':2,'amount':4}]),{'count':1,'total':4}),('history',lambda:history.rolling_previous({'values':[10,20,90],'k':2}),[None,10,15]),('monitor',lambda:monitor.alert_windows({'windows':[{'id':'x','n':0,'missing':0}],'limit':0.2}),['x'])]
    results=[]
    for name,fn,expected in tests:
        actual=fn();results.append({'name':name,'ok':actual==expected,'actual':actual,'expected':expected})
    return results
if __name__=='__main__':
    r=check();print(r);raise SystemExit(0 if all(x['ok'] for x in r) else 1)
