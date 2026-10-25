def aggregate_events(rows):
    return {'count':len(rows),'total':sum(r['amount'] for r in rows)}
def audit_schema(data):
    return {'missing':sorted(set(data['required'])-set(data['actual'])),'wrong_type':[]}
