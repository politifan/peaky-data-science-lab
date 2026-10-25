def alert_windows(data):
    return [r['id'] for r in data['windows'] if r['n'] and r['missing']/r['n']>=data['limit']]
def audit_predictions(data):
    return {'missing':[],'invalid':[],'duplicates':[],'extra':[]}
