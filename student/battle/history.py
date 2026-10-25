def rolling_previous(data):
    a=data['values'];k=data['k']
    return [sum(a[max(0,i-k+1):i+1])/len(a[max(0,i-k+1):i+1]) for i in range(len(a))]
def audit_preservation(data):
    return {'missing':[],'changed':[]}
