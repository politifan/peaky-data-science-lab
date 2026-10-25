def audit_features(data):
    # Intentional incident: availability checked in the wrong direction.
    return sorted(r['name'] for r in data['features'] if r['available_at']<data['decision_time'])
def leaked_users(rows):
    return sorted({r['id'] for r in rows if r['part']=='train'} & {r['id'] for r in rows if r['part']=='test'})
