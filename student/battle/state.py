from pathlib import Path
import json,sys
def update(path,batch):
    # Intentional incident: the previous file is never loaded.
    state={}
    for r in batch:state[r['event_id']]=r
    Path(path).write_text(json.dumps(state),encoding='utf-8')
    return {'count':len(state),'total':sum(r['amount'] for r in state.values())}
if __name__=='__main__':print(json.dumps(update(sys.argv[1],json.load(sys.stdin))))
