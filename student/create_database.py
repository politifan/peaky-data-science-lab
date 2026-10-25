from pathlib import Path
import sqlite3,csv
ROOT=Path(__file__).resolve().parent
def build(out):
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    # Preserve earlier learner work; a retry must select a fresh destination.
    with out.open('xb'):
        pass
    c=sqlite3.connect(out)
    c.executescript('DROP TABLE IF EXISTS orders; DROP TABLE IF EXISTS customers; CREATE TABLE orders(order_id TEXT PRIMARY KEY,customer_id TEXT,created_at TEXT,distance REAL,weight REAL,city TEXT,weather TEXT,hour INTEGER,duration REAL,late INTEGER); CREATE TABLE customers(customer_id TEXT PRIMARY KEY,segment TEXT);')
    with (ROOT/'data/orders.csv').open(encoding='utf-8') as f:
        a=list(csv.DictReader(f))
    c.executemany('INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?)',[(r['order_id'],r['customer_id'],r['created_at'],float(r['distance']) if r['distance'] else None,float(r['weight']),r['city'],r['weather'],int(r['hour']),float(r['duration']),int(r['late'])) for r in a])
    with (ROOT/'data/customers.csv').open(encoding='utf-8') as f:b=list(csv.DictReader(f))
    c.executemany('INSERT INTO customers VALUES (?,?)',[(r['customer_id'],r['segment']) for r in b]);c.commit()
    result={'orders':c.execute('SELECT COUNT(*) FROM orders').fetchone()[0],'unmatched':c.execute('SELECT COUNT(*) FROM orders o LEFT JOIN customers c ON o.customer_id=c.customer_id WHERE c.customer_id IS NULL').fetchone()[0]};c.close();return result
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'outputs/orders.sqlite');a=p.parse_args()
    print(build(a.out))
