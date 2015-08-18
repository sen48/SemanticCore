__author__ = 'lvova'
from search_engine_tk.ya_query import queries_from_file

if __name__ == "__main__":
    queries = queries_from_file('c:/_Work/vostok/2.txt', 213)
    domains = {}
    for q in queries:
        print(q.query)
        items = q.get_serp(10)
        for item in items:
            host = item.get_domain()
            print(host)
            if host in domains:
                domains[host] += 1
            else:
                domains[host] = 1

    res = [(k, v) for k, v in domains.items() if 'vostok' not in k and 'tiu.ru' not in k  and 'mail.ru' not in k and
           'pulscen.ru' not in k and '.all.biz' not in k and 'dmir.ru' not in k and 'wikipedia' not in k and
           'istorya.pro' not in k ]
    counter = 0
    print([v for v in sorted(res, key=lambda v: -v[1])])
    print([v[0] for v in sorted(res, key=lambda v: -v[1])])