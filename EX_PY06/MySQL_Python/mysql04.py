import pymysql

conn = pymysql.connect(host='localhost', user='id', password='password',
                        db='sqlclass_db', charset='utf8')
curs = conn.cursor()
sql = """insert	into customer(name, category, region)
         values	(%s, %s, %s)"""

curs.execute(sql, ('홍길동', 1, '서울')) 
curs.execute(sql, ('이연수', 2, '서울')) 
conn.commit()

print('INSERT 완료')

curs.execute('select * from customer')
rows = curs.fetchall() # 모든 데이터를 가져옴
print(rows)

curs.close() 
conn.close()