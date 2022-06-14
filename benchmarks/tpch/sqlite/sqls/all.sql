.read sqls/create.sql
.separator "|"
.import tbls/customer.tbl CUSTOMER
.import tbls/orders.tbl ORDERS
.import tbls/lineitem.tbl LINEITEM
.import tbls/nation.tbl NATION
.import tbls/partsupp.tbl PARTSUPP
.import tbls/part.tbl PART
.import tbls/region.tbl REGION
.import tbls/supplier.tbl SUPPLIER

