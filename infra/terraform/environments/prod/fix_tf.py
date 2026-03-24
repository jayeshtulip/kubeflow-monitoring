with open("main.tf") as f:
    c = f.read()

c = c.replace('engine_version    = "15.4"', 'engine_version    = "15.8"')
c = c.replace('desired_size   = 1\n\n      disk_size', 'desired_size   = 0\n\n      disk_size')

with open("main.tf", "w") as f:
    f.write(c)

print("Fixed - changes made:")
print("  PostgreSQL 15.4 -> 15.8")
print("  GPU node desired_size 1 -> 0")
