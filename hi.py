from itertools import repeat

HBdata = ['K001',"K002",'K003','K004','K005','K006']

ir_faults  = ['KA01','KA03','KA05','KA06','KA07','KA08','KA09','KA04','KA15','KA16','KA22','KA30']
com_faults = ['KB23','KB24','KB27']
or_faults  = ['KI01','KI03','KI05','KI07','KI08','KI04','KI14','KI16','KI17','KI18','KI21']

samples = (
    list(zip(HBdata,     repeat("healthy"))) +
    list(zip(ir_faults,  repeat("inner_race"))) +
    list(zip(com_faults, repeat("combined"))) +
    list(zip(or_faults,  repeat("outer_race")))
)

# stable mapping
class_to_idx = {"healthy": 0, "inner_race": 1, "combined": 2, "outer_race": 3}

ALL_DATA  = [sid for sid, _ in samples]
ALL_LABEL = [class_to_idx[c] for _, c in samples]

print(ALL_DATA)
print(ALL_LABEL)