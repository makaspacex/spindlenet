from standardbench2_candidates.DUAL_a_Odancukmk7hdtfnp_r45_C_trinorm_dsa3 import test as test1
from standardbench2_candidates.DUAL_a_Odancukmk7hnp_r45_C_trinorm_dsa3 import test as test2
from standardbench2_candidates.DUAL_a_Odancukmk8ahdtfnp_r45_C_trinorm_dsa3 import test as test3
from standardbench2_candidates.DUAL_a_Odancukmk8ahdtfnp_r45pttpt_C_trinorm_dsa3 import test as test4
from standardbench2_candidates.DUAL_b_Odancukmk8ahdtfnp_r45pttpt_C_trinorm_dsa3 import test as test5

def main():
    test1.main(only_conf=True)
    test2.main(only_conf=True)
    test3.main(only_conf=True)
    test4.main(only_conf=True)
    test5.main(only_conf=True)

if __name__ == '__main__':
    # main()
    # test1.main(only_conf=False)
    # test2.main(only_conf=False)
    test3.main(only_conf=False)
    # test4.main(only_conf=False)
    test5.main(only_conf=False)
