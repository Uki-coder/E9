import matplotlib as mpl
from calc import*

mpl.use('TkAgg')


input_u2u1 = ['input/u1u2/u1.txt', 'input/u1u2/u2.txt']
output_u2u1 = ['output/u1u2/res_u2u1.txt', 'output/u1u2/delta_u2u1.txt', 'output/u1u2/epsilon_u2u1.txt']
calc_lin_u2u1(input_u2u1, output_u2u1)


input_u2n2 = ['input/u2n2/n2.txt', 'input/u2n2/u2.txt', 'input/u2n2/u1.txt']
output_u2n2 = ['output/u2n2/res_u2n2.txt', 'output/u2n2/delta_u2n2.txt',\
               'output/u2n2/epsilon_u2n2.txt', 'output/u2n2/th.txt']
calc_lin_u2n2(input_u2n2, output_u2n2)


input_u2n1 = ['input/u2n1/n1.txt', 'input/u2n1/u2.txt', 'input/u2n1/u1.txt']
output_u2n1 = ['output/u2n1/res_u2n1.txt', 'output/u2n1/delta_u2n1.txt',\
               'output/u2n1/epsilon_u2n1.txt', 'output/u2n1/th.txt']
calc_hyp_u2n1(input_u2n1, output_u2n1)


calc_lin_i2i1()