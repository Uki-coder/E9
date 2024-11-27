import scipy.optimize as opt
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

def linear(x,a,b):
    return b*x + a

def hyperbola(x,a,b):
    return b/x + a

def u1_av_calc(filename):
    u1_nums = np.loadtxt(filename)
    u1 = np.average(u1_nums)
    u1_diffs = u1_nums - u1
    u1_err = np.sqrt(sum(u1_diffs**2)/len(u1_diffs))
    u1_err = u1_err/np.sqrt(len(u1_diffs))

    return np.array([u1, 3*u1_err, np.abs(3*u1_err/u1)*100])

def calc_lin_u2u1(input_adresses, output_adresses, p0 = [1,0]):
    plt.rcParams['text.usetex'] = True

    u1 = np.loadtxt(input_adresses[0])
    u2 = np.loadtxt(input_adresses[1])

    lin_reg = stats.linregress(u1,u2)

    res = [lin_reg.slope, lin_reg.intercept, lin_reg.stderr, lin_reg.intercept_stderr]
    np.savetxt(output_adresses[0], res)

    delta = [res[2] * 3, res[3] * 3]
    np.savetxt(output_adresses[1], delta)

    eps = [np.abs(res[2]/res[0])*100, np.abs(res[3]/res[1])*100]
    np.savetxt(output_adresses[2], eps)

    fig, ax = plt.subplots()
    u1_linspace = np.linspace(min(u1)-1, max(u1)+1,1000)
    ax.plot(u1_linspace, linear(u1_linspace, res[1],res[0]), label='dopasowanie', c='b')
    ax.scatter(u1, u2, c='r', marker='o', label='pomiary')
    ax.set_xlabel('$U_1$ [V]')
    ax.set_ylabel('$U_2$ [V]')
    ax.legend()
    plt.grid()
    plt.show()

def calc_lin_u2n2(input_adresses, output_adresses):
    plt.rcParams['text.usetex'] = True

    u2 = np.loadtxt(input_adresses[1])
    n2 = np.loadtxt(input_adresses[0])

    u1 = u1_av_calc(input_adresses[2])
    u1[0] /= 140
    u1[1] /= 140
    np.savetxt(output_adresses[3], u1)

    lin_reg = stats.linregress(n2, u2)

    res = [lin_reg.slope, lin_reg.intercept, lin_reg.stderr, lin_reg.intercept_stderr]
    np.savetxt(output_adresses[0], res)

    delta = [res[2] * 3, res[3] * 3]
    np.savetxt(output_adresses[1], delta)

    eps = [np.abs(res[2] / res[0]) * 100, np.abs(res[3] / res[1]) * 100]
    np.savetxt(output_adresses[2], eps)

    fig, ax = plt.subplots()
    n2_linspace = np.linspace(min(n2) - 1, max(n2) + 1, 1000)
    ax.plot(n2_linspace, linear(n2_linspace, res[1], res[0]), label='dopasowanie', c='b')
    ax.scatter(n2, u2, c='r', marker='o', label='pomiary')
    ax.set_xlabel('$n_2$')
    ax.set_ylabel('$U_2$ [V]')
    ax.legend()
    plt.grid()
    plt.show()

def calc_hyp_u2n1(input_adresses, output_adresses):
    plt.rcParams['text.usetex'] = True

    u2 = np.loadtxt(input_adresses[1])
    n1 = np.loadtxt(input_adresses[0])

    u1 = u1_av_calc(input_adresses[2])
    u1[0] *= 140
    u1[1] *= 140
    np.savetxt(output_adresses[3],u1)



    popt, pcov = opt.curve_fit(hyperbola, n1, u2,p0=[0,610])
    perr = np.sqrt(np.diag(pcov))

    res = [popt[1], popt[0], perr[1], perr[0]]
    np.savetxt(output_adresses[0], res)

    delta = [res[2] * 3, res[3] * 3]
    np.savetxt(output_adresses[1], delta)

    eps = [np.abs(res[2] / res[0]) * 100, np.abs(res[3] / res[1]) * 100]
    np.savetxt(output_adresses[2], eps)

    fig, ax = plt.subplots()
    n1_linspace = np.linspace(min(n1) - 1, max(n1) + 1, 1000)
    ax.plot(n1_linspace, hyperbola(n1_linspace, res[1], res[0]), label='dopasowanie', c='b')
    ax.scatter(n1, u2, c='r', marker='.', label='pomiary')
    ax.set_xlabel('$n_1$')
    ax.set_ylabel('$U_2$ [V]')
    ax.legend()
    plt.grid()
    plt.show()

def calc_lin_i2i1():
    plt.rcParams['text.usetex'] = True

    i2_1 = np.loadtxt('input/i2i1_1/i2.txt')
    i1_1 = np.loadtxt('input/i2i1_1/i1.txt')

    i2_2 = np.loadtxt('input/i2i1_2/i2.txt')
    i1_2 = np.loadtxt('input/i2i1_2/i1.txt')

    i2i1_1 = stats.linregress(i1_1, i2_1)
    i2i1_2 = stats.linregress(i1_2, i2_2)

    res1 = [i2i1_1.slope, i2i1_1.intercept, i2i1_1.stderr, i2i1_1.intercept_stderr]
    np.savetxt('output/i2i1_1/i2i1_res.txt', res1)

    delta1 = [res1[2] * 3, res1[3] * 3]
    np.savetxt('output/i2i1_1/i2i1_delta.txt', delta1)

    eps1 = [np.abs(res1[2] * 3 / res1[0]) * 100, np.abs(res1[3] * 3 / res1[1]) * 100]
    np.savetxt('output/i2i1_1/i2i1_eps.txt', eps1)

    res2 = [i2i1_2.slope, i2i1_2.intercept, i2i1_2.stderr, i2i1_2.intercept_stderr]
    np.savetxt('output/i2i1_2/i2i1_res.txt', res2)

    delta2 = [res2[2] * 3, res2[3] * 3]
    np.savetxt('output/i2i1_2/i2i1_delta.txt', delta2)

    eps2 = [np.abs(res2[2] * 3 / res2[0]) * 100, np.abs(res2[3] * 3 / res2[1]) * 100]
    np.savetxt('output/i2i1_2/i2i1_eps.txt', eps2)

    fig, ax = plt.subplots()
    i1_1linspace = np.linspace(min(i1_1) - 0.1, max(i1_1) + 0.1, 1000)
    ax.plot(i1_1linspace, linear(i1_1linspace, res1[1], res1[0]), label='dopasowanie', c='b')
    ax.scatter(i1_1, i2_1, c='r', marker='o', label='pomiary')
    ax.set_xlabel('$I_1$ [A]')
    ax.set_ylabel('$I_2$ [A]')
    ax.legend()
    plt.grid()
    plt.show()

    fig, ax = plt.subplots()
    i1_2linspace = np.linspace(min(i1_2) - 0.1, max(i1_2) + 0.1, 1000)
    ax.plot(i1_2linspace, linear(i1_2linspace, res2[1], res2[0]), label='dopasowanie', c='b')
    ax.scatter(i1_2, i2_2, c='r', marker='o', label='pomiary')
    ax.set_xlabel('$I_1$ [A]')
    ax.set_ylabel('$I_2$ [A]')
    ax.legend()
    plt.grid()
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(i1_1linspace, linear(i1_1linspace, res1[1], res1[0]), label=r'dopasowanie $I_2(I_1)$ gdy $\frac{n_1}{n_2}=1$', c='b')
    ax.plot(i1_2linspace, linear(i1_2linspace, res2[1], res2[0]), label=r'dopasowanie $I_2(I_1)$ gdy $\frac{n_1}{n_2}=2$', c='r')
    ax.set_xlabel('$I_1$ [A]')
    ax.set_ylabel('$I_2$ [A]')
    ax.legend()
    plt.grid()
    plt.show()