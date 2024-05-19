import numpy as np
from scipy.signal import freqz
from scipy.integrate import simps

def burg_algorithm(x, order):
    """
    Estimate AR parameters using the Burg method.
    """
    N = len(x)
    if order >= N:
        raise ValueError("Order must be less than the length of the time series")

    # Initialize variables
    a = np.zeros(order + 1)
    a[0] = 1.0
    ef = np.zeros(N)
    eb = np.zeros(N)
    ef[1:] = x[:-1]
    eb[:-1] = x[1:]

    # Initial error
    e = x.copy()

    # Main loop to compute AR coefficients
    for k in range(1, order + 1):
        # Compute reflection coefficient
        num = -2.0 * np.dot(eb[:-k], ef[k:])
        den = np.dot(eb[:-k], eb[:-k]) + np.dot(ef[k:], ef[k:])
        kf = num / den

        # Update forward and backward prediction errors
        ef_new = ef[k:] + kf * eb[:-k]
        eb[:-k] = eb[:-k] + kf * ef[k:]
        ef[k:] = ef_new

        # Update AR coefficients
        a_new = a[:k + 1] + kf * np.flip(a[:k + 1])
        a[:k + 1] = a_new

    return a

def power_spectrum_burg(x, order, nfft=None):
    """
    Calculate the power spectrum using the Burg algorithm.
    """
    # Estimate AR parameters
    ar_params = burg_algorithm(x, order)

    # Compute the PSD
    if nfft is None:
        nfft = 2 ** (np.ceil(np.log2(len(x)))).astype(int)
    # freqz returns the frequency response of the filter
    ps, h = freqz([1], ar_params, worN=nfft, whole=True)
    psd = 1 / np.abs(h) ** 2

    power = simps(psd, ps)

    return ps, power

def burg_():
    import numpy as np
    from scipy.signal import lfilter

    def burg_power_spectrum(signal, order):
        N = len(signal)
        a = np.zeros(order + 1)
        k = np.zeros(order + 1)
        P = np.zeros(order + 1)
        E = np.sum(signal ** 2)  # Compute the total energy of the signal

        # Initialize the recursion
        P[0] = E / N
        for j in range(N):
            k[0] += signal[j] * signal[j]

        for i in range(1, order + 1):
            den = 0
            num = 0
            for j in range(i, N):
                den += signal[j] * signal[j - i]
                num += signal[j] * signal[j]
            a[i] = (2 * den - P[i - 1] * k[i - 1]) / (E - P[i - 1] * P[i - 1])
            k[i] = (1 - a[i] * a[i]) * k[i - 1]
            for j in range(i):
                a[j] = a[j] - a[i] * a[i - j]

            P[i] = (1 - a[i] * a[i]) * P[i - 1]

        # Compute the power spectrum using the coefficients
        w, h = np.fft.rfftfreq(N), np.zeros(N)
        h[0] = P[0]
        for i in range(1, N):
            for j in range(1, min(i, order) + 1):
                if i - j >= 0:
                    if not np.isnan(a[j]) and not np.isnan(h[i - j]):
                        h[i] += a[j] * h[i - j]
            h[i] += P[order]

        return w, h

    # Example usage
    # Replace signal with your own data and set the order according to your requirements
    # signal = np.random.randn(1000)
    # print(signal)
    ai = (pd.read_csv('sentence_length/Cgtd_ai.csv'))['Y']
    human = (pd.read_csv('bow_vector/Cgtd_human.csv'))['Y']
    ai = np.array(ai)
    ai = ai / ai.sum()

    human = np.array(human)
    human = human / human.sum()


    order = 3
    freq, power_spectrum = burg_power_spectrum(ai, order)
    freq1, power_spectrum1 = burg_power_spectrum(human, order)
    print(power_spectrum)


    # plt.plot(power_spectrum,label = 'ai')
    plt.plot(power_spectrum1,label = 'human')
    # plt.yscale('log')
    # plt.xscale('log')
    plt.legend()
    plt.show()


def burg_2(u):
    import numpy as np
    import matplotlib.pyplot as plt
    import math
    '''
    Burg算法求解5阶AR模型参数
    并绘制p分别取3,4,5时的功率谱曲线
    '''

    N = len(u)
    # print("数据长度为:" + str(N))
    k = 10  # 阶数
    p_list= []
    # 数据初始化
    fO = u[:]  # 0阶前向误差
    bO = u[:]  # 0阶反向误差
    f = u[:]  # 用于更新的误差变量
    b = u[:]
    a = np.array(np.zeros((k + 1, k + 1)))  # 模型参数初始化
    for i in range(k + 1):
        a[i][0] = 1
    # 计算P0 1/N*sum(u*2)
    P0 = 0
    for i in range(N):
        P0 += u[i] ** 2
    P0 /= N
    # print("P0:" + str(P0))

    P = [P0]

    # Burg 算法更新模型参数
    for p in range(1, k + 1):
        Ka = 0  # 反射系数的分子
        Kb = 0  # 反射系数的分母
        for n in range(p, N):
            Ka += f[n] * b[n - 1]
            Kb = Kb + f[n] ** 2 + b[n - 1] ** 2
        K = 2 * Ka / (Kb+1e-9)
        # print("第%d阶反射系数:%f" % (p, K))
        # 更新前向误差和反向误差
        fO = f[:]
        bO = b[:]
        for n in range(p, N):
            b[n] = -K * fO[n] + bO[n - 1]
            f[n] = fO[n] - K * bO[n - 1]
        # 更新此时的模型参数
        # print("第%d阶模型参数：" % p)
        for i in range(1, p + 1):
            if (i == p):
                a[p][i] = -K
            else:
                a[p][i] = a[p - 1][i] - K * a[p - 1][p - i]
            # print("a%d=%f" % (i, a[p][i]))
        P.append((1 - K ** 2) * P[p - 1])
        p_list.append(P[p])

    # 计算第k阶的功率谱
    def calPSD(k, l=512):
        H = np.array(np.zeros(l), dtype=complex)
        for f in range(1, l):
            f1 = f * 0.5 / l  # 频率值
            for i in range(1, k + 1):
                H[f] += complex(a[k][i] * np.cos(2 * np.pi * f1 * i), -a[k][i] * np.sin(2 * np.pi * f1 * i))
            H[f] += 1
            H[f] = 1 / H[f]  # 系统函数的表达式
            H[f] = 10 * math.log10(np.abs(H[f]) ** 2 * P[k])
        return H
    h_list = []
    for i in range(10):
        i = i+1
        h_list.append(calPSD(i))


    return p_list, h_list

def plt_demo(ai_file, human_file, s1, s2, title,flag,leg):
        ai = np.array((pd.read_csv(f'sentence_length/{ai_file}'))['Y'])
        human = np.array((pd.read_csv(f'sentence_length/{human_file}'))['Y'])

        ai_avg, aih_list = burg_2(ai)
        print(aih_list)
        human_avg, huh_list = burg_2(human)
        print(huh_list)

        l = 512
        p = 1
        plt.subplot(2, 3, s1)
        for item, i in enumerate(aih_list):
            plt.plot(np.arange(0, p, p / l), i, '-.', color=colors[item])

        for item, j in enumerate(huh_list):
            plt.plot(np.arange(0, p, p / l), j, label="$p$ = {}".format(item + 1), color=colors[item])

        # plt.xlabel("Frequency (Hz)", fontsize=16)
        if flag==1:
            plt.ylabel('Power spectral density (dB/Hz)', fontsize=16)
        if leg ==1:
            plt.legend(prop={'size': 15},ncol = 2)
        plt.title('${}$'.format(title), fontsize=18)

        plt.subplot(2, 3, s2)

        plt.plot(ai_avg, label='AI', color=colors[14])
        plt.plot(human_avg, label='Human', color=colors[13])
        for a, b in zip(range(10), ai_avg):
            plt.text(a, b, round(b, 5), ha='right', va='bottom', fontsize=12, color=colors[3])
        for c, d in zip(range(10), human_avg):
            plt.text(c, d, round(d, 5), ha='left', va='top', fontsize=12, color=colors[4])
        # plt.xscale("log")
        plt.yscale("log")

        # plt.xlabel("Order", fontsize=16)
        if flag==1:
            plt.ylabel('Average power', fontsize=16)
        if leg==1:
            plt.legend(prop={'size': 15},ncol = 2)