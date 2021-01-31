'''моделирование двумерного распределения примеси и толщины оксида кремния
после проведения процесса диффузии в кремнии в окислительной среде через окно в Si3N4. (2 мкм).
Хахождение распределение глубины залегания pn-перехода вдоль поверхности структуры'''

import matplotlib.pyplot as plt #Библиотека для работы с отображением данных
import numpy as np #Библиотека для работы с данными
import math #Библиотека для математических операций
import math as mt
from mpl_toolkits.mplot3d import Axes3D #Импорт инструмента для рисования в 3D
from scipy.special import erfc #Импорт функции ошибок
from matplotlib.widgets import Slider #слайдер для того, чтобы выбирать время

#Константы
Q = 5e14
E = 180
T = 1150+273
p = 2
k = 8.62e-5
N0 = 2e17
time = 20

#Класс кремния, включает зависимости и константы спецефичные для него
class Si:
    def __init__(self):
        # set constants
        self.m_dn = 1.08
        self.m_dp = 0.59
        self.epsilon = 11.7
        self.mu_n_300 = 1350
        self.mu_p_300 = 450
    def Eg(self, temperature):
        return 1.17 - 4.73e-4 / (temperature + 636) * temperature ** 2


    def Ei(self, temperature):
        return self.Eg(temperature) / 2 - k * temperature / 4

    def Nv(self, temperature):
        return 4.82e15 * (self.m_dp ** 1.5) * (temperature ** 1.5)

    def Nc(self, temperature):
        return 4.82e15 * (self.m_dn ** 1.5) * (temperature ** 1.5)

    def ni(self, temperature):
        return ((self.Nc(temperature) * self.Nv(temperature)) ** 0.5) * mt.exp(
            -1 * self.Eg(temperature) / (2 * temperature * k))

    def Rp(self, E):
        return self.a1 * E ** self.a2 + self.a3

    def dRp(self, E):
        return self.a4 * E ** self.a5 + self.a6

    def gamma(self, E):
        return self.a7 / (self.a8 + E) + self.a9

    def betta(self, E):
        return self.a10 / (self.a11 + E) + self.a12 + self.a13 * E

    def dRpl(self, E):
        return np.sqrt(2 * self.Rp(E) / (self.Se + self.Sn) * E - self.dRp(E) ** 2 - self.dRp(E) ** 2)

    def C_minus(self, temperature):
        return mt.exp((self.Ei(temperature) + 0.57 - self.Eg(temperature)) / k / temperature)

    def C_plus(self, temperature):
        return mt.exp((0.35 - self.Ei(temperature)) / k / temperature)

    def C_doubleminus(self, temperature):
        return mt.exp((2 * self.Ei(temperature) + 1.25 - 3 * self.Eg(temperature)) / k / temperature)

    def gamma_l(self, temperature):
        return 2620 * mt.exp(-1.1 / k / temperature)

    def gamma_p(self, temperature):
        return 9.63e-16 * mt.exp(2.83 / k / temperature)
        # Подстроечный параметр, см^0.66
    def Vp(self, temperature):
        return 9.63e-16 * np.exp(2.83 / k / temperature)

    def Vn(self, temperature, n):
        return (1 + self.C_plus(temperature) * self.ni(temperature) / n
                + self.C_minus(temperature) * n / self.ni(temperature)
                + self.C_doubleminus(temperature) * (n / self.ni(temperature)) ** 2) \
               / (1 + self.C_minus(temperature) + self.C_plus(temperature) + self.C_doubleminus(temperature))
    # Подстроечный параметр
    def Vl(self, temperature):
        return 2620 * np.exp(-1.1 / k / temperature)

    def ni(self, temperature):
        return ((self.Nc(temperature) * self.Nv(temperature)) ** 0.5) * np.exp(
            -1 * self.Eg(temperature) / (2 * temperature * k))

#Класс фосфора создается на основе кремния. Вносятся зависимости a-шек
class Phosfor(Si):

    def __init__(self):
        self.a1 = 0.001555
        self.a2 = 0.958
        self.a3 = 0.000828
        self.a4 = 0.002242
        self.a5 = 0.659
        self.a6 = -0.003435
        self.a7 = 336.2
        self.a8 = 199.3
        self.a9 = -1.386
        self.a10 = 54.45
        self.a11 = 55.74
        self.a12 = 1.865
        self.a13 = 0.00482
        # для dRpl
        self.Se = 520
        self.Sn = 1833.33
        # Дозаполнение параметров самого кремния
        self.m_dn = 1.08
        self.m_dp = 0.59
        self.epsilon = 11.7
        self.mu_n_300 = 1350
        self.mu_p_300 = 450

#Cоздание объектов классов фосфора и кремния в дальнейшем работа будет с ними
Phosfor = Phosfor()
Si = Si()

#Функция расчета коэфициента диффузии для фосфора
def Dif(C, temperature, ni):
    #Составной D для фосфора
    D0 = 3.85*math.exp(-3.66/(k*temperature))
    D1 = 4.44*2*C/ni*math.exp(-4/(k*temperature))
    D2 = 44.2*math.exp(-4.37/(k*temperature))*((C*2/ni)**2)
    return D0+D1+D2

#Функция для расчета начального распределения
def implantation(x, y, Q, dRp, Rp, dRpl, a):
    return Q / np.sqrt(2 * np.pi) / dRp / 1e-4 * np.exp(-(x - Rp) ** 2 / 2 / dRp ** 2) / 2 * (
                erfc((y - a) / np.sqrt(2) / dRpl) - erfc((y + a) / np.sqrt(2) / dRpl))
#
def raspredelenie_in_time(y, n, dx, T, ni):
    a = np.empty(n)
    b = np.empty(n)
    d = np.empty(n)
    r = np.empty(n)
    delta = np.empty(n)
    lyamda = np.empty(n)
    if n == 40:
        d[0] = 1
        a[0] = -1
    else:
        d[0] = 0
        a[0] = 1
    b[0] = 0
    r[0] = 0
    d[-1] = 0
    a[-1] = 1
    b[-1] = 0
    r[-1] = 0
    delta[0] = -d[0] / a[0]
    lyamda[0] = r[0] / a[0]
    dt = 60
    for i in range(1, n - 1, 1):
        a[i] = -(2 + (dx ** 2 * 1e-8) / (Dif(y[i], T, ni) * dt))
        r[i] = (y[i+1]-(2-dx**2*1e-8/(Dif(y[i], T, ni) * dt))*y[i]+y[i-1])
        b[i] = 1
        d[i] = 1
    for i in range(1, n, 1):
        delta[i] = -d[i] / (a[i] + b[i] * delta[i - 1])
        lyamda[i] = (r[i] - b[i] * lyamda[i - 1]) / (a[i] + b[i] * delta[i - 1])
    y[-1] = lyamda[-1]
    for i in range(n - 2, -1, -1):
       y[i] = delta[i] * y[i + 1] + lyamda[i]
    return y


def oxidation(time, T, concentration, xi):
    sc = Si
    P = 2 # давление
    C2 = 2e17 # концентрация из давления
    B = K(A=7, Ea=0.78, temperature=T) * (1 + sc.Vp(T) * sc.ni(T) ** 0.22)*P
    A = B / (K(A=2.96e6, Ea=2.05, temperature=T) * (1 + sc.Vl(T) * (sc.Vn(T, C2) - 1))) * 1.68/P
    # вызов функции расчета толщины оксида с новыми параметрами
    x = x_t(time, A, B, xi)
    return x

def oxidation_firstlayer():
    sc = Si
    x0 = 0  # Начальная толщина оксила равна нулю т.к. окисление в парах воды
    xi1 = 0.1 / 0.45  # мкм  Толщина первого слоя окисла 0,1. 0,45 кремния от превого слоя x0 поглащается поэтому еще делим на 0.45
    t = []
    x = []
    x1 = []
    t1 = []
    P = 2
    dt = 1
    # "А" для первого слоя с концентрацией C1 = 1e18
    C1 = 1e18
    C2 = 2e17
    B = K(A=7, Ea=0.78, temperature=T) * (1 + sc.Vp(T) * sc.ni(T) ** 0.22) * P
    A = B / (K(A=2.96e6, Ea=2.05, temperature=T) * (1 + sc.Vl(T) * (sc.Vn(T, C2) - 1))) * 1.68 / P
    A1 = B / (K(A=2.96e6, Ea=2.05, temperature=T) * (1 + sc.Vl(T) * (sc.Vn(T, C1) - 1))) * 1.68 / P
    # Присвоение начальных значений для первого слоя
    t1.append(0)
    x1.append(x_t(0, A1, B, xi1))
    while x1[-1] <= xi1:
        t1.append(t1[-1] + dt)
        x1.append(x_t(t1[-1], A1, B, x0))
    t.append(t1[-1])
    x.append(x_t(t[-1] - t1[-1], A, B, x1[-1]))
    return t, x, x[-1]

#Функция расчитывающая линейную параболическую константу
def K(A, Ea, temperature):
    return A * mt.exp(- Ea / k / temperature)

def x_t(t, A, B, xi):
    tau = (xi ** 2 + A * xi) / B
    return (A / 2) * (mt.sqrt(1 + (t + tau) / A ** 2 * 4 * B) - 1)

def oxide(T, time, concentration, x0):
    sc = Si
    n = time * 60
    t = np.empty(n)
    x = np.empty(n)
    t[0] = 0
    x[0] = x0
    dt=1
    P=2
    if time == 1:
        return oxidation_firstlayer()
    else:
        for i in range(1, n):
            t[i] = t[i - 1] + dt
            x[i] = oxidation(t[i], T, concentration, x0)
        return t, x, x[-1]

class Plot:
    Y = []
    X = []
    ox_plot_x = []
    ox_plot_y = []
    z_ox = []
    def draw(self, start, end, x, y, dx, dy, xmax, T, Rp, dRp, dRpl, Q, ni, t):
        for i in range(len(x)-1, 0, -1):
            if i % 2 == 0:
                self.Y.append(-2*x[i])
        for j in range(0, len(x), 1):
            self.X.append(x[j])
            if j % 2 == 0:
                self.Y.append(2*x[j])
        dy = dx
        X, Y = np.meshgrid(self.X, self.Y)
        Z = X ** 2 - Y ** 2
        for i in range(0, len(self.Y), 1):
            for j in range(0, len(self.X), 1):
                Z[i][j] = implantation(X[i][j], Y[i][j], Q, dRp, Rp, dRpl, xmax / 2)
        fig = plt.figure('graph')
        ax = fig.add_subplot(projection='3d')
        fig.suptitle('Распределение концентрации фосфора')
        ax.set_xlim(0, xmax)
        ax.set_ylim(-2 * xmax, 2 * xmax)
        ax.set_zlim(0, 3e19)
        ax.set_xlabel('x, мкм')
        ax.set_ylabel('y, мкм')
        ax.set_zlabel('Концентрация, C')
        surface = ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color='red', linewidth=0.5, label='Загонка')
        surface2 = ax.plot_surface(X, Y, Z, cmap='winter', edgecolor='none')
        fig.suptitle('Имплантация фосфора P в кремнии')
        ax_time = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        Time = Slider(ax_time, 'Время', 1, 20, valinit=0, valstep=1, dragging=True)
        colourbar = fig.colorbar(surface2, shrink=0.3, aspect=5, orientation='vertical', label='Разгонка')
        def get_junct(x, y ,t, end, conc = N0):
            Z_r = Z.copy()
            z_new = np.zeros(end)
            junc_arr = np.zeros(end)
            index = []
            for j in range(1, t, 1):
                for i in range(0, len(Z_r[1, :]), 1):
                    Z_r[:, i] = raspredelenie_in_time(Z_r[:, i], end - 1, dx, T, ni)
                for i in range(0, len(Z[:, 1]), 1):
                    Z_r[i] = raspredelenie_in_time(Z_r[i], end, dy, T, ni)
            for i in range(0, end-1):
                junc_arr[i] = Z_r[int(end / 2)][i]
                if (junc_arr[i] > conc and junc_arr[i - 1] < conc) or (junc_arr[i] < conc and junc_arr[i - 1] > conc):
                    index.append(i - 1)
            for i in range(0, end - 1):
                z_new[i]= (Z_r[i][index[0]])

            fig = plt.figure('Сечение p-n перехода')
            fig.suptitle('Сечение p-n перехода')
            plt.xlabel('y, мкм')
            plt.ylabel('Концентрация, см-3')
            plt.plot(y, z_new)
            plt.show()

        def spread(t, y, end, data1):
            tox = np.zeros(end)
            for j in range(0, end):
                if y[j] > 1 and y[j] < 3:
                    tox[j] = data1[-1]
                else:
                    if y[j] < 1:
                        tox[j] = data1[-1] / 2 * (1 + 1 \
                             / (1 + 0.27893 * 2 ** 0.5 / 2 \
                                 * (1 - y[j]) / data1[-1] \
                                 + 0.230389 * (2 ** 0.5 / 2 * (1 - y[j]) \
                                               / data1[-1]) ** 2 + 0.000972 *
                                 (2 ** 0.5 / 2 * (1 - y[j]) / data1[-1]) ** 3 \
                                 + 0.078108 * (2 ** 0.5 / 2 * (1 - y[j]) / data1[-1]) ** 4) ** 4)
                    else:
                        tox[j] = data1[-1] / 2 * (1 + 1 \
                                / (1 + 0.27893 * 2 ** 0.5 / 2 \
                                   * (y[j] - 3) / data1[-1] \
                                   + 0.230389 * (2 ** 0.5 / 2 * (y[j] - 3) \
                                                 / data1[-1]) ** 2 + 0.000972 *
                                   (2 ** 0.5 / 2 * (y[j] - 3) / data1[-1]) ** 3 \
                                   + 0.078108 * (2 ** 0.5 / 2 * (y[j] - 3) / data1[-1]) ** 4) ** 4)
            return y, tox

        def get_Z(t, end):
            Z_r = Z.copy()
            for j in range(1, t, 1):
                for i in range(0, len(Z_r[1, :]), 1):
                    Z_r[:, i] = raspredelenie_in_time(Z_r[:, i], end - 1, dx, T, ni)
                for i in range(0, len(Z[:, 1]), 1):
                    Z_r[i] = raspredelenie_in_time(Z_r[i], end, dy, T, ni)
            index = int(end / 2)
            return Z_r[index][index]

        # Обновляет график при изменении значения слайдера
        def update(val):
            ax.clear()
            X_r = X.copy()
            Y_r = Y.copy()
            Z_r = Z.copy()
            ax.set_xlim(0, xmax)
            ax.set_ylim(-2 * xmax, 2 * xmax)
            ax.set_zlim(0, 3e19)
            ax.set_xlabel('x, мкм')
            ax.set_ylabel('y, мкм')
            ax.set_zlabel('Концентрация, C')
            t = int(Time.val)
            for j in range(1, t, 1):
                for i in range(0, len(Z_r[1, :]), 1):
                    Z_r[:, i] = raspredelenie_in_time(Z_r[:, i], end - 1, dx, T, ni)
                for i in range(0, len(Z[:, 1]), 1):
                    Z_r[i] = raspredelenie_in_time(Z_r[i], end, dy, T, ni)
            surface2 = ax.plot_surface(X_r, Y_r, Z_r, cmap='winter', edgecolor='none')
            surface = ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color='red', linewidth=0.5, label='Загонка')
            ax.legend(title="Обозначения")
        # запускает обновление графика при изменении слайдера
        Time.on_changed(update)

        x0_arr = [50e-3]
        for i in range(1, t):
            if i == 1:
                x0 = x0_arr[0]
            else:
                x0 = x0_arr[i - 1]
            data = oxide(T, i, get_Z(i, end), x0)
            x0_arr.append(data[2])
        get_junct(x, y, t, end)
        max_ox = max(data[1])
        spread_data = spread(t * 60, y, end, data[1])
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.25, bottom=0.25)
        sp_ox, = plt.plot(spread_data[0], spread_data[1])
        plt.xlabel("у, мкм")
        plt.ylabel('Толщина окисла, мкм')
        #spox_time = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        #Time_spox = Slider(spox_time , 'Время', 1, 20, valinit=0, valstep=1, dragging=True)
        plt.show()

Plot = Plot()

def fulldesp(Q, E, T, t):
    sc = Phosfor
    Rp = sc.Rp(E)
    dRp = sc.dRp(E)
    dRpl = sc.dRpl(E)
    ni = sc.ni(T)
    n = 70
    xmax = 1
    ymax = 4
    dx = xmax / n
    dy = ymax / n
    y = np.empty(n)
    x = np.empty(n)
    for i in range(0, n):
        if i == 0:
            x[i] = 0
            y[i] = 0
        else:
            x[i] = x[i - 1] + dx
            y[i] = y[i - 1] + dy
    Plot.draw(0, n, x, y, dx, dy, xmax, T, Rp, dRp, dRpl, Q, ni, t)

fulldesp(Q, E, T, 20)