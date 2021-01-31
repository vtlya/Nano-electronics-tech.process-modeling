import matplotlib.pyplot as plt #Библиотека для работы с отображением данных
import numpy as np #Библиотека для работы с данными
import math #Библиотека для математических операций
from mpl_toolkits.mplot3d import Axes3D #Импорт инструмента для рисования в 3D
from scipy.special import erfc #Импорт функции ошибок
from matplotlib.widgets import Slider #слайдер для того, чтобы выбирать время
#Константы
q = 1.6e-19
k = 1.38e-23 / q

#Класс кремния, включает зависимости и константы спецефичные для него
class Si:

    def Eg(self, temperature):
        return 1.17 - 4.73e-4 / (temperature + 636) * temperature ** 2

    def Nv(self, temperature):
        return 4.82e15 * (self.m_dp ** 1.5) * (temperature ** 1.5)

    def Nc(self, temperature):
        return 4.82e15 * (self.m_dn ** 1.5) * (temperature ** 1.5)

    def ni(self, temperature):
        return ((self.Nc(temperature) * self.Nv(temperature)) ** 0.5) * math.exp(
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

#Создаем объект фосфора
Phosfor = Phosfor()

#Функция расчитывающая поток примеси через двумерную поверхность
def F(x, y, Q, dRp, Rp, dRpl, a):
    return Q / np.sqrt(2 * np.pi) / dRp / 1e-4 * np.exp(-(x - Rp) ** 2 / 2 / dRp ** 2) / 2 * (
                erfc((y - a) / np.sqrt(2) / dRpl) - erfc((y + a) / np.sqrt(2) / dRpl))


#Функция расчета коэфициента диффузии
def Dif(C, temperature, ni):
    #Составной D для фосфора
    D0 = 3.85*math.exp(-3.66/(k*temperature))
    D1 = 4.44*2*C/ni*math.exp(-4/(k*temperature))
    D2 = 44.2*math.exp(-4.37/(k*temperature))*((C*2/ni)**2)
    return D0+D1+D2

#Функция для расчета распределения с течением времени
def in_time(y, n, dx, T, ni):
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
        r[i] = -(((dx ** 2 * 1e-8) * y[i]) / (Dif(y[i], T, ni) * dt))
        b[i] = 1
        d[i] = 1
    for i in range(1, n, 1):
        delta[i] = -d[i] / (a[i] + b[i] * delta[i - 1])
        lyamda[i] = (r[i] - b[i] * lyamda[i - 1]) / (a[i] + b[i] * delta[i - 1])
    y[-1] = lyamda[-1]
    for i in range(n - 2, -1, -1):
       y[i] = delta[i] * y[i + 1] + lyamda[i]
    return y


#Класс рисующий в 3d пространсве с входными данными методолм draw
class Plot:
    Y = []
    X = []
    #Функция выполняющая расчет с задаными параметрами и отвечающая за отображение всей области графика
    def draw(self, start, end, x, dx, xmax, T, Rp, dRp, dRpl, Q, ni, t):
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
                Z[i][j] = F(X[i][j], Y[i][j], Q, dRp, Rp, dRpl, xmax / 2)
        fig = plt.figure('graph')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(0, xmax)
        ax.set_ylim(-2*xmax , 2*xmax)
        ax.set_zlim(0, 3e19)
        ax.set_xlabel('x, мкм')
        ax.set_ylabel('y, мкм')
        ax.set_zlabel('Концентрация, C')
        surface = ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color='red', linewidth=0.5, label='Загонка')
        surface2 = ax.plot_surface(X, Y, Z, cmap='winter', edgecolor='none')
        fig.suptitle('Имплантация фосфора P в кремнии')
        ax_time = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        Time = Slider(ax_time, 'Время', 1, 20, valinit=0,  valstep=1, dragging=True)
        colourbar = fig.colorbar(surface2, shrink=0.3, aspect=5, orientation='vertical', label='Разгонка')
        #Обновляет график при изменении значения слайдера
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
                    Z_r[:, i] = in_time(Z_r[:, i], end - 1, dx, T, ni)
                for i in range(0, len(Z[:, 1]), 1):
                    Z_r[i] = in_time(Z_r[i], end, dy, T, ni)
            surface2 = ax.plot_surface(X_r, Y_r, Z_r, cmap='winter', edgecolor='none')
            surface = ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color='red', linewidth=0.5, label='Загонка')
            ax.legend(title="Обозначения")
        #обновляет зграфик при изменении слайдера
        Time.on_changed(update)
        plt.show()

Plot = Plot()


#Объеденяющая функция задающая параметры загонки и разгонки примеси по входным данным (передаются в draw)
def implant(impurity, Q, E, T, t):
    if impurity == 'Phosfor':
        sc = Phosfor
    Rp = sc.Rp(E)
    dRp = sc.dRp(E)
    dRpl = sc.dRpl(E)
    g = sc.gamma(E)
    b = sc.betta(E)
    ni = sc.ni(T)
    print("Rp = %.4f " % Rp, "dRp = %.4f " % dRp, "dRpl = %.4f " % dRpl, "gamma = %.4f " % g, "betta = %.4f " % b)
    print("ni = " + str(sc.ni(T)))
    print('Пожалуйста, подождите. Моделирование процесса занимает какое-то время...')
    n = 40
    xmax = 1
    dx = xmax / n
    x = np.empty(n)
    for i in range(0, n):
        if i == 0:
            x[i] = 0
        else:
            x[i] = x[i - 1] + dx
    Plot.draw(0, n, x, dx, xmax, T, Rp, dRp, dRpl, Q, ni, t)

#Вызов функции для расчета
implant('Phosfor', 5e14, 180, 1371, 20)
